#pragma once

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <functional>
#include <vector>

#include "VirtualOrch/reduction/DenseTypes.h"
#include "VirtualOrch/TdTokens.h"

/**
 * The token stream of the anticipation-style models trained on 2026-10-20:
 *   T3''  piano (given, instrument 0)            -> reduction (generated, instrument 1)   dense 11776
 *   T8''  piano melody + bass (given, instr. 0)  -> reduction melody + bass (instr. 1)    dense 11776
 *   T7''  piano events + melody/bass CONTROLS    -> reduction minus melody/bass (instr. 1) AMT 55293
 *
 * Every stream is [time, duration, note, velocity] quads in onset order, in one of two alphabets:
 *   Vocab::Dense  event ids only: onset t, duration 10000 + d, note 11000 + 128 * instr + pitch,
 *                 velocity 11640 + v, TD 11768..11775. The context starts with the first quad.
 *   Vocab::Amt    the AMT-native vocab: the same event ids but velocity 55028 + v; a CONTROL quad is
 *                 ATIME 27513 + t, ADUR 37513 + d, ANOTE 38513 + 128 * instr + pitch, AVEL 55156 + v;
 *                 REST quads (t, 10000, 27512, 55028) and the ANTICIPATE flag (55027) first.
 *
 * Given notes (and controls) are never sampled. The event clock is the onset of the last generated
 * event. A pending given note is inserted once the clock has reached its ACTUAL time (onset - delta;
 * delta 0 for given events, 3 s for the T7'' melody controls); the transformer also inserts one when
 * a sampled onset lies at or after it and draws the onset again. TD tokens (one per 2 s boundary)
 * follow the first note quad, event or given, whose actual time reaches the boundary.
 *
 * All times are absolute app centiseconds; quads are written relative to baseCs (a multiple of the
 * TD boundary spacing). A held key is PROVISIONAL: its duration is frozen at the value it had when
 * it was inserted until the note-off confirms it (or the hold limit commits it); committedLen() is
 * the first token that may still change.
 *
 * W2/W3 (waitToken >= 0, vocab 11777): a WAIT token ("the soloist plays next") is written right
 * before the first quad of every new given onset group, and given notes within mergeGivenCs of a
 * group's first onset join that group (onset moved to it, end time kept), as in training.
 *
 * Late input (holdLate / reorderLate, Accompaniment): given notes behind the event clock are held
 * back and, at most once a second, the stream is rewound to the earliest of them and rewritten in
 * onset order with every note generated after it KEPT (WAIT and TD tokens placed again).
 */
class AnticipatedStream {
public:
    enum class Vocab { Dense, Amt };

    /** The instrument of the given notes (piano / melody+bass). */
    static constexpr int32_t kGivenInstrument = 0;
    static constexpr int32_t kControlInstrument = kGivenInstrument; // earlier name, kept for callers
    static constexpr int32_t kEventInstrument = 1;
    static constexpr int32_t kMaxModelTimeCs = 9999;
    /** Safety net: rebase when the clock is this far into the 0..9999 range (shift by kRebaseShiftCs). */
    static constexpr int32_t kRebaseAtCs = 9000;
    static constexpr int32_t kRebaseShiftCs = 6000;
    static constexpr int32_t kHoldCommitCs = 1000;
    /**
     * Eviction rebase (the models are trained on onsets up to ~30-40 s): when the cache evicts a
     * note whose onset is more than this far from the origin, the origin moves forward by the
     * largest multiple of this that keeps every retained note at onset >= 0.
     */
    static constexpr int32_t kEvictRebaseCs = 3000;

    static constexpr int32_t kAnticipate = 55027;
    static constexpr int32_t kRestId = 27512;
    static constexpr int32_t kAmtTimeCtl = 27513;
    static constexpr int32_t kAmtDurCtl = 37513;
    static constexpr int32_t kAmtNoteCtl = 38513;
    static constexpr int32_t kAmtVelCtl = 55156;
    static constexpr int32_t kAmtVelEvent = 55028;

    struct Note {
        int32_t onsetCs = 0;
        int32_t durationCs = 0;
        int32_t pitch = 60;
        int32_t velocity = 64;
    };

    /** How a pending note is written and when it is released. */
    struct ControlSpec {
        int32_t instrument = kGivenInstrument;
        int32_t deltaCs = 0;  // released when the clock reaches onset - deltaCs
        int32_t priority = 0; // ties in actual time: lower first
        bool controlToken = false; // AMT only: write the ATIME/ADUR/ANOTE/AVEL control ids
        int line = -1;        // >= 0: per-line monophony check (melody 0 / bass 1)
    };

    explicit AnticipatedStream(Vocab v = Vocab::Dense) : vocab(v) {}

    /** W2/W3: the WAIT id (11776); -1 = no WAIT tokens (T3'', T8'', T7''). */
    int32_t waitToken = -1;
    /** W2/W3: given notes within this of a group's first onset join the group (0 = off). */
    int32_t mergeGivenCs = 0;

    /** Called when a TD boundary is written; returns the TD token id for it. */
    std::function<int32_t()> tdTokenProvider = [] { return 0; };

    auto reset(int32_t nowCs) -> void {
        items.clear();
        pending.clear();
        late.clear();
        lastGivenGroupOnsetCs = INT32_MIN;
        lastEventPitchValue = -1;
        baseCs = (std::max(nowCs, 0) / TdTokens::kBoundaryCs) * TdTokens::kBoundaryCs;
        nextBoundaryCs = baseCs;
        clockCs = baseCs;
        lastEventOnsetCs = INT32_MIN;
        lastEndByLine = {INT32_MIN, INT32_MIN};
        started = true;
    }

    [[nodiscard]] auto isStarted() const -> bool { return started; }
    [[nodiscard]] auto base() const -> int32_t { return baseCs; }
    /** Onset of the last generated event (the base before the first). */
    [[nodiscard]] auto clock() const -> int32_t { return clockCs; }
    /** True when a generated event has been written at exactly this absolute onset. */
    [[nodiscard]] auto eventEmittedAt(int32_t onsetAbs) const -> bool {
        return lastEventOnsetCs == onsetAbs;
    }
    [[nodiscard]] auto vocabKind() const -> Vocab { return vocab; }
    [[nodiscard]] auto prefixTokens() const -> size_t { return vocab == Vocab::Amt ? 1u : 0u; }

    /** A note arrives (held = key still down: its duration is only provisional). */
    auto addControl(const Note &note, bool held, const ControlSpec &spec) -> void {
        Control c;
        c.note = note;
        c.originalOnsetCs = note.onsetCs;
        c.provisional = held;
        c.spec = spec;
        if (mergeGivenCs > 0 && isGivenNote(c) && mergeIntoGroup(c))
            return;
        pending.push_back(c);
        sortPending();
    }

    /**
     * Undo the generated notes from cutCs on (leaving Accompaniment). The stream is cut at the
     * first generated event (or AMT rest) with onset >= cutCs; given notes after that point go
     * back to pending (they are written again once the clock reaches them), TD tokens after it
     * are dropped and written again, and the clock, the last event onset and the per-line
     * monophony ends are recomputed from what is kept. Given notes for which `drop` returns true
     * are removed wherever they are (MelodyFirst: the controls of the melody notes rewound too).
     * Returns true when anything changed (the cache must then be rebuilt).
     */
    auto rewindTo(int32_t cutCs, const std::function<bool(const ControlSpec &, const Note &)> &drop = {})
        -> bool {
        const auto dropped = [&](const Control &c) { return drop && drop(c.spec, c.note); };
        size_t cut = items.size();
        for (size_t i = 0; i < items.size(); ++i)
            if ((items[i].kind == Kind::Event || items[i].kind == Kind::Rest)
                && items[i].control.note.onsetCs >= cutCs) {
                cut = i;
                break;
            }
        bool changed = cut < items.size();
        bool droppedBoundary = false;
        int32_t firstDroppedBoundary = 0;
        for (size_t i = cut; i < items.size(); ++i) {
            const auto &item = items[i];
            if (item.kind == Kind::Control && ! dropped(item.control)) {
                auto c = item.control;
                c.trimmed = false;
                pending.push_back(c);
            } else if (item.kind == Kind::Td && ! droppedBoundary) {
                droppedBoundary = true;
                firstDroppedBoundary = item.tdBoundaryCs;
            }
        }
        items.erase(items.begin() + static_cast<std::ptrdiff_t>(cut), items.end());

        const size_t itemsBefore = items.size();
        items.erase(std::remove_if(items.begin(), items.end(),
                                   [&](const Item &i) { return i.kind == Kind::Control && dropped(i.control); }),
                    items.end());
        for (auto &c: late) // held-back late notes go back to pending: the clock re-inserts them
            pending.push_back(c);
        late.clear();
        const size_t pendingBefore = pending.size();
        pending.erase(std::remove_if(pending.begin(), pending.end(), dropped), pending.end());
        changed = changed || items.size() != itemsBefore || pending.size() != pendingBefore;
        sortPending();

        if (droppedBoundary)
            nextBoundaryCs = firstDroppedBoundary;
        // a WAIT left at the end belonged to a group that went back to pending
        while (! items.empty() && items.back().kind == Kind::Wait)
            items.pop_back();
        clockCs = baseCs;
        lastEventOnsetCs = INT32_MIN;
        lastEventPitchValue = -1;
        lastEndByLine = {INT32_MIN, INT32_MIN};
        lastGivenGroupOnsetCs = INT32_MIN;
        for (const auto &item: items) {
            const auto &n = item.control.note;
            if (item.kind == Kind::Control && isGivenNote(item.control))
                lastGivenGroupOnsetCs = n.onsetCs;
            if (item.kind == Kind::Event) {
                clockCs = std::max(clockCs, n.onsetCs);
                lastEventOnsetCs = n.onsetCs;
                lastEventPitchValue = n.pitch;
            } else if (item.kind == Kind::Rest) {
                clockCs = std::max(clockCs, n.onsetCs);
            } else if (item.kind == Kind::Control && item.control.spec.line >= 0) {
                auto &end = lastEndByLine[static_cast<size_t>(item.control.spec.line)];
                end = std::max(end, n.onsetCs + n.durationCs);
            }
        }
        return changed;
    }

    auto addControl(const Note &note, bool held, int32_t instrument = kGivenInstrument,
                    int32_t deltaCs = 0, int32_t priority = 0) -> void {
        ControlSpec spec;
        spec.instrument = instrument;
        spec.deltaCs = deltaCs;
        spec.priority = priority;
        addControl(note, held, spec);
    }

    /** True when a pending note would be inserted at event clock `clockCsNow`. */
    [[nodiscard]] auto hasPendingDueBy(int32_t clockCsNow) const -> bool {
        return ! pending.empty() && pending.front().actualCs() <= clockCsNow;
    }

    /** Actual time of the next pending note, or INT32_MAX. */
    [[nodiscard]] auto nextPendingActual() const -> int32_t {
        return pending.empty() ? INT32_MAX : pending.front().actualCs();
    }

    /** Note-off: the held note gets its final duration (pending or already inserted). */
    auto confirmControl(int32_t onsetCs, int32_t pitch, int32_t finalDurationCs,
                        int32_t instrument = kGivenInstrument) -> bool {
        // Only a HELD note is confirmed (an AMT melody control with the same onset and pitch is not).
        // Matched on the onset the note arrived with (a merged note's onset moved to its group);
        // a merged note keeps its end time.
        const auto confirm = [&](Control &c) {
            c.note.durationCs = std::max(0, c.originalOnsetCs + finalDurationCs - c.note.onsetCs);
            c.provisional = false;
        };
        const auto matches = [&](const Control &c) {
            return c.provisional && c.originalOnsetCs == onsetCs && c.note.pitch == pitch
                   && c.spec.instrument == instrument;
        };
        for (auto &c: pending)
            if (matches(c)) {
                confirm(c);
                return true;
            }
        for (auto &c: late)
            if (matches(c)) {
                confirm(c);
                return true;
            }
        for (auto it = items.rbegin(); it != items.rend(); ++it)
            if (it->kind == Kind::Control && matches(it->control)) {
                confirm(it->control);
                return true;
            }
        return false;
    }

    /** A held key past the hold limit is committed with the duration it was written with. */
    auto commitLongHolds(int32_t nowCs) -> void {
        for (auto &item: items)
            if (item.kind == Kind::Control && item.control.provisional
                && nowCs - item.control.note.onsetCs > kHoldCommitCs)
                item.control.provisional = false;
        for (auto &c: pending)
            if (c.provisional && nowCs - c.note.onsetCs > kHoldCommitCs)
                c.provisional = false;
        for (auto &c: late)
            if (c.provisional && nowCs - c.note.onsetCs > kHoldCommitCs)
                c.provisional = false;
    }

    // ---- late input (Accompaniment) ----

    /**
     * Move the pending given notes that can no longer be appended in onset order (onset before
     * the last note already written, given or generated) to the late queue.
     */
    auto holdLate() -> void {
        int32_t lastWritten = INT32_MIN;
        for (const auto &item: items)
            if (item.kind == Kind::Event || item.kind == Kind::Control)
                lastWritten = std::max(lastWritten, item.control.note.onsetCs);
        if (lastWritten == INT32_MIN)
            return;
        for (auto it = pending.begin(); it != pending.end();) {
            if (isGivenNote(*it) && it->actualCs() < lastWritten) {
                late.push_back(*it);
                it = pending.erase(it);
            } else {
                ++it;
            }
        }
    }
    [[nodiscard]] auto hasLate() const -> bool { return ! late.empty(); }
    [[nodiscard]] auto lateCount() const -> size_t { return late.size(); }

    /**
     * Rewind to the earliest late note and rewrite everything from there in onset order: the late
     * notes plus the given and generated notes already written after that point (generated notes
     * are kept). Ties: given first unless `generatedFirstAtTies`; ascending pitch within each.
     * WAIT and TD tokens are placed again. The event clock does not change. Returns true when the
     * stream changed (the KV cache sees the new prefix and rebuilds from it).
     */
    auto reorderLate(int32_t nowCs, bool generatedFirstAtTies = false) -> bool {
        if (late.empty())
            return false;
        int32_t cutOnset = INT32_MAX;
        for (auto &c: late) {
            freezeProvisional(c, nowCs);
            cutOnset = std::min(cutOnset, c.note.onsetCs);
        }
        size_t cut = items.size();
        for (size_t i = 0; i < items.size(); ++i)
            if (items[i].kind != Kind::Td && items[i].control.note.onsetCs >= cutOnset) {
                cut = i;
                break;
            }
        // the WAIT of a group that starts at the cut is rewritten too
        while (cut > 0 && items[cut - 1].kind == Kind::Wait)
            --cut;

        std::vector<Item> notes;
        bool droppedBoundary = false;
        int32_t firstDroppedBoundary = 0;
        for (size_t i = cut; i < items.size(); ++i) {
            if (items[i].kind == Kind::Event || items[i].kind == Kind::Control)
                notes.push_back(items[i]);
            else if (items[i].kind == Kind::Td && ! droppedBoundary) {
                droppedBoundary = true;
                firstDroppedBoundary = items[i].tdBoundaryCs;
            }
        }
        for (const auto &c: late) {
            Item item;
            item.kind = Kind::Control;
            item.control = c;
            notes.push_back(item);
        }
        late.clear();
        std::stable_sort(notes.begin(), notes.end(), [generatedFirstAtTies](const Item &a, const Item &b) {
            if (a.control.note.onsetCs != b.control.note.onsetCs)
                return a.control.note.onsetCs < b.control.note.onsetCs;
            if (a.kind != b.kind) {
                const bool aGiven = a.kind == Kind::Control;
                return generatedFirstAtTies ? ! aGiven : aGiven;
            }
            return a.control.note.pitch < b.control.note.pitch;
        });

        items.erase(items.begin() + static_cast<std::ptrdiff_t>(cut), items.end());
        if (droppedBoundary)
            nextBoundaryCs = firstDroppedBoundary;
        lastGivenGroupOnsetCs = INT32_MIN;
        for (const auto &item: items)
            if (item.kind == Kind::Control && isGivenNote(item.control))
                lastGivenGroupOnsetCs = item.control.note.onsetCs;
        for (const auto &item: notes) {
            if (item.kind == Kind::Control)
                writeWaitBeforeGroup(item.control);
            items.push_back(item);
            if (item.kind == Kind::Control && isGivenNote(item.control))
                lastGivenGroupOnsetCs = item.control.note.onsetCs;
            if (item.kind == Kind::Event)
                lastEventPitchValue = item.control.note.pitch;
            writeBoundariesUpTo(item.kind == Kind::Control ? item.control.actualCs()
                                                           : item.control.note.onsetCs);
        }
        return true;
    }

    // ---- W2/W3 WAIT ----

    /** A given group can be written now (pending or held back as late). */
    [[nodiscard]] auto hasGivenGroup() const -> bool {
        if (! late.empty())
            return true;
        return std::any_of(pending.begin(), pending.end(), [this](const Control &c) { return isGivenNote(c); });
    }

    /**
     * WAIT resolved: write the next given onset group (late ones first), each note no earlier
     * than `atCs` (the event clock: the soloist "plays next"; a moved note keeps its end time).
     * Returns how many notes were written.
     */
    auto insertNextGroupAt(int32_t atCs, int32_t nowCs) -> int {
        auto &from = ! late.empty() ? late : pending;
        std::sort(from.begin(), from.end(), [](const Control &a, const Control &b) {
            return a.note.onsetCs != b.note.onsetCs ? a.note.onsetCs < b.note.onsetCs
                                                    : a.note.pitch < b.note.pitch;
        });
        auto first = std::find_if(from.begin(), from.end(), [this](const Control &c) { return isGivenNote(c); });
        if (first == from.end())
            return 0;
        const int32_t groupOnset = first->note.onsetCs;
        std::vector<Control> group;
        for (auto it = from.begin(); it != from.end();) {
            if (isGivenNote(*it) && it->note.onsetCs == groupOnset) {
                group.push_back(*it);
                it = from.erase(it);
            } else {
                ++it;
            }
        }
        if (&from == &pending)
            sortPending();
        for (auto &c: group)
            if (c.note.onsetCs < atCs) {
                freezeProvisional(c, nowCs);
                const int32_t end = c.note.onsetCs + c.note.durationCs;
                c.note.onsetCs = atCs;
                c.note.durationCs = std::max(0, end - atCs);
            }
        return writeBatch(std::move(group), nowCs);
    }

    /** Onset of the last given onset group written (INT32_MIN before the first). */
    [[nodiscard]] auto lastGivenGroupOnset() const -> int32_t { return lastGivenGroupOnsetCs; }

    /** Pitch of the last generated event (-1 before the first). */
    [[nodiscard]] auto lastEventPitch() const -> int32_t { return lastEventPitchValue; }

    /**
     * Insert every pending note whose actual time is <= clockCsNow (< when `inclusive` is false:
     * the T3'' rule writes the first generated note at a shared onset BEFORE the given notes
     * there). Returns how many were inserted.
     */
    auto insertDue(int32_t clockCsNow, int32_t nowCs, bool inclusive = true) -> int {
        std::vector<Control> batch;
        while (! pending.empty()
               && (inclusive ? pending.front().actualCs() <= clockCsNow
                             : pending.front().actualCs() < clockCsNow)) {
            batch.push_back(pending.front());
            pending.erase(pending.begin());
        }
        return writeBatch(std::move(batch), nowCs);
    }

    /**
     * Insert only the next pending note (the transformer then draws the onset again). With
     * `givenOnly` the AMT melody controls are skipped: they are released by the clock alone.
     */
    auto insertNext(int32_t nowCs, bool givenOnly = false) -> bool {
        for (auto it = pending.begin(); it != pending.end(); ++it) {
            if (givenOnly && it->spec.controlToken)
                continue;
            std::vector<Control> batch{*it};
            pending.erase(it);
            return writeBatch(std::move(batch), nowCs) > 0;
        }
        return false;
    }

    /** Actual time of the next pending GIVEN note (not an AMT control), or INT32_MAX. */
    [[nodiscard]] auto nextPendingGivenActual() const -> int32_t {
        for (const auto &c: pending)
            if (! c.spec.controlToken)
                return c.actualCs();
        return INT32_MAX;
    }

    auto appendEvent(const Note &note) -> void {
        Item item;
        item.kind = Kind::Event;
        item.control.note = note;
        item.control.spec.instrument = kEventInstrument;
        items.push_back(item);
        clockCs = note.onsetCs;
        lastEventOnsetCs = note.onsetCs;
        lastEventPitchValue = note.pitch;
        writeBoundariesUpTo(note.onsetCs);
    }

    /** AMT: a visible rest quad at r (the clock moves to it). */
    auto appendRest(int32_t restCs) -> void {
        Item item;
        item.kind = Kind::Rest;
        item.control.note.onsetCs = restCs;
        items.push_back(item);
        clockCs = std::max(clockCs, restCs);
    }

    /** The shift the stream wants before sampling an event at or after lowerBoundCs. */
    [[nodiscard]] auto needsRebase(int32_t lowerBoundCs) const -> bool {
        return started && std::max(clockCs, lowerBoundCs) - baseCs >= kRebaseAtCs;
    }

    /** Safety net: move the origin forward; older items are dropped and the cache is rebuilt. */
    auto rebase() -> void {
        baseCs += kRebaseShiftCs;
        items.erase(std::remove_if(items.begin(), items.end(),
                                   [this](const Item &i) { return itemTime(i) < baseCs; }),
                    items.end());
        pending.erase(std::remove_if(pending.begin(), pending.end(),
                                     [this](const Control &c) { return c.note.onsetCs < baseCs; }),
                      pending.end());
        late.erase(std::remove_if(late.begin(), late.end(),
                                  [this](const Control &c) { return c.note.onsetCs < baseCs; }),
                   late.end());
        nextBoundaryCs = std::max(nextBoundaryCs, baseCs);
        clockCs = std::max(clockCs, baseCs);
    }

    /**
     * The cache evicted `evictedTokens` tokens after its `sinkTokens` sink rows. When one of the
     * evicted notes lies more than kEvictRebaseCs after the origin, move the origin forward by
     * k = the largest multiple of kEvictRebaseCs that keeps every retained note at onset >= 0
     * (so a multiple of the TD boundary spacing too), and keep only the retained window:
     *   - everything evicted, and the old sink quad, are dropped (their times would go negative);
     *     the first retained quad becomes the new sink, because the cache is rebuilt from the
     *     retained tokens by a full prefill (fresh RoPE positions). The ANTICIPATE flag of the AMT
     *     vocab is the mode token and always stays the first token;
     *   - leading TD tokens are skipped so the window starts on a quad;
     *   - pending notes older than the new origin are dropped.
     * Every note keeps its absolute onset; only the origin (baseCs) and the written times change.
     * Returns k (0 = nothing happened).
     */
    auto rebaseForEviction(size_t evictedTokens, size_t sinkTokens) -> int32_t {
        if (! started || evictedTokens == 0)
            return 0;
        const size_t evictedBegin = sinkTokens;
        const size_t evictedEnd = sinkTokens + evictedTokens;

        int32_t lastEvictedOnset = INT32_MIN;
        size_t tokenIndex = prefixTokens();
        size_t firstRetained = items.size();
        for (size_t i = 0; i < items.size(); ++i) {
            const size_t begin = tokenIndex;
            const size_t end = begin + tokenCount(items[i]);
            tokenIndex = end;
            if (begin >= evictedEnd) {
                firstRetained = i;
                break;
            }
            if (end > evictedBegin && items[i].kind != Kind::Td && items[i].kind != Kind::Wait)
                lastEvictedOnset = std::max(lastEvictedOnset, items[i].control.note.onsetCs);
        }
        if (lastEvictedOnset == INT32_MIN || lastEvictedOnset - baseCs <= kEvictRebaseCs)
            return 0;
        while (firstRetained < items.size()
               && (items[firstRetained].kind == Kind::Td || items[firstRetained].kind == Kind::Wait))
            ++firstRetained;
        if (firstRetained >= items.size())
            return 0;

        int32_t minOnset = INT32_MAX;
        for (size_t i = firstRetained; i < items.size(); ++i)
            if (items[i].kind != Kind::Td && items[i].kind != Kind::Wait)
                minOnset = std::min(minOnset, items[i].control.note.onsetCs);
        const int32_t k = ((minOnset - baseCs) / kEvictRebaseCs) * kEvictRebaseCs;
        if (k <= 0)
            return 0;

        items.erase(items.begin(), items.begin() + static_cast<std::ptrdiff_t>(firstRetained));
        baseCs += k;
        pending.erase(std::remove_if(pending.begin(), pending.end(),
                                     [this](const Control &c) { return c.note.onsetCs < baseCs; }),
                      pending.end());
        late.erase(std::remove_if(late.begin(), late.end(),
                                  [this](const Control &c) { return c.note.onsetCs < baseCs; }),
                   late.end());
        nextBoundaryCs = std::max(nextBoundaryCs, baseCs);
        clockCs = std::max(clockCs, baseCs);
        return k;
    }

    [[nodiscard]] auto tokens() const -> std::vector<int32_t> {
        std::vector<int32_t> out;
        out.reserve(prefixTokens() + items.size() * 4);
        if (vocab == Vocab::Amt)
            out.push_back(kAnticipate);
        for (const auto &item: items)
            appendTokens(out, item);
        return out;
    }

    /** Tokens before the first one that can still change (a held note's quad). */
    [[nodiscard]] auto committedLen() const -> size_t {
        size_t n = prefixTokens();
        for (const auto &item: items) {
            if (item.kind == Kind::Control && item.control.provisional)
                return n;
            n += tokenCount(item);
        }
        return n;
    }

    [[nodiscard]] auto pendingCount() const -> size_t { return pending.size(); }
    [[nodiscard]] auto itemCount() const -> size_t { return items.size(); }
    /** Nothing has been written yet (the model has no context to condition on). */
    [[nodiscard]] auto isEmpty() const -> bool { return items.empty(); }

    /** Token ids of a dense quad for these values (used while sampling an event field by field). */
    [[nodiscard]] static auto quadFor(const Note &note, int32_t instrument, int32_t baseCsIn)
        -> std::vector<int32_t> {
        const int32_t time = std::clamp(note.onsetCs - baseCsIn, 0, kMaxModelTimeCs);
        const int32_t dur = DenseQuantize::snapDurationCs(note.durationCs);
        const int32_t pitch = DenseQuantize::snapPianoPitch(note.pitch);
        const int32_t vel = DenseQuantize::snapVelocity(note.velocity);
        return {time,
                static_cast<int32_t>(DenseVocab::DurOffset) + dur,
                static_cast<int32_t>(DenseVocab::NoteOffset) + DenseConfig::MaxPitch * instrument
                    + pitch,
                static_cast<int32_t>(DenseVocab::VelocityOffset) + vel};
    }

private:
    enum class Kind { Control, Event, Td, Rest, Wait };
    struct Control {
        Note note;
        int32_t originalOnsetCs = 0; // the onset it arrived with (note-offs match on it)
        bool provisional = false;
        ControlSpec spec;
        bool trimmed = false; // shortened by the per-line monophony check
        [[nodiscard]] auto actualCs() const -> int32_t { return note.onsetCs - spec.deltaCs; }
    };
    struct Item {
        Kind kind = Kind::Event;
        Control control;
        int32_t tdBoundaryCs = 0;
        int32_t tdToken = 0;
    };

    auto sortPending() -> void {
        std::stable_sort(pending.begin(), pending.end(), [](const Control &a, const Control &b) {
            if (a.actualCs() != b.actualCs())
                return a.actualCs() < b.actualCs();
            if (a.spec.priority != b.spec.priority)
                return a.spec.priority < b.spec.priority;
            if (a.note.onsetCs != b.note.onsetCs)
                return a.note.onsetCs < b.note.onsetCs;
            return a.note.pitch < b.note.pitch;
        });
    }

    [[nodiscard]] static auto tokenCount(const Item &i) -> size_t {
        return i.kind == Kind::Td || i.kind == Kind::Wait ? 1u : 4u;
    }

    [[nodiscard]] static auto isGivenNote(const Control &c) -> bool {
        return c.spec.instrument == kGivenInstrument && ! c.spec.controlToken;
    }

    static auto freezeProvisional(Control &c, int32_t nowCs) -> void {
        if (c.provisional)
            c.note.durationCs = std::min(std::max(nowCs - c.note.onsetCs, 0), DenseQuantize::DurationMaxCs);
    }

    /** WAIT before the first quad of a new given onset group (W2/W3). */
    auto writeWaitBeforeGroup(const Control &c) -> void {
        if (waitToken < 0 || ! isGivenNote(c) || c.note.onsetCs == lastGivenGroupOnsetCs)
            return;
        Item wait;
        wait.kind = Kind::Wait;
        wait.control.note.onsetCs = c.note.onsetCs;
        items.push_back(wait);
    }

    /**
     * Join the group whose first onset is the latest one at most mergeGivenCs before c (pending,
     * late or written): the onset moves to it and the end time is kept. A repeated pitch in the
     * group keeps the longer copy. Returns true when c was absorbed by an existing copy.
     */
    auto mergeIntoGroup(Control &c) -> bool {
        int32_t group = INT32_MIN;
        const auto consider = [&](const Control &o) {
            if (isGivenNote(o) && o.note.onsetCs <= c.note.onsetCs
                && c.note.onsetCs - o.note.onsetCs <= mergeGivenCs)
                group = std::max(group, o.note.onsetCs);
        };
        for (const auto &o: pending)
            consider(o);
        for (const auto &o: late)
            consider(o);
        for (const auto &item: items)
            if (item.kind == Kind::Control)
                consider(item.control);
        if (group == INT32_MIN || group == c.note.onsetCs)
            return absorbRepeat(c);
        const int32_t end = c.note.onsetCs + c.note.durationCs;
        c.note.onsetCs = group;
        c.note.durationCs = c.provisional ? 0 : std::max(0, end - group);
        return absorbRepeat(c);
    }

    /** Same onset and pitch already pending / late: keep the longer copy, drop c. */
    auto absorbRepeat(const Control &c) -> bool {
        for (auto *list: {&pending, &late})
            for (auto &o: *list)
                if (isGivenNote(o) && o.note.onsetCs == c.note.onsetCs && o.note.pitch == c.note.pitch) {
                    if (! c.provisional && (o.provisional || c.note.durationCs > o.note.durationCs)) {
                        o.note.durationCs = c.note.durationCs;
                        o.provisional = false;
                    }
                    return true;
                }
        return false;
    }

    [[nodiscard]] static auto itemTime(const Item &i) -> int32_t {
        return i.kind == Kind::Td ? i.tdBoundaryCs : i.control.note.onsetCs;
    }

    auto writeBatch(std::vector<Control> batch, int32_t nowCs) -> int {
        int written = 0;
        std::vector<size_t> batchItems;
        for (auto &c: batch) {
            if (c.provisional) // frozen at what it has now
                c.note.durationCs = std::min(std::max(nowCs - c.note.onsetCs, 0),
                                             DenseQuantize::DurationMaxCs);
            const int line = c.spec.line;
            if (line >= 0) {
                // Per-line monophony: the newcomer must start at or after the previous end.
                if (lastEndByLine[static_cast<size_t>(line)] > c.note.onsetCs) {
                    bool shortened = false;
                    for (auto it = batchItems.rbegin(); it != batchItems.rend(); ++it) {
                        auto &prev = items[*it].control;
                        if (prev.spec.line != line)
                            continue;
                        prev.note.durationCs = std::max(0, c.note.onsetCs - prev.note.onsetCs);
                        prev.trimmed = true;
                        shortened = true;
                        break;
                    }
                    if (! shortened)
                        continue; // the previous one is already in the stream: skip the newcomer
                }
                lastEndByLine[static_cast<size_t>(line)] = c.note.onsetCs + c.note.durationCs;
            }
            Item item;
            item.kind = Kind::Control;
            item.control = c;
            writeWaitBeforeGroup(c);
            items.push_back(item);
            if (isGivenNote(c))
                lastGivenGroupOnsetCs = c.note.onsetCs;
            batchItems.push_back(items.size() - 1);
            ++written;
            writeBoundariesUpTo(c.actualCs()); // TD right after this quad
        }
        // drop batch notes that the monophony check shortened to nothing
        const auto before = items.size();
        items.erase(std::remove_if(items.begin(), items.end(),
                                   [](const Item &i) {
                                       return i.kind == Kind::Control && i.control.trimmed
                                              && i.control.note.durationCs < DenseQuantize::DurationMinCs;
                                   }),
                    items.end());
        written -= static_cast<int>(before - items.size());
        return written;
    }

    auto writeBoundariesUpTo(int32_t actualCs) -> void {
        while (nextBoundaryCs <= actualCs) {
            Item td;
            td.kind = Kind::Td;
            td.tdBoundaryCs = nextBoundaryCs;
            td.tdToken = tdTokenProvider();
            items.push_back(td);
            nextBoundaryCs += TdTokens::kBoundaryCs;
        }
    }

    auto appendTokens(std::vector<int32_t> &out, const Item &item) const -> void {
        if (item.kind == Kind::Td) {
            out.push_back(item.tdToken);
            return;
        }
        if (item.kind == Kind::Wait) {
            out.push_back(waitToken);
            return;
        }
        const auto &n = item.control.note;
        const int32_t time = std::clamp(n.onsetCs - baseCs, 0, kMaxModelTimeCs);
        const int32_t dur = DenseQuantize::snapDurationCs(n.durationCs);
        const int32_t pitch = DenseQuantize::snapPianoPitch(n.pitch);
        const int32_t vel = DenseQuantize::snapVelocity(n.velocity);
        const auto note = [&](int32_t instrument) {
            return static_cast<int32_t>(DenseVocab::NoteOffset)
                   + DenseConfig::MaxPitch * instrument + pitch;
        };
        if (item.kind == Kind::Rest) {
            out.insert(out.end(), {time, static_cast<int32_t>(DenseVocab::DurOffset), kRestId,
                                   kAmtVelEvent});
            return;
        }
        const int32_t instrument =
            item.kind == Kind::Event ? kEventInstrument : item.control.spec.instrument;
        if (vocab == Vocab::Amt && item.kind == Kind::Control && item.control.spec.controlToken) {
            out.insert(out.end(), {kAmtTimeCtl + time, kAmtDurCtl + dur,
                                   kAmtNoteCtl + DenseConfig::MaxPitch * instrument + pitch,
                                   kAmtVelCtl + vel});
            return;
        }
        const int32_t velocityId = vocab == Vocab::Amt
                                       ? kAmtVelEvent + vel
                                       : static_cast<int32_t>(DenseVocab::VelocityOffset) + vel;
        out.insert(out.end(), {time, static_cast<int32_t>(DenseVocab::DurOffset) + dur,
                               note(instrument), velocityId});
    }

    Vocab vocab;
    int32_t baseCs = 0;
    int32_t nextBoundaryCs = 0;
    int32_t clockCs = 0;
    int32_t lastEventOnsetCs = INT32_MIN;
    std::array<int32_t, 2> lastEndByLine{INT32_MIN, INT32_MIN};
    bool started = false;
    std::vector<Item> items;
    std::vector<Control> pending;
    std::vector<Control> late;           // given notes behind the clock (Accompaniment reorder)
    int32_t lastGivenGroupOnsetCs = INT32_MIN;
    int32_t lastEventPitchValue = -1;
};
