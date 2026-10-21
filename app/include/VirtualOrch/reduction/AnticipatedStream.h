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

    /** Called when a TD boundary is written; returns the TD token id for it. */
    std::function<int32_t()> tdTokenProvider = [] { return 0; };

    auto reset(int32_t nowCs) -> void {
        items.clear();
        pending.clear();
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
        c.provisional = held;
        c.spec = spec;
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
        const size_t pendingBefore = pending.size();
        pending.erase(std::remove_if(pending.begin(), pending.end(), dropped), pending.end());
        changed = changed || items.size() != itemsBefore || pending.size() != pendingBefore;
        sortPending();

        if (droppedBoundary)
            nextBoundaryCs = firstDroppedBoundary;
        clockCs = baseCs;
        lastEventOnsetCs = INT32_MIN;
        lastEndByLine = {INT32_MIN, INT32_MIN};
        for (const auto &item: items) {
            const auto &n = item.control.note;
            if (item.kind == Kind::Event) {
                clockCs = std::max(clockCs, n.onsetCs);
                lastEventOnsetCs = n.onsetCs;
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
        for (auto &c: pending)
            if (c.provisional && c.note.onsetCs == onsetCs && c.note.pitch == pitch
                && c.spec.instrument == instrument) {
                c.note.durationCs = finalDurationCs;
                c.provisional = false;
                return true;
            }
        for (auto it = items.rbegin(); it != items.rend(); ++it)
            if (it->kind == Kind::Control && it->control.provisional
                && it->control.note.onsetCs == onsetCs && it->control.note.pitch == pitch
                && it->control.spec.instrument == instrument) {
                it->control.note.durationCs = finalDurationCs;
                it->control.provisional = false;
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
    }

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
            const size_t end = begin + (items[i].kind == Kind::Td ? 1u : 4u);
            tokenIndex = end;
            if (begin >= evictedEnd) {
                firstRetained = i;
                break;
            }
            if (end > evictedBegin && items[i].kind != Kind::Td)
                lastEvictedOnset = std::max(lastEvictedOnset, items[i].control.note.onsetCs);
        }
        if (lastEvictedOnset == INT32_MIN || lastEvictedOnset - baseCs <= kEvictRebaseCs)
            return 0;
        while (firstRetained < items.size() && items[firstRetained].kind == Kind::Td)
            ++firstRetained;
        if (firstRetained >= items.size())
            return 0;

        int32_t minOnset = INT32_MAX;
        for (size_t i = firstRetained; i < items.size(); ++i)
            if (items[i].kind != Kind::Td)
                minOnset = std::min(minOnset, items[i].control.note.onsetCs);
        const int32_t k = ((minOnset - baseCs) / kEvictRebaseCs) * kEvictRebaseCs;
        if (k <= 0)
            return 0;

        items.erase(items.begin(), items.begin() + static_cast<std::ptrdiff_t>(firstRetained));
        baseCs += k;
        pending.erase(std::remove_if(pending.begin(), pending.end(),
                                     [this](const Control &c) { return c.note.onsetCs < baseCs; }),
                      pending.end());
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
            n += item.kind == Kind::Td ? 1 : 4;
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
    enum class Kind { Control, Event, Td, Rest };
    struct Control {
        Note note;
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
            items.push_back(item);
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
};
