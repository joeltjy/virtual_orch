#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

#include "VirtualOrch/reduction/DenseTypes.h"
#include "VirtualOrch/TdTokens.h"

/**
 * The token stream of the anticipation-style dense models (maestro_rope_td_T3 export):
 *
 *   [time, duration, note, velocity] quads in the order the model saw them in training.
 *   - Events are the model's own reduction notes (instrument 1), sampled one quad at a time.
 *   - Controls are the user's piano notes, written as instrument 3 quads. They are never
 *     sampled: a control is inserted once the event clock has passed its onset ("past due").
 *   - Event clock = onset of the last generated event (LastEventClock, no rest tokens).
 *   - TD tokens (one per 2 s boundary) follow the first quad, event or control, whose onset
 *     reaches the boundary; a quad crossing several boundaries is followed by all of them.
 *
 * All times here are absolute app centiseconds; the stream writes them relative to baseCs
 * (a multiple of the TD boundary spacing), so the 0..9999 cs onset range covers 100 s.
 * A held key is a PROVISIONAL control: its duration is frozen at the value it had when it was
 * inserted until the note-off confirms it (or the hold limit commits it). committedLen() is the
 * first token that may still change.
 */
class AnticipatedStream {
public:
    static constexpr int32_t kControlInstrument = 3;
    static constexpr int32_t kEventInstrument = 1;
    static constexpr int32_t kMaxModelTimeCs = 9999;
    /** Rebase when the clock gets this far into the 0..9999 range (and shift by kRebaseShiftCs). */
    static constexpr int32_t kRebaseAtCs = 9000;
    static constexpr int32_t kRebaseShiftCs = 6000;
    static constexpr int32_t kHoldCommitCs = 1000;

    struct Note {
        int32_t onsetCs = 0;
        int32_t durationCs = 0;
        int32_t pitch = 60;
        int32_t velocity = 64;
    };

    /** Called when a TD boundary is written; returns the TD token id for it. */
    std::function<int32_t()> tdTokenProvider = [] { return 0; };

    auto reset(int32_t nowCs) -> void {
        items.clear();
        pending.clear();
        baseCs = (std::max(nowCs, 0) / TdTokens::kBoundaryCs) * TdTokens::kBoundaryCs;
        nextBoundaryCs = baseCs;
        clockCs = baseCs;
        started = true;
    }

    [[nodiscard]] auto isStarted() const -> bool { return started; }
    [[nodiscard]] auto base() const -> int32_t { return baseCs; }
    /** Onset of the last generated event (the base before the first). */
    [[nodiscard]] auto clock() const -> int32_t { return clockCs; }

    /**
     * A control note arrives (held = key still down: duration is only provisional). Its actual
     * time is onset - deltaCs; controls are inserted in actual-time order, ties to the lower
     * priority number (the double-anticipation melody has priority 0, the piano 1).
     */
    auto addControl(const Note &note, bool held, int32_t instrument = kControlInstrument,
                    int32_t deltaCs = 0, int32_t priority = 0) -> void {
        Control c;
        c.note = note;
        c.provisional = held;
        c.instrument = instrument;
        c.deltaCs = deltaCs;
        c.priority = priority;
        pending.push_back(c);
        std::stable_sort(pending.begin(), pending.end(), [](const Control &a, const Control &b) {
            if (a.actualCs() != b.actualCs())
                return a.actualCs() < b.actualCs();
            if (a.priority != b.priority)
                return a.priority < b.priority;
            if (a.note.onsetCs != b.note.onsetCs)
                return a.note.onsetCs < b.note.onsetCs;
            return a.note.pitch < b.note.pitch;
        });
    }

    /** True when a pending control would be inserted at event clock `clockCsNow`. */
    [[nodiscard]] auto hasPendingDueBy(int32_t clockCsNow) const -> bool {
        return ! pending.empty() && pending.front().actualCs() <= clockCsNow;
    }

    /** Note-off: the held note gets its final duration (pending or already inserted). */
    auto confirmControl(int32_t onsetCs, int32_t pitch, int32_t finalDurationCs,
                        int32_t instrument = kControlInstrument) -> bool {
        for (auto &c: pending)
            if (c.note.onsetCs == onsetCs && c.note.pitch == pitch && c.instrument == instrument) {
                c.note.durationCs = finalDurationCs;
                c.provisional = false;
                return true;
            }
        for (auto it = items.rbegin(); it != items.rend(); ++it)
            if (it->kind == Kind::Control && it->control.note.onsetCs == onsetCs
                && it->control.note.pitch == pitch && it->control.instrument == instrument) {
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

    /** Insert every pending control with onset <= clockCsNow, in onset order. */
    auto insertDue(int32_t clockCsNow, int32_t nowCs) -> int {
        int inserted = 0;
        while (! pending.empty() && pending.front().actualCs() <= clockCsNow) {
            Control c = pending.front();
            pending.erase(pending.begin());
            if (c.provisional) // frozen at what it has now
                c.note.durationCs = std::min(std::max(nowCs - c.note.onsetCs, 0),
                                             DenseQuantize::DurationMaxCs);
            Item item;
            item.kind = Kind::Control;
            item.control = c;
            items.push_back(item);
            writeBoundariesUpTo(c.actualCs());
            ++inserted;
        }
        return inserted;
    }

    auto appendEvent(const Note &note) -> void {
        Item item;
        item.kind = Kind::Event;
        item.control.note = note;
        items.push_back(item);
        clockCs = note.onsetCs;
        writeBoundariesUpTo(note.onsetCs);
    }

    /** The shift the stream wants before sampling an event at or after lowerBoundCs. */
    [[nodiscard]] auto needsRebase(int32_t lowerBoundCs) const -> bool {
        return started && std::max(clockCs, lowerBoundCs) - baseCs >= kRebaseAtCs;
    }

    /** Move the origin forward: older items are dropped and the cache has to be rebuilt. */
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

    [[nodiscard]] auto tokens() const -> std::vector<int32_t> {
        std::vector<int32_t> out;
        out.reserve(items.size() * 4);
        for (const auto &item: items)
            appendTokens(out, item);
        return out;
    }

    /** Tokens before the first one that can still change (a held control's quad). */
    [[nodiscard]] auto committedLen() const -> size_t {
        size_t n = 0;
        for (const auto &item: items) {
            if (item.kind == Kind::Control && item.control.provisional)
                return n;
            n += item.kind == Kind::Td ? 1 : 4;
        }
        return n;
    }

    [[nodiscard]] auto pendingCount() const -> size_t { return pending.size(); }
    [[nodiscard]] auto itemCount() const -> size_t { return items.size(); }

    /** Token ids of a quad for these values (used while sampling an event field by field). */
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
    enum class Kind { Control, Event, Td };
    struct Control {
        Note note;
        bool provisional = false;
        int32_t instrument = kControlInstrument;
        int32_t deltaCs = 0;
        int32_t priority = 0;
        [[nodiscard]] auto actualCs() const -> int32_t { return note.onsetCs - deltaCs; }
    };
    struct Item {
        Kind kind = Kind::Event;
        Control control;
        int32_t tdBoundaryCs = 0;
        int32_t tdToken = 0;
    };

    [[nodiscard]] static auto itemTime(const Item &i) -> int32_t {
        return i.kind == Kind::Td ? i.tdBoundaryCs : i.control.note.onsetCs;
    }

    auto writeBoundariesUpTo(int32_t onsetCs) -> void {
        while (nextBoundaryCs <= onsetCs) {
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
        const int32_t instrument =
            item.kind == Kind::Control ? item.control.instrument : kEventInstrument;
        const auto quad = quadFor(item.control.note, instrument, baseCs);
        out.insert(out.end(), quad.begin(), quad.end());
    }

    int32_t baseCs = 0;
    int32_t nextBoundaryCs = 0;
    int32_t clockCs = 0;
    bool started = false;
    std::vector<Item> items;
    std::vector<Control> pending;
};
