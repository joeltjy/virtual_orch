#pragma once

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <set>
#include <tuple>
#include <vector>

#include "VirtualOrch/reduction/DenseTypes.h"
#include "VirtualOrch/reduction/MelodyBassState.h"

/**
 * Token stream of the mb model (T8, mb_phase2, vocab 55285, learned positions): the user's
 * melody + bass as AMT control quads on instrument 0, the model's own melody + bass as event
 * quads on instrument 1, and VISIBLE 1 s rest quads. It starts with ANTICIPATE.
 *
 *   control quad: ATIME (27513 + t), ADUR (37513 + d), ANOTE (38513 + 128*instr + pitch),
 *                 AVEL (55156 + v)
 *   event quad:   t, 10000 + d, 11000 + 128*instr + pitch, 55028 + v
 *   rest quad:    t, 10000, REST (27512), 55028
 *
 * Controls come from R (MelodyBassState). They are inserted once the clock has passed their
 * onset (a late one simply lands out of onset order), and a final per-line monophony check
 * (the mb model must see strictly monophonic melody and bass) shortens a not-yet-written
 * predecessor or skips the newcomer. A control whose source key is still held is PROVISIONAL:
 * it keeps the duration it had when it was written until the note-off, and committedLen()
 * stops at it.
 *
 * The stream is trimmed like the cache policy of T8: when it passes 256 tokens the oldest items
 * are dropped down to 128 tokens (the cache is rebuilt at positions 0..127).
 */
class MelodyStream {
public:
    static constexpr int32_t kTimeCtl = 27513;
    static constexpr int32_t kDurCtl = 37513;
    static constexpr int32_t kNoteCtl = 38513;
    static constexpr int32_t kVelCtl = 55156;
    static constexpr int32_t kRest = 27512;
    static constexpr int32_t kAnticipate = 55027;
    static constexpr int32_t kVelEvent = 55028;
    static constexpr int32_t kMaxModelTimeCs = 9999;
    static constexpr int32_t kRestStepCs = 100;
    static constexpr int32_t kMaxTokens = 256;
    static constexpr int32_t kKeepTokens = 128;
    static constexpr int32_t kHoldCommitCs = 1000;
    /** Same rule as AnticipatedStream::kEvictRebaseCs, applied when the stream is cut. */
    static constexpr int32_t kEvictRebaseCs = 3000;
    static constexpr int32_t kRebaseAtCs = 9000;
    static constexpr int32_t kRebaseShiftCs = 6000;

    struct Note {
        int32_t onsetCs = 0;
        int32_t durationCs = 0;
        int pitch = 60;
        int velocity = 64;
    };

    auto reset(int32_t nowCs) -> void {
        items.clear();
        pending.clear();
        inserted.clear();
        lastEnd = {INT32_MIN, INT32_MIN};
        baseCs = (std::max(nowCs, 0) / 200) * 200;
        clockCs = baseCs;
        started = true;
        forcedCommits = 0;
    }

    [[nodiscard]] auto isStarted() const -> bool { return started; }
    [[nodiscard]] auto base() const -> int32_t { return baseCs; }
    [[nodiscard]] auto clock() const -> int32_t { return clockCs; }
    [[nodiscard]] auto forcedCommitCount() const -> int { return forcedCommits; }
    /** The origin shift of the last cut (0 when the cut did not rebase). */
    [[nodiscard]] auto lastCutShift() const -> int32_t { return lastShift; }
    [[nodiscard]] auto pendingCount() const -> size_t { return pending.size(); }

    /** Bring the controls in line with R: new pieces become pending, note-offs fix durations. */
    auto syncControls(const std::vector<MelodyBassState::Piece> &pieces) -> void {
        std::vector<Control> next;
        for (const auto &p: pieces) {
            const Id id{p.key, p.onset, p.pitch, p.line};
            if (inserted.count(id)) {
                // Written already: only a confirmed duration (note-off) may still change it.
                if (! p.provisional)
                    for (auto &item: items)
                        if (item.kind == Kind::Control && item.control.provisional
                            && idOf(item.control) == id) {
                            item.control.note.durationCs = p.durationCs;
                            item.control.provisional = false;
                        }
                continue;
            }
            if (p.onset < baseCs)
                continue; // from before this stream began
            Control c;
            c.key = p.key;
            c.note = {p.onset, p.durationCs, p.pitch, p.velocity};
            c.line = p.line;
            c.provisional = p.provisional;
            next.push_back(c);
        }
        std::stable_sort(next.begin(), next.end(), [](const Control &a, const Control &b) {
            if (a.note.onsetCs != b.note.onsetCs)
                return a.note.onsetCs < b.note.onsetCs;
            if (a.note.durationCs != b.note.durationCs)
                return a.note.durationCs < b.note.durationCs;
            return a.note.pitch < b.note.pitch;
        });
        pending = std::move(next);
    }

    /** Held controls older than the hold limit are committed with the duration they have. */
    auto commitLongHolds(int32_t nowCs) -> void {
        for (auto &item: items)
            if (item.kind == Kind::Control && item.control.provisional
                && nowCs - item.control.note.onsetCs > kHoldCommitCs)
                item.control.provisional = false;
    }

    /** Write every pending control with onset <= clockNow. Returns how many were written. */
    auto insertDue(int32_t clockNow) -> int {
        std::vector<Control> batch;
        while (! pending.empty() && pending.front().note.onsetCs <= clockNow) {
            batch.push_back(pending.front());
            pending.erase(pending.begin());
        }
        int written = 0;
        std::vector<size_t> batchItems;
        for (auto &c: batch) {
            inserted.insert(idOf(c)); // never offered again, whether written or skipped
            const int line = c.line;
            const int32_t end = c.note.onsetCs + c.note.durationCs;
            // Monophony per line: the newcomer must start at or after the previous end.
            if (lastEnd[static_cast<size_t>(line)] > c.note.onsetCs) {
                bool shortened = false;
                for (auto it = batchItems.rbegin(); it != batchItems.rend(); ++it) {
                    auto &prev = items[*it].control;
                    if (prev.line != line)
                        continue;
                    prev.note.durationCs = std::max(0, c.note.onsetCs - prev.note.onsetCs);
                    prev.trimmed = true;
                    shortened = true;
                    break;
                }
                if (! shortened)
                    continue; // the previous control is already in the stream: skip the newcomer
            }
            lastEnd[static_cast<size_t>(line)] = end;
            Item item;
            item.kind = Kind::Control;
            item.control = c;
            items.push_back(item);
            batchItems.push_back(items.size() - 1);
            ++written;
        }
        // drop batch controls that were shortened to nothing
        items.erase(std::remove_if(items.begin(), items.end(),
                                   [](const Item &i) {
                                       return i.kind == Kind::Control && i.control.trimmed
                                              && i.control.note.durationCs < DenseQuantize::DurationMinCs;
                                   }),
                    items.end());
        return written;
    }

    auto appendEvent(const Note &note) -> void {
        Item item;
        item.kind = Kind::Event;
        item.control.note = note;
        items.push_back(item);
        clockCs = std::max(clockCs, note.onsetCs);
    }

    /** A visible rest at r (the clock moves to it). */
    auto appendRest(int32_t restCs) -> void {
        Item item;
        item.kind = Kind::Rest;
        item.control.note.onsetCs = restCs;
        items.push_back(item);
        clockCs = std::max(clockCs, restCs);
    }

    [[nodiscard]] auto needsRebase(int32_t lowerBoundCs) const -> bool {
        return started && std::max(clockCs, lowerBoundCs) - baseCs >= kRebaseAtCs;
    }

    auto rebase() -> void {
        baseCs += kRebaseShiftCs;
        items.erase(std::remove_if(items.begin(), items.end(),
                                   [this](const Item &i) { return i.control.note.onsetCs < baseCs; }),
                    items.end());
        pending.erase(std::remove_if(pending.begin(), pending.end(),
                                     [this](const Control &c) { return c.note.onsetCs < baseCs; }),
                      pending.end());
        for (auto it = inserted.begin(); it != inserted.end();)
            it = std::get<1>(*it) < baseCs ? inserted.erase(it) : std::next(it);
        clockCs = std::max(clockCs, baseCs);
    }

    /** Drop the oldest items once the stream passes kMaxTokens (cache rebuild at 0..127). */
    auto trimIfNeeded() -> bool {
        lastShift = 0;
        if (tokenCount() <= kMaxTokens)
            return false;
        size_t drop = 0;
        while (tokenCount(drop) > kKeepTokens && drop < items.size())
            ++drop;
        int32_t droppedOnset = INT32_MIN;
        for (size_t i = 0; i < drop; ++i) {
            if (items[i].kind == Kind::Control && items[i].control.provisional)
                ++forcedCommits; // safety rule: the dropped held control is final now
            droppedOnset = std::max(droppedOnset, items[i].control.note.onsetCs);
        }
        items.erase(items.begin(), items.begin() + static_cast<std::ptrdiff_t>(drop));

        // A note older than 30 s was cut away: move the origin forward by the largest multiple of
        // 30 s that keeps every retained note at onset >= 0 (the cache is rebuilt right now
        // anyway, so the shifted times cost nothing extra). ANTICIPATE stays token 0.
        if (droppedOnset != INT32_MIN && droppedOnset - baseCs > kEvictRebaseCs && ! items.empty()) {
            int32_t minOnset = INT32_MAX;
            for (const auto &item: items)
                minOnset = std::min(minOnset, item.control.note.onsetCs);
            const int32_t k = ((minOnset - baseCs) / kEvictRebaseCs) * kEvictRebaseCs;
            if (k > 0) {
                baseCs += k;
                pending.erase(std::remove_if(pending.begin(), pending.end(),
                                             [this](const Control &c) { return c.note.onsetCs < baseCs; }),
                              pending.end());
                for (auto it = inserted.begin(); it != inserted.end();)
                    it = std::get<1>(*it) < baseCs ? inserted.erase(it) : std::next(it);
                clockCs = std::max(clockCs, baseCs);
                lastShift = k;
            }
        }
        return true;
    }

    [[nodiscard]] auto tokens() const -> std::vector<int32_t> {
        std::vector<int32_t> out;
        out.reserve(1 + items.size() * 4);
        out.push_back(kAnticipate);
        for (const auto &item: items)
            appendTokens(out, item);
        return out;
    }

    /** Tokens before the first held control (the ANTICIPATE token is always committed). */
    [[nodiscard]] auto committedLen() const -> size_t {
        size_t n = 1;
        for (const auto &item: items) {
            if (item.kind == Kind::Control && item.control.provisional)
                return n;
            n += 4;
        }
        return n;
    }

    [[nodiscard]] auto itemCount() const -> size_t { return items.size(); }

private:
    enum class Kind { Control, Event, Rest };
    using Id = std::tuple<int64_t, int32_t, int, int>; // key, onset, pitch, line

    struct Control {
        int64_t key = 0;
        Note note;
        int line = 0;
        bool provisional = false;
        bool trimmed = false; // shortened by the monophony check
    };
    struct Item {
        Kind kind = Kind::Event;
        Control control;
    };

    [[nodiscard]] static auto idOf(const Control &c) -> Id {
        return {c.key, c.note.onsetCs, c.note.pitch, c.line};
    }

    [[nodiscard]] auto tokenCount(size_t skipItems = 0) const -> int32_t {
        return 1 + 4 * static_cast<int32_t>(items.size() - std::min(skipItems, items.size()));
    }

    auto appendTokens(std::vector<int32_t> &out, const Item &item) const -> void {
        const auto &n = item.control.note;
        const int32_t time = std::clamp(n.onsetCs - baseCs, 0, kMaxModelTimeCs);
        switch (item.kind) {
            case Kind::Control:
                out.push_back(kTimeCtl + time);
                out.push_back(kDurCtl + durationOf(n));
                out.push_back(kNoteCtl + 0 * 128 + n.pitch);
                out.push_back(kVelCtl + DenseQuantize::snapVelocity(n.velocity));
                break;
            case Kind::Event:
                out.push_back(time);
                out.push_back(10000 + durationOf(n));
                out.push_back(11000 + 128 * 1 + n.pitch);
                out.push_back(kVelEvent + DenseQuantize::snapVelocity(n.velocity));
                break;
            case Kind::Rest:
                out.push_back(time);
                out.push_back(10000);
                out.push_back(kRest);
                out.push_back(kVelEvent);
                break;
        }
    }

    [[nodiscard]] static auto durationOf(const Note &n) -> int32_t {
        return DenseQuantize::snapDurationCs(n.durationCs);
    }

    int32_t baseCs = 0;
    int32_t clockCs = 0;
    bool started = false;
    int forcedCommits = 0;
    int32_t lastShift = 0;
    std::array<int32_t, 2> lastEnd{INT32_MIN, INT32_MIN};
    std::vector<Item> items;
    std::vector<Control> pending;
    std::set<Id> inserted;
};
