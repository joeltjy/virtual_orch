#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

/**
 * Algorithm A: melody + bass of a polyphonic piano passage, a port of
 * ~/orchestration src/orchestration/preprocessing/melody_bass.py (version 1, rule-based,
 * skyline_and_bass) that produced the instrument-0 data of the mb model (T8).
 *
 * Times are centiseconds (ticks = cs, seconds = cs / 100). Whole notes are kept or dropped by
 * the cover test, the repeated-note chains and the troughs; only the last stage (cut at a later
 * onset, then the tick-wise top / bottom line) shortens notes.
 */
namespace MelodyBass {

// onset_div is centiseconds, so 10 cs is the 100 ms containment tolerance.
inline constexpr double kCoverFraction = 0.33;
inline constexpr int64_t kCoverCapMs = 100;
inline constexpr int64_t kOnsetWindowMs = 30;
inline constexpr int64_t kChainTolMs = 40;
inline constexpr int kChainMinNotes = 4;
inline constexpr double kTroughMaxSec = 1.0;
inline constexpr int kTroughOctave = 13;
inline constexpr double kTroughSingleMaxSec = 0.5;
inline constexpr int kTroughSingleSemitones = 10;

struct Note {
    int index = 0;
    int32_t start = 0; // cs
    int32_t end = 0;   // cs
    int pitch = 60;
    int velocity = 64;
};

struct Segment {
    int32_t start = 0;
    int32_t end = 0;
    int pitch = 60;
    int velocity = 64;
    int sourceIndex = 0;
};

struct Result {
    std::vector<Segment> melody;
    std::vector<Segment> bass;
};

namespace detail {

[[nodiscard]] inline auto ms(int32_t cs) -> int64_t { return static_cast<int64_t>(cs) * 10; }
[[nodiscard]] inline auto sec(int32_t cs) -> double { return static_cast<double>(cs) / 100.0; }

/** Python's round(): halves go to the even neighbour. */
[[nodiscard]] inline auto pyRound(double v) -> int64_t {
    return static_cast<int64_t>(std::nearbyint(v));
}

[[nodiscard]] inline auto tailMs(const Note &note) -> int64_t {
    const int64_t durationMs = ms(note.end) - ms(note.start);
    if (durationMs <= 0)
        return 0;
    return std::min<int64_t>(pyRound(kCoverFraction * static_cast<double>(durationMs)), kCoverCapMs);
}

/** Keep a note whole unless another pitch covers its middle or onset. */
[[nodiscard]] inline auto onsetKept(const std::vector<Note> &notes, bool lower)
    -> std::vector<Segment> {
    std::vector<Segment> chosen;
    for (const auto &note: notes) {
        const int64_t startMs = ms(note.start);
        const int64_t endMs = ms(note.end);
        if (endMs <= startMs)
            continue;
        const int64_t trim = tailMs(note);
        const int64_t windowStart = startMs + trim;
        const int64_t windowEnd = endMs - trim;
        bool blocked = false;
        for (const auto &other: notes) {
            if (other.index == note.index)
                continue;
            if (lower) {
                if (other.pitch >= note.pitch)
                    continue;
            } else if (other.pitch <= note.pitch) {
                continue;
            }
            const int64_t otherStart = ms(other.start);
            const int64_t otherEnd = ms(other.end);
            const bool onsetHit = std::llabs(otherStart - startMs) <= kOnsetWindowMs;
            const bool middleHit =
                windowStart <= windowEnd && otherStart <= windowEnd && otherEnd >= windowStart;
            if (onsetHit || middleHit) {
                blocked = true;
                break;
            }
        }
        if (blocked)
            continue;
        chosen.push_back({note.start, note.end, note.pitch, note.velocity, note.index});
    }
    return chosen;
}

/** Notes on a same-pitch chain of at least four whose gaps agree within 40 ms. */
[[nodiscard]] inline auto chainForbidden(const std::vector<Note> &notes) -> std::set<int> {
    std::set<int> forbidden;
    std::map<int, std::vector<Note>> byPitch;
    for (const auto &n: notes)
        byPitch[n.pitch].push_back(n);

    for (auto &[pitch, group]: byPitch) {
        std::sort(group.begin(), group.end(), [](const Note &a, const Note &b) {
            return a.start != b.start ? a.start < b.start : a.index < b.index;
        });
        struct Timed {
            const Note *note;
            int64_t ioiMs;
            std::vector<const Note *> landing;
        };
        std::vector<Timed> timed;
        for (size_t i = 0; i < group.size(); ++i) {
            const Note *later = nullptr;
            for (size_t j = i + 1; j < group.size(); ++j)
                if (group[j].start > group[i].start) {
                    later = &group[j];
                    break;
                }
            if (later == nullptr)
                continue;
            const int64_t ioi = pyRound((sec(later->start) - sec(group[i].start)) * 1000.0);
            if (ioi < 0)
                continue;
            Timed t{&group[i], ioi, {}};
            for (const auto &c: group)
                if (c.start == later->start)
                    t.landing.push_back(&c);
            timed.push_back(std::move(t));
        }
        size_t right = 0;
        for (size_t left = 0; left < timed.size(); ++left) {
            if (right < left)
                right = left;
            while (right + 1 < timed.size()) {
                int64_t lo = timed[left].ioiMs;
                int64_t hi = lo;
                for (size_t k = left; k <= right + 1; ++k) {
                    lo = std::min(lo, timed[k].ioiMs);
                    hi = std::max(hi, timed[k].ioiMs);
                }
                if (hi - lo > kChainTolMs)
                    break;
                ++right;
            }
            if (right - left + 1 >= static_cast<size_t>(kChainMinNotes)) {
                for (size_t k = left; k <= right; ++k)
                    forbidden.insert(timed[k].note->index);
                for (const auto *l: timed[right].landing)
                    forbidden.insert(l->index);
            }
        }
    }
    return forbidden;
}

/** Melody notes in a short run below their neighbours. */
[[nodiscard]] inline auto troughIndices(const std::vector<Note> &notes) -> std::set<int> {
    std::vector<Note> ordered = notes;
    std::sort(ordered.begin(), ordered.end(), [](const Note &a, const Note &b) {
        if (sec(a.start) != sec(b.start))
            return sec(a.start) < sec(b.start);
        if (a.pitch != b.pitch)
            return a.pitch < b.pitch;
        return a.index < b.index;
    });
    std::set<int> forbidden;
    const int count = static_cast<int>(ordered.size());
    for (int start = 1; start < count - 1; ++start) {
        for (int end = start; end < count - 1; ++end) {
            if (sec(ordered[static_cast<size_t>(end)].end) - sec(ordered[static_cast<size_t>(start)].start)
                >= kTroughMaxSec)
                continue;
            int highest = ordered[static_cast<size_t>(start)].pitch;
            for (int k = start; k <= end; ++k)
                highest = std::max(highest, ordered[static_cast<size_t>(k)].pitch);
            if (ordered[static_cast<size_t>(start - 1)].pitch - highest < kTroughOctave)
                continue;
            if (ordered[static_cast<size_t>(end + 1)].pitch - highest < kTroughOctave)
                continue;
            for (int k = start; k <= end; ++k)
                forbidden.insert(ordered[static_cast<size_t>(k)].index);
        }
    }
    for (int i = 1; i < count - 1; ++i) {
        const auto &note = ordered[static_cast<size_t>(i)];
        if (sec(note.end) - sec(note.start) >= kTroughSingleMaxSec)
            continue;
        if (ordered[static_cast<size_t>(i - 1)].pitch - note.pitch <= kTroughSingleSemitones)
            continue;
        if (ordered[static_cast<size_t>(i + 1)].pitch - note.pitch <= kTroughSingleSemitones)
            continue;
        forbidden.insert(note.index);
    }
    return forbidden;
}

/** End an earlier note at the next onset when the two would overlap. */
[[nodiscard]] inline auto truncateBeforeLaterOnsets(std::vector<Segment> segments)
    -> std::vector<Segment> {
    std::sort(segments.begin(), segments.end(), [](const Segment &a, const Segment &b) {
        if (a.start != b.start)
            return a.start < b.start;
        if (a.pitch != b.pitch)
            return a.pitch < b.pitch;
        return a.sourceIndex < b.sourceIndex;
    });
    std::vector<Segment> out;
    for (size_t i = 0; i < segments.size(); ++i) {
        int32_t end = segments[i].end;
        for (size_t j = i + 1; j < segments.size(); ++j) {
            if (segments[j].start <= segments[i].start)
                continue;
            if (segments[j].start < end)
                end = segments[j].start;
            break;
        }
        if (end - segments[i].start < 1)
            continue;
        Segment s = segments[i];
        s.end = end;
        out.push_back(s);
    }
    return out;
}

/** Keep the tick-wise top (melody) or bottom (bass) line; drop a note that loses half. */
[[nodiscard]] inline auto monophonicSlices(const std::vector<Segment> &segments, bool lower)
    -> std::vector<Segment> {
    if (segments.empty())
        return {};
    struct Event {
        int32_t tick;
        int kind; // 0 = end, 1 = start
        size_t index;
    };
    std::vector<Event> events;
    for (size_t i = 0; i < segments.size(); ++i) {
        events.push_back({segments[i].start, 1, i});
        events.push_back({segments[i].end, 0, i});
    }
    std::sort(events.begin(), events.end(), [](const Event &a, const Event &b) {
        if (a.tick != b.tick)
            return a.tick < b.tick;
        if (a.kind != b.kind)
            return a.kind < b.kind;
        return a.index < b.index;
    });

    std::set<size_t> active;
    std::vector<int64_t> owned(segments.size(), 0);
    std::vector<std::vector<std::pair<int32_t, int32_t>>> pieces(segments.size());
    std::optional<size_t> owner;
    int32_t ownerStart = 0;

    const auto winner = [&]() -> std::optional<size_t> {
        std::optional<size_t> best;
        for (const size_t i: active) {
            if (! best) {
                best = i;
                continue;
            }
            const auto &a = segments[i];
            const auto &b = segments[*best];
            const int pa = lower ? a.pitch : -a.pitch;
            const int pb = lower ? b.pitch : -b.pitch;
            const auto ka = std::make_tuple(pa, a.start, a.sourceIndex, i);
            const auto kb = std::make_tuple(pb, b.start, b.sourceIndex, *best);
            if (ka < kb)
                best = i;
        }
        return best;
    };

    for (size_t e = 0; e < events.size();) {
        const int32_t tick = events[e].tick;
        while (e < events.size() && events[e].tick == tick) {
            if (events[e].kind == 0)
                active.erase(events[e].index);
            else
                active.insert(events[e].index);
            ++e;
        }
        const auto next = winner();
        if (next == owner)
            continue;
        if (owner && tick > ownerStart) {
            owned[*owner] += tick - ownerStart;
            pieces[*owner].emplace_back(ownerStart, tick);
        }
        owner = next;
        ownerStart = tick;
    }

    std::vector<Segment> kept;
    for (size_t i = 0; i < segments.size(); ++i) {
        const int64_t original = segments[i].end - segments[i].start;
        if (original <= 0 || owned[i] * 2 < original)
            continue;
        for (const auto &[s, e]: pieces[i]) {
            if (e - s < 1)
                continue;
            Segment seg = segments[i];
            seg.start = s;
            seg.end = e;
            kept.push_back(seg);
        }
    }
    std::sort(kept.begin(), kept.end(), [](const Segment &a, const Segment &b) {
        if (a.start != b.start)
            return a.start < b.start;
        if (a.pitch != b.pitch)
            return a.pitch < b.pitch;
        return a.sourceIndex < b.sourceIndex;
    });
    return kept;
}

} // namespace detail

/** Melody (top line) and bass (bottom line) of `notes` (sorted by onset, pitch, offset). */
[[nodiscard]] inline auto skylineAndBass(const std::vector<Note> &notes) -> Result {
    using namespace detail;
    const auto coverMelody = onsetKept(notes, false);
    const auto chain = chainForbidden(notes);
    std::vector<Segment> melody;
    for (const auto &s: coverMelody)
        if (! chain.count(s.sourceIndex))
            melody.push_back(s);

    std::map<int, Note> byIndex;
    for (const auto &n: notes)
        byIndex[n.index] = n;
    std::vector<Note> melodyNotes;
    for (const auto &s: melody)
        melodyNotes.push_back(byIndex[s.sourceIndex]);
    const auto trough = troughIndices(melodyNotes);

    std::vector<Segment> melodyKept;
    for (const auto &s: melody)
        if (! trough.count(s.sourceIndex))
            melodyKept.push_back(s);

    std::set<int> blocked;
    for (const auto &s: coverMelody)
        if (! trough.count(s.sourceIndex))
            blocked.insert(s.sourceIndex);
    std::vector<Segment> bass;
    for (const auto &s: onsetKept(notes, true))
        if (! blocked.count(s.sourceIndex))
            bass.push_back(s);

    Result result;
    result.melody = monophonicSlices(truncateBeforeLaterOnsets(melodyKept), false);
    result.bass = monophonicSlices(truncateBeforeLaterOnsets(bass), true);
    return result;
}

} // namespace MelodyBass
