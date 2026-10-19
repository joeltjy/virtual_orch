#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/orchestration-models/OrchestrationModel.h"

/**
 * Sustain-pedal clear of the playback queue.
 *
 * Rule: a note that has not started sounding when the pedal goes down never sounds; a note
 * that is sounding ends by pedalTime + kGraceCs. The pedal clear is also sent upstream as a
 * ClearQueue token whose velocity is kMarkerVelocity, so the orchestration transformer can
 * keep it behind the notes of its batch and OutputPlayback can tell it from the other clears.
 */
namespace PedalClear {

/** Latest end of a note that was already sounding when the pedal went down (0.3 s). */
inline constexpr int32_t kGraceCs = 30;

/** How long OutputPlayback keeps dropping notes that start after the pedal press while it
 *  waits for the upstream ClearQueue token (safety net if that token never arrives). */
inline constexpr int32_t kDropWindowCs = 100;

/** Velocity carried by the pedal's ClearQueue token (the other ClearQueue tokens use 100). */
inline constexpr int32_t kMarkerVelocity = 127;

[[nodiscard]] inline auto isClearQueue(const Token &token) -> bool {
    return token.note == static_cast<int32_t>(Vocab::ClearQueue);
}

[[nodiscard]] inline auto isPedalClear(const Token &token) -> bool {
    return isClearQueue(token) && token.velocity == kMarkerVelocity;
}

[[nodiscard]] inline auto makeClearToken(int32_t pedalTimeCs) -> Token {
    Token token{pedalTimeCs, static_cast<int32_t>(Vocab::DurOffset),
                static_cast<int32_t>(Vocab::ClearQueue)};
    token.velocity = kMarkerVelocity;
    return token;
}

/**
 * Apply a pedal press at `pedalTimeCs` to the playback containers.
 *  - `queued`: notes whose note-on has not been sent. All are dropped.
 *  - `playing`: notes whose note-on was sent. Any note ending after pedalTimeCs + kGraceCs is
 *    shortened to end there; if that time has already passed (`nowCs`), `noteOff(note)` is
 *    called at once and the note leaves `playing`.
 * Returns true if anything was removed or shortened.
 */
template <typename QueuedSet, typename PlayingSet, typename NoteOffFn>
auto apply(QueuedSet &queued, PlayingSet &playing, int32_t pedalTimeCs, int32_t nowCs,
           NoteOffFn &&noteOff) -> bool {
    bool changed = ! queued.empty();
    queued.clear();

    const int32_t cutEnd = pedalTimeCs + kGraceCs;
    std::vector<typename PlayingSet::value_type> revised;
    for (auto it = playing.begin(); it != playing.end();) {
        const int32_t end = it->token.time + it->token.getRealDuration();
        if (end <= cutEnd) {
            ++it;
            continue;
        }
        auto note = *it;
        it = playing.erase(it);
        changed = true;
        if (cutEnd <= nowCs) {
            noteOff(note);
            continue;
        }
        note.token.duration = static_cast<int32_t>(Vocab::DurOffset) + (cutEnd - note.token.time);
        revised.push_back(note);
    }
    for (const auto &note: revised)
        playing.insert(note);
    return changed;
}

/**
 * After a pedal press, notes that start after the press are dropped as they arrive until the
 * upstream ClearQueue token has passed (or the window expires).
 */
struct DropWindow {
    bool active = false;
    int32_t pedalTimeCs = 0;
    int32_t expiresCs = 0;

    auto start(int32_t pedalTime, int32_t nowCs) -> void {
        active = true;
        pedalTimeCs = pedalTime;
        expiresCs = std::max(pedalTime, nowCs) + kDropWindowCs;
    }

    auto finish() -> void { active = false; }

    /** True when a note with this onset must not be scheduled. */
    [[nodiscard]] auto drops(int32_t onsetCs, int32_t nowCs) -> bool {
        if (! active)
            return false;
        if (nowCs >= expiresCs) {
            active = false;
            return false;
        }
        return onsetCs > pedalTimeCs;
    }
};

/**
 * Order for the orchestration transformer's output: notes by onset (then duration), but
 * pedal-clear tokens last, so they erase the notes of the same batch that start after the
 * press instead of being sorted in front of them.
 */
inline auto sortForOutput(std::vector<OrchestrationNote> &notes) -> void {
    std::stable_sort(notes.begin(), notes.end(),
                     [](const OrchestrationNote &a, const OrchestrationNote &b) {
                         const bool aClear = isPedalClear(a.token);
                         const bool bClear = isPedalClear(b.token);
                         if (aClear != bClear)
                             return bClear;
                         if (a.token.time != b.token.time)
                             return a.token.time < b.token.time;
                         return a.token.duration < b.token.duration;
                     });
}

} // namespace PedalClear
