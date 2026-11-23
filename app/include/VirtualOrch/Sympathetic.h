#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string_view>
#include <vector>

#include "VirtualOrch/InstrumentConstants.h"
#include "VirtualOrch/MusicToken.h"

/**
 * Sympathetic Orchestra gesture layer (see README.md).
 *
 * Gesture events arrive as OSC (/sym/...) from the webcam tracker in sympathetic/. This header
 * holds the shared state they set and the pure functions the engine applies:
 *  - warp factor (Tempo): stretches each predicted reduction note before the models see it;
 *  - cued parts (Cueing): only cued parts may be given notes by the orchestration model;
 *  - selected part (Select) and expression (palm spread): velocity at playback.
 *
 * Off (enabled == false) until the first /sym/ message, so the app behaves as before without the
 * tracker.
 */
namespace Sympathetic {

// ---------------------------------------------------------------------------------------------
// Parts of the Sympathetic Orchestra grid -> virtual_orch local instrument ids

[[nodiscard]] constexpr auto bit(int32_t localId) -> uint32_t {
    return uint32_t{1} << static_cast<uint32_t>(localId);
}

struct Part {
    std::string_view name;
    uint32_t localIds; // bit mask of local instrument ids
};

/** Harp and Conductor have no counterpart and are not listed. */
inline constexpr std::array<Part, 16> kParts{{
    {"Flute", bit(7)},
    {"Oboe", bit(8)},
    {"Clarinet", bit(9)},
    {"Bassoon", bit(10)},
    {"French Horns", bit(12)},
    {"Trumpets", bit(11)},
    {"Trombones", bit(13)},
    {"Tuba", bit(14)},
    {"Timpani", bit(16)},
    {"Percussion", bit(17)},
    {"Piano", bit(15)},
    {"Violin 1", bit(0)},
    {"Violin 2", bit(0)},
    {"Viola", bit(1)},
    {"Cello", bit(2)},
    {"Bass", bit(3)},
}};

/** Solo strings (violin, viola, cello, contrabass). */
inline constexpr uint32_t kSoloStringsMask = bit(0) | bit(1) | bit(2) | bit(3);
/** Tremolo strings, pizzicato strings, string ensemble: articulations, not parts. */
inline constexpr uint32_t kStringArticulationsMask = bit(4) | bit(5) | bit(6);

[[nodiscard]] inline auto partIndex(std::string_view name) -> std::optional<int32_t> {
    for (size_t i = 0; i < kParts.size(); ++i)
        if (kParts[i].name == name)
            return static_cast<int32_t>(i);
    return std::nullopt;
}

[[nodiscard]] constexpr auto inMask(uint32_t mask, int32_t localId) -> bool {
    return localId >= 0 && localId < 32 && (mask & bit(localId)) != 0;
}

/** May `localId` be given notes, given the cued mask? Articulations follow any cued solo string. */
[[nodiscard]] constexpr auto isAllowed(uint32_t cuedMask, int32_t localId) -> bool {
    if (inMask(cuedMask, localId))
        return true;
    return inMask(kStringArticulationsMask, localId) && (cuedMask & kSoloStringsMask) != 0;
}

/**
 * The instrument a note actually sounds on: String Ensemble 1 is routed by pitch to a solo
 * string (same rule as InstrumentConstants::outputChannelForLocalInstrumentId).
 */
[[nodiscard]] inline auto soundingLocalId(int32_t localId, int32_t pitch) -> int32_t {
    if (localId != InstrumentConstants::kStringEnsembleLocalId)
        return localId;
    for (const auto solo: InstrumentConstants::kSoloStringLocalIds)
        if (InstrumentConstants::localInstrumentAllowsPitch(solo, pitch))
            return solo;
    return localId;
}

// ---------------------------------------------------------------------------------------------
// Warp (Tempo)

inline constexpr float kMinWarp = 0.5f;
inline constexpr float kMaxWarp = 2.0f;

/**
 * Stretch a predicted note about the current time `nowCs`:
 *   onset' = now + warp * (onset - now),  duration' = warp * duration.
 * An onset already at or before now keeps its time (warping it would move it further into the
 * past). The warped onset never goes before `floorCs` (the previous generated onset) unless the
 * model's own onset already did, so a falling warp cannot reorder notes. Duration stays in
 * [1, maxDurCs]. Times and durations are in centiseconds (onsets absolute).
 */
[[nodiscard]] inline auto warpNote(int32_t onsetCs, int32_t durationCs, int32_t nowCs, float warp,
                                   int32_t floorCs, int32_t maxDurCs)
    -> std::pair<int32_t, int32_t> {
    int32_t onset = onsetCs;
    if (onsetCs > nowCs) {
        onset = nowCs
                + static_cast<int32_t>(std::lround(warp * static_cast<float>(onsetCs - nowCs)));
        onset = std::max(onset, std::min(onsetCs, floorCs));
    }
    const auto duration = std::clamp(
        static_cast<int32_t>(std::lround(warp * static_cast<float>(durationCs))), 1, maxDurCs);
    return {onset, duration};
}

/**
 * Beat tracker: the beat period of the first kReferenceBeats beats sets warp = 1; afterwards
 * warp = (smoothed beat period) / reference, clamped to [kMinWarp, kMaxWarp]. Faster conducting
 * gives a shorter period, so warp < 1 and the music is compressed.
 */
class BeatTracker {
public:
    static constexpr int kReferenceBeats = 4;     // 3 intervals
    static constexpr double kMinIntervalMs = 200;  // 300 BPM
    static constexpr double kMaxIntervalMs = 2000; // 30 BPM: a longer gap restarts the count
    static constexpr double kSmoothing = 0.4;      // weight of the newest interval

    /** Returns the new warp, or nothing while the reference is still being set. */
    auto onBeat(double timeMs) -> std::optional<float> {
        if (lastBeatMs.has_value()) {
            const double interval = timeMs - *lastBeatMs;
            if (interval < kMinIntervalMs)
                return std::nullopt; // double detection: keep the first beat
            if (interval > kMaxIntervalMs) {
                // The conductor stopped: a new phrase starts from this beat, same reference.
                lastBeatMs = timeMs;
                return std::nullopt;
            }
            lastBeatMs = timeMs;
            if (referenceMs <= 0.0) {
                referenceIntervals.push_back(interval);
                if (static_cast<int>(referenceIntervals.size()) < kReferenceBeats - 1)
                    return std::nullopt;
                auto sorted = referenceIntervals;
                std::sort(sorted.begin(), sorted.end());
                referenceMs = sorted[sorted.size() / 2];
                smoothedMs = referenceMs;
                return 1.0f;
            }
            smoothedMs = kSmoothing * interval + (1.0 - kSmoothing) * smoothedMs;
            return std::clamp(static_cast<float>(smoothedMs / referenceMs), kMinWarp, kMaxWarp);
        }
        lastBeatMs = timeMs;
        return std::nullopt;
    }

    auto reset() -> void {
        lastBeatMs.reset();
        referenceIntervals.clear();
        referenceMs = 0.0;
        smoothedMs = 0.0;
    }

    [[nodiscard]] auto referenceIntervalMs() const -> double { return referenceMs; }

private:
    std::optional<double> lastBeatMs;
    std::vector<double> referenceIntervals;
    double referenceMs = 0.0;
    double smoothedMs = 0.0;
};

// ---------------------------------------------------------------------------------------------
// Velocity (Select, Expression)

inline constexpr int32_t kDefaultSelectBoost = 20;

/** Palm spread 0..1 -> velocity scale 0.5..1.5 (0.5 spread = unchanged). */
[[nodiscard]] constexpr auto expressionScaleForSpread(float spread) -> float {
    return 0.5f + std::clamp(spread, 0.0f, 1.0f);
}

/**
 * Velocity of a note on `soundingId`: scaled by expression, then +boost if it belongs to the
 * selected part and -boost otherwise (no change while nothing is selected). String articulations
 * count as part of a selected string part. Clamped to MIDI 1..127.
 */
[[nodiscard]] inline auto velocityFor(int32_t velocity, int32_t soundingId, int32_t selectedPart,
                                      float expressionScale, int32_t boost) -> int32_t {
    auto v = static_cast<int32_t>(std::lround(static_cast<float>(velocity) * expressionScale));
    if (selectedPart >= 0 && selectedPart < static_cast<int32_t>(kParts.size())) {
        uint32_t mask = kParts[static_cast<size_t>(selectedPart)].localIds;
        if ((mask & kSoloStringsMask) != 0)
            mask |= kStringArticulationsMask;
        v += inMask(mask, soundingId) ? boost : -boost;
    }
    return std::clamp(v, 1, 127);
}

// ---------------------------------------------------------------------------------------------
// Shared state (written by the OSC thread, read by the model and playback threads)

struct State {
    std::atomic<bool> enabled{false};
    std::atomic<float> warp{1.0f};
    std::atomic<uint32_t> cuedMask{0};
    std::atomic<int32_t> selectedPart{-1};
    std::atomic<float> expressionScale{1.0f};
    std::atomic<int32_t> selectBoost{kDefaultSelectBoost};

    std::mutex beatLock;
    BeatTracker beats;

    /** Everything back to the start: no part cued, nothing selected, warp 1, new reference. */
    auto reset() -> void {
        warp.store(1.0f);
        cuedMask.store(0);
        selectedPart.store(-1);
        expressionScale.store(1.0f);
        const std::lock_guard lock(beatLock);
        beats.reset();
    }

    /**
     * Point at a part: cue it if none of its instruments is cued yet, else select it.
     * Returns true if it was a cue.
     */
    auto point(int32_t partIndex) -> bool {
        if (partIndex < 0 || partIndex >= static_cast<int32_t>(kParts.size()))
            return false;
        const auto mask = kParts[static_cast<size_t>(partIndex)].localIds;
        if ((cuedMask.load() & mask) != mask) {
            cuedMask.fetch_or(mask);
            return true;
        }
        selectedPart.store(partIndex);
        return false;
    }

    auto beat(double timeMs) -> void {
        const std::lock_guard lock(beatLock);
        if (const auto w = beats.onBeat(timeMs))
            warp.store(*w);
    }

    /** Filter an instrument list down to what may play (all of it while disabled). */
    [[nodiscard]] auto filterAllowed(const std::vector<int32_t> &instruments) const
        -> std::vector<int32_t> {
        if (! enabled.load(std::memory_order_relaxed))
            return instruments;
        const auto mask = cuedMask.load(std::memory_order_relaxed);
        std::vector<int32_t> out;
        out.reserve(instruments.size());
        for (const auto id: instruments)
            if (isAllowed(mask, id))
                out.push_back(id);
        return out;
    }

    /** Playback velocity for a note (unchanged while disabled). */
    [[nodiscard]] auto playbackVelocity(int32_t velocity, int32_t localId, int32_t pitch) const
        -> int32_t {
        if (! enabled.load(std::memory_order_relaxed))
            return velocity;
        return velocityFor(velocity, soundingLocalId(localId, pitch),
                           selectedPart.load(std::memory_order_relaxed),
                           expressionScale.load(std::memory_order_relaxed),
                           selectBoost.load(std::memory_order_relaxed));
    }

    /** May a note on this instrument sound now (true while disabled)? */
    [[nodiscard]] auto maySound(int32_t localId, int32_t pitch) const -> bool {
        if (! enabled.load(std::memory_order_relaxed))
            return true;
        return isAllowed(cuedMask.load(std::memory_order_relaxed), soundingLocalId(localId, pitch));
    }

    [[nodiscard]] auto currentWarp() const -> float {
        return enabled.load(std::memory_order_relaxed) ? warp.load(std::memory_order_relaxed)
                                                       : 1.0f;
    }
};

[[nodiscard]] inline auto state() -> State & {
    static State instance;
    return instance;
}

} // namespace Sympathetic
