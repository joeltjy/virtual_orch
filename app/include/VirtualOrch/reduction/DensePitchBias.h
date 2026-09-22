#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "VirtualOrch/reduction/DenseTypes.h"

/**
 * Online pitch-bias statistics for ReductionTransformerV2.
 *
 * Idea: look at pitches sounded in the last 5 s of clock time, estimate how
 * often each pitch (π) and each pitch leap (s_y) appears, then subtract
 * τ·log(π) and τ′·log(s_y) from note logits so overused pitches/intervals become
 * less likely. τ holds/ramps after S gains a pitch; τ′ after the window gains a
 * new pitch interval (Δ). Aging out alone does not reset either.
 *
 * Full walkthrough: PITCH_BIAS.md in this folder.
 */
namespace DensePitchBias {

inline constexpr int32_t WindowCs = 500; // 5 s
inline constexpr float UnigramEpsilon = 1.0f / 128.0f;
inline constexpr float DeltaEpsilon = 0.01f;
inline constexpr float TauBase = 0.5f;
inline constexpr float TauHoldSeconds = 2.0f;

/** Delta index: MIDI pitch delta in [-127, 127] → [0, 254]. */
inline constexpr int DeltaBins = 255;
inline constexpr int DeltaOffset = 127;

[[nodiscard]] inline auto deltaIndex(int delta) -> int {
    const int idx = delta + DeltaOffset;
    return std::clamp(idx, 0, DeltaBins - 1);
}

[[nodiscard]] inline auto pitchFromDenseNoteToken(int32_t noteToken) -> int32_t {
    const auto rel = noteToken - static_cast<int32_t>(DenseVocab::NoteOffset);
    if (rel < 0)
        return -1;
    return rel % DenseConfig::MaxPitch;
}

struct PitchWindowStats {
    /** Smoothed pitch frequencies π[p] over the window (sums ≈ 1). */
    std::array<float, DenseConfig::MaxPitch> pi{};
    /** Smoothed consecutive pitch-delta frequencies; index = delta + 127. */
    std::array<float, DeltaBins> sDelta{};
    /** Sorted unique pitches in the window (S); drives the τ schedule. */
    std::vector<int32_t> uniquePitches;
    /** Sorted unique consecutive pitch deltas (Δ) in the window; drives τ′. */
    std::vector<int32_t> uniqueDeltas;
    /** Last pitch in stream order within the window; -1 if none. */
    int32_t lastPitch = -1;
    int noteCount = 0;
};

/**
 * Collect onset pitches from dense inputData with onset in (now - WindowCs, now],
 * where now = max(nowCs, last onset) when nowCs >= 0 (live clock / UI), or last
 * onset alone when nowCs < 0. Matches amt_causal's prefix-relative 5 s window when
 * generation is ahead of the wall clock.
 */
[[nodiscard]] auto collectPitchWindow(const std::vector<int32_t> &denseInputData, int32_t nowCs)
    -> PitchWindowStats;

/**
 * Same window using an exact prefix "now" (the just-sampled onset). Offline
 * amt_causal applies τ / π after onset+duration are in the prefix — call this
 * before the note-field logit adjustment, not only at the start of the note.
 */
[[nodiscard]] auto collectPitchWindowAt(const std::vector<int32_t> &denseInputData,
                                        int32_t prefixNowCs) -> PitchWindowStats;

/**
 * Subtract τ·log(π[p]) and τ′·log(s_y[p−lastPitch]) from every instrument band's
 * pitch logits. No-op when the window is empty. Leaves non-finite entries alone.
 */
auto applyNoteLogitsBias(std::vector<float> &logits,
                         const PitchWindowStats &stats,
                         float tau,
                         float tauPrime) -> void;

/**
 * Strength of the pitch / interval bias over time (independent clocks):
 * - τ: after S *gains* a pitch → hold TauBase for TauHoldSeconds, then ramp.
 * - τ′: after the window *gains* a pitch interval Δ → same hold/ramp.
 * Shrinkage alone (pitches / intervals leaving) does not reset.
 */
class PitchTauScheduler {
public:
    struct Result {
        float tau = TauBase;
        float tauPrime = TauBase;
    };

    [[nodiscard]] auto evaluate(int32_t nowCs,
                                const std::vector<int32_t> &uniquePitches,
                                const std::vector<int32_t> &uniqueDeltas) -> Result;

    [[nodiscard]] auto lastUniquePitches() const -> const std::vector<int32_t> & { return lastS; }

    [[nodiscard]] auto lastUniqueDeltas() const -> const std::vector<int32_t> & { return lastDeltas; }

    auto reset() -> void;

private:
    std::vector<int32_t> lastS;
    std::vector<int32_t> lastDeltas;
    int32_t sStableSinceCs = 0;
    int32_t deltaStableSinceCs = 0;
    bool hasS = false;
    bool hasDeltas = false;
};

} // namespace DensePitchBias
