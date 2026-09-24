#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "VirtualOrch/reduction/DenseTypes.h"

/**
 * Online onset-gap bias for ReductionTransformerV2 / PianoReduction.
 *
 * Mirrors the pitch-interval (s_y) half of DensePitchBias: look at consecutive
 * onset differences in the last 5 s, estimate frequencies over Δ ∈ [10, 100],
 * then subtract τ·log(s_Δ) from onset logits for candidate gaps in that range.
 * Same-onset / sub-10 cs gaps are out of range (no s_y term). Every bin gets
 * GapEpsilon Laplace smoothing so log(s) is always defined.
 *
 * τ stays fixed at TauBase (no hold/ramp), matching duration bias.
 */
namespace DenseOnsetGapBias {

inline constexpr int32_t WindowCs = 500; // 5 s
inline constexpr float TauBase = 0.5f;
inline constexpr float TauHoldSeconds = 2.0f; // unused; API parity

/** Consecutive onset-gap range for s_y (centiseconds). */
inline constexpr int32_t GapLoCs = 10;
inline constexpr int32_t GapHiCs = 100;
inline constexpr int GapBins = GapHiCs - GapLoCs + 1; // 91
inline constexpr int GapIndexOffset = -GapLoCs;       // index = gap - 10

/** Additive smoothing on every gap bin (zeros included). */
inline constexpr float GapEpsilon = 0.01f;

[[nodiscard]] inline auto isInRangeGapCs(int32_t gapCs) -> bool {
    return gapCs >= GapLoCs && gapCs <= GapHiCs;
}

[[nodiscard]] inline auto gapIndex(int32_t gapCs) -> int {
    return gapCs + GapIndexOffset;
}

struct OnsetGapWindowStats {
    /** Smoothed consecutive onset-gap frequencies; index via gapIndex. */
    std::array<float, GapBins> sGap{};
    /** Sorted unique in-range consecutive onset gaps (drives τ schedule API). */
    std::vector<int32_t> uniqueGaps;
    /** Last onset cs in stream order within the window; -1 if none. */
    int32_t lastOnsetCs = -1;
    int noteCount = 0;
    /** Number of consecutive pairs whose gap fell in [GapLoCs, GapHiCs]. */
    int inRangeGapPairs = 0;
};

[[nodiscard]] auto collectOnsetGapWindow(const std::vector<int32_t> &denseInputData,
                                         int32_t nowCs,
                                         int32_t skipProvisionalCs = -1) -> OnsetGapWindowStats;

/** Prefix-relative window (exact "now"), like collectPitchWindowAt. */
[[nodiscard]] auto collectOnsetGapWindowAt(const std::vector<int32_t> &denseInputData,
                                           int32_t prefixNowCs,
                                           int32_t skipProvisionalCs = -1) -> OnsetGapWindowStats;

/**
 * Subtract τ·log(s_gap[t − lastOnset]) from onset logits for gaps in range.
 * `timeOffset` is the relativization offset so logit index t maps to absolute
 * onset t + timeOffset. No-op if the window is empty.
 */
auto applyOnsetLogitsBias(std::vector<float> &logits,
                          const OnsetGapWindowStats &stats,
                          float tau,
                          int32_t timeOffset) -> void;

/** Same API as DurationTauScheduler; onset-gap τ is fixed (no ramp). */
class OnsetGapTauScheduler {
public:
    struct Result {
        float tau = TauBase;
    };

    [[nodiscard]] auto evaluate(int32_t /*nowCs*/,
                                const std::vector<int32_t> & /*uniqueGaps*/,
                                float tauBase = TauBase) -> Result {
        return {tauBase};
    }

    auto reset() -> void {}
};

} // namespace DenseOnsetGapBias
