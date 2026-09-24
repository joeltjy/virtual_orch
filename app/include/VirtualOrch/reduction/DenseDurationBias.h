#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "VirtualOrch/reduction/DenseTypes.h"

/**
 * Online duration-bias for ReductionTransformerV2 (mirrors DensePitchBias).
 *
 * Last 5 s of notes → π over legal duration cs (multiples of DurationStepCs) and
 * s_y over consecutive duration deltas (also step-aligned, |Δ| ≤ DeltaAbsMaxCs).
 * Subtract τ·log(π) and τ′·log(s_y) from duration logits.
 *
 * τ_d hold/ramps like pitch: hold base for min(2 s, 10 notes), then
 * base · TauRampPerSecond^(notes/10) (pitch uses 2^(notes/10)).
 * τ′_d defaults to TauPrimeBase (0) unless raised in the UI.
 */
namespace DenseDurationBias {

inline constexpr int32_t WindowCs = 500; // 5 s
inline constexpr float TauBase = 0.5f;
/** Hold at base until this many seconds *or* TauHoldNotes, whichever is earlier. */
inline constexpr float TauHoldSeconds = 2.0f;
inline constexpr int TauHoldNotes = 10;
/** After hold: τ_d = base · TauRampPerSecond^(notesSinceReset / 10). */
inline constexpr float TauRampPerSecond = 1.5f;
/** Default τ′ for consecutive duration-delta bias (off unless raised in UI). */
inline constexpr float TauPrimeBase = 0.0f;

/** Consecutive duration-delta range for s_y / τ′ (centiseconds, step-aligned). */
inline constexpr int32_t DeltaAbsMaxCs = 200; // ±2 s
inline constexpr int32_t DeltaLoCs = -DeltaAbsMaxCs;
inline constexpr int32_t DeltaHiCs = DeltaAbsMaxCs;
/** Number of step-aligned bins in [DeltaLoCs, DeltaHiCs]. */
inline constexpr int DeltaBins =
    (DeltaHiCs - DeltaLoCs) / DenseQuantize::DurationStepCs + 1; // 81
inline constexpr int DeltaIndexOffset =
    -DeltaLoCs / DenseQuantize::DurationStepCs; // index = Δ/step + offset

/** Legal duration grid [DurationMinCs, DurationMaxCs] step DurationStepCs. */
inline constexpr int LegalDurBins =
    (DenseQuantize::DurationMaxCs - DenseQuantize::DurationMinCs)
        / DenseQuantize::DurationStepCs
    + 1; // 198

inline constexpr float UnigramEpsilon = 1.0f / static_cast<float>(LegalDurBins);
inline constexpr float DeltaEpsilon = 0.01f;

[[nodiscard]] inline auto isLegalDurationCs(int32_t cs) -> bool {
    return DenseQuantize::isLegalDurationCs(cs);
}

[[nodiscard]] inline auto isInRangeDeltaCs(int32_t deltaCs) -> bool {
    return deltaCs >= DeltaLoCs && deltaCs <= DeltaHiCs
           && (deltaCs % DenseQuantize::DurationStepCs) == 0;
}

[[nodiscard]] inline auto deltaIndex(int32_t deltaCs) -> int {
    return deltaCs / DenseQuantize::DurationStepCs + DeltaIndexOffset;
}

[[nodiscard]] inline auto legalDurIndex(int32_t cs) -> int {
    return (cs - DenseQuantize::DurationMinCs) / DenseQuantize::DurationStepCs;
}

[[nodiscard]] inline auto durationCsFromToken(int32_t durToken) -> int32_t {
    return durToken - static_cast<int32_t>(DenseVocab::DurOffset);
}

struct DurationWindowStats {
    /** Smoothed duration frequencies over legal grid (sums ≈ 1). */
    std::array<float, LegalDurBins> pi{};
    /** Smoothed consecutive duration-delta frequencies; index via deltaIndex. */
    std::array<float, DeltaBins> sDelta{};
    /** Sorted unique legal durations in the window (drives τ). */
    std::vector<int32_t> uniqueDurations;
    /** Sorted unique in-range consecutive duration deltas (drives τ′). */
    std::vector<int32_t> uniqueDeltas;
    /** Last duration cs in stream order within the window; -1 if none. */
    int32_t lastDurationCs = -1;
    int noteCount = 0;
};

[[nodiscard]] auto collectDurationWindow(const std::vector<int32_t> &denseInputData,
                                         int32_t nowCs,
                                         int32_t skipProvisionalCs = -1,
                                         int32_t onlyInstrument = -1) -> DurationWindowStats;

/** Prefix-relative window (just-sampled onset as now), like collectPitchWindowAt. */
[[nodiscard]] auto collectDurationWindowAt(const std::vector<int32_t> &denseInputData,
                                           int32_t prefixNowCs,
                                           int32_t skipProvisionalCs = -1,
                                           int32_t onlyInstrument = -1) -> DurationWindowStats;

/**
 * Subtract τ·log(π[d]) and τ′·log(s_y[d−lastDur]) from legal duration logits.
 * τ′ only when Δ is step-aligned and |Δ| ≤ DeltaAbsMaxCs. No-op if window empty.
 */
auto applyDurationLogitsBias(std::vector<float> &logits,
                             const DurationWindowStats &stats,
                             float tau,
                             float tauPrime) -> void;

/** Hold-then-ramp for τ_d / τ′_d; slower growth than PitchTauScheduler. */
class DurationTauScheduler {
public:
    struct Result {
        float tau = TauBase;
        float tauPrime = TauPrimeBase;
    };

    [[nodiscard]] auto evaluate(int32_t nowCs,
                                const std::vector<int32_t> &uniqueDurations,
                                const std::vector<int32_t> &uniqueDeltas,
                                float tauBase = TauBase,
                                float tauPrimeBase = TauPrimeBase,
                                bool countNote = false) -> Result;

    auto reset() -> void;

private:
    std::vector<int32_t> lastDurations;
    std::vector<int32_t> lastDeltas;
    int32_t durationStableSinceCs = 0;
    int32_t deltaStableSinceCs = 0;
    int32_t lastScheduleNowCs = 0;
    int durationNotesSinceReset = 0;
    int deltaNotesSinceReset = 0;
    bool hasDurations = false;
    bool hasDeltas = false;
    bool hasScheduleNow = false;
};

} // namespace DenseDurationBias