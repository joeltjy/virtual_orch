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
 * often each pitch (π) and each in-range pitch leap (s_y) appears, then subtract
 * τ·log(π) and τ′·log(s_y) from note logits so overused pitches/intervals become
 * less likely. τ holds/ramps after S gains a pitch; τ′ after the window gains a
 * new pitch interval (Δ ∈ [-12, 12]). Aging out alone does not reset either.
 *
 * Full walkthrough: PITCH_BIAS.md in this folder.
 */
namespace DensePitchBias {

inline constexpr int32_t WindowCs = 500; // 5 s
inline constexpr float UnigramEpsilon = 1.0f / 128.0f;
inline constexpr float DeltaEpsilon = 0.01f;
inline constexpr float TauBase = 0.2f;
/** Hold at base until this many seconds *or* TauHoldNotes, whichever is earlier. */
inline constexpr float TauHoldSeconds = 2.0f;
inline constexpr int TauHoldNotes = 10;
/** After hold: τ = base · TauRampBase^(notesSinceReset / 10). */
inline constexpr float TauRampBase = 2.0f;

/** amt_causal / offline: only leaps in [-12, 12] enter D and s_y. */
inline constexpr int DeltaLo = -12;
inline constexpr int DeltaHi = 12;
inline constexpr int DeltaBins = DeltaHi - DeltaLo + 1; // 25
inline constexpr int DeltaOffset = -DeltaLo;            // index = delta + 12

[[nodiscard]] inline auto isInRangeDelta(int delta) -> bool {
    return delta >= DeltaLo && delta <= DeltaHi;
}

[[nodiscard]] inline auto deltaIndex(int delta) -> int {
    return delta + DeltaOffset;
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
    /** Smoothed consecutive pitch-delta frequencies for Δ ∈ [-12, 12]; index = Δ + 12. */
    std::array<float, DeltaBins> sDelta{};
    /** Sorted unique pitches in the window (S); drives the τ schedule. */
    std::vector<int32_t> uniquePitches;
    /** Sorted unique in-range consecutive pitch deltas (Δ); drives τ′. */
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
[[nodiscard]] auto collectPitchWindow(const std::vector<int32_t> &denseInputData,
                                      int32_t nowCs,
                                      int32_t skipProvisionalCs = -1) -> PitchWindowStats;

/**
 * Same window using an exact prefix "now" (the just-sampled onset). Offline
 * amt_causal applies τ / π after onset+duration are in the prefix — call this
 * before the note-field logit adjustment, not only at the start of the note.
 */
[[nodiscard]] auto collectPitchWindowAt(const std::vector<int32_t> &denseInputData,
                                        int32_t prefixNowCs,
                                        int32_t skipProvisionalCs = -1) -> PitchWindowStats;

/**
 * Subtract τ·log(π[p]) and τ′·log(s_y[p−lastPitch]) from every instrument band's
 * pitch logits. τ′ term only when |p−lastPitch| ≤ 12. No-op when the window is
 * empty. Leaves non-finite entries alone.
 */
auto applyNoteLogitsBias(std::vector<float> &logits,
                         const PitchWindowStats &stats,
                         float tau,
                         float tauPrime) -> void;

/**
 * Strength of the pitch / interval bias over time (independent clocks):
 * - τ: after S *gains* a pitch → hold base for min(2 s, 10 notes), then
 *   base · 2^(notes/10).
 * - τ′: after the window *gains* an in-range pitch interval Δ → same hold/ramp.
 * Shrinkage alone (pitches / intervals leaving) does not reset.
 * Empty S / D snaps back to base.
 */
class PitchTauScheduler {
public:
    struct Result {
        float tau = TauBase;
        float tauPrime = TauBase;
    };

    [[nodiscard]] auto evaluate(int32_t nowCs,
                                const std::vector<int32_t> &uniquePitches,
                                const std::vector<int32_t> &uniqueDeltas,
                                float tauBase = TauBase,
                                float tauPrimeBase = TauBase,
                                bool countNote = false) -> Result;

    [[nodiscard]] auto lastUniquePitches() const -> const std::vector<int32_t> & { return lastS; }

    [[nodiscard]] auto lastUniqueDeltas() const -> const std::vector<int32_t> & { return lastDeltas; }

    auto reset() -> void;

private:
    std::vector<int32_t> lastS;
    std::vector<int32_t> lastDeltas;
    int32_t sStableSinceCs = 0;
    int32_t deltaStableSinceCs = 0;
    int32_t lastScheduleNowCs = 0;
    int sNotesSinceReset = 0;
    int deltaNotesSinceReset = 0;
    bool hasS = false;
    bool hasDeltas = false;
    bool hasScheduleNow = false;
};

} // namespace DensePitchBias
