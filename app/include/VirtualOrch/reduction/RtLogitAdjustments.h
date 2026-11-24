#pragma once

#include <atomic>

#include "VirtualOrch/reduction/DenseDurationBias.h"
#include "VirtualOrch/reduction/DenseOnsetGapBias.h"
#include "VirtualOrch/reduction/DensePitchBias.h"

/**
 * The logit adjustments of a reduction that the Logit Adjustments widget and presets control:
 * the T3'' / W2-W3 / Hybrid transformers (TwoInstrumentAnticipatedReductionTransformer) and
 * Hybrid v1.1 (HybridInformativeWaitReductionTransformer).
 *
 *  - The main side (T3'' / W2-W3 / Hybrid's W / Hybrid v1.1's informative-WAIT model 2): the
 *    pitch, duration and onset-gap biases with their tau, the short-gap penalty, LogitTD; all OFF
 *    by default. The global BiasSettings switch and silence gate apply to the three biases.
 *  - A hybrid's silent side (v1 / model 1): its own switches, hybridV1*, all ON by default, with
 *    the base taus.
 */
class RtLogitAdjustments {
public:
    static constexpr float kDefaultWaitMinPlay = 0.30f;

    virtual ~RtLogitAdjustments() = default;

    /** Two models: the hybridV1* switches are the silent side's (the widget's side toggle). */
    [[nodiscard]] virtual auto hasSilentSide() const -> bool = 0;
    /** W2/W3: the WAIT P(play) threshold applies. */
    [[nodiscard]] virtual auto usesWaitMinPlay() const -> bool = 0;

    [[nodiscard]] auto isPitchBiasEnabled() const -> bool { return pitchBiasEnabled.load(); }
    auto setPitchBiasEnabled(bool on) -> void { pitchBiasEnabled.store(on); }
    [[nodiscard]] auto isDurationBiasEnabled() const -> bool { return durationBiasEnabled.load(); }
    auto setDurationBiasEnabled(bool on) -> void { durationBiasEnabled.store(on); }
    [[nodiscard]] auto isOnsetGapBiasEnabled() const -> bool { return onsetGapBiasEnabled.load(); }
    auto setOnsetGapBiasEnabled(bool on) -> void { onsetGapBiasEnabled.store(on); }
    [[nodiscard]] auto getPitchTauBase() const -> float { return pitchTau.load(); }
    [[nodiscard]] auto getPitchTauPrimeBase() const -> float { return pitchTauPrime.load(); }
    [[nodiscard]] auto getDurationTauBase() const -> float { return durationTau.load(); }
    [[nodiscard]] auto getDurationTauPrimeBase() const -> float { return durationTauPrime.load(); }
    [[nodiscard]] auto getOnsetGapTauBase() const -> float { return onsetGapTau.load(); }
    auto setPitchTauBase(float t) -> void { pitchTau.store(t); }
    auto setPitchTauPrimeBase(float t) -> void { pitchTauPrime.store(t); }
    auto setDurationTauBase(float t) -> void { durationTau.store(t); }
    auto setDurationTauPrimeBase(float t) -> void { durationTauPrime.store(t); }
    auto setOnsetGapTauBase(float t) -> void { onsetGapTau.store(t); }
    /** The 1..8 cs short-gap penalty (TdOnsetBias::shortGapPenalty). */
    [[nodiscard]] auto isShortGapPenaltyEnabled() const -> bool { return shortGapPenalty.load(); }
    auto setShortGapPenaltyEnabled(bool on) -> void { shortGapPenalty.store(on); }
    /** LogitTD for this model (its own switch, not the global one). */
    [[nodiscard]] auto isLogitTdEnabled() const -> bool { return logitTd.load(); }
    auto setLogitTdEnabled(bool on) -> void { logitTd.store(on); }
    /** W2/W3 threshold rule: play when P(play) >= this (0.30 recommended). */
    [[nodiscard]] auto getWaitMinPlay() const -> float { return waitMinPlay.load(); }
    auto setWaitMinPlay(float p) -> void { waitMinPlay.store(p); }

    /** A hybrid's silent side: its own logit adjustments, all ON by default. */
    std::atomic<bool> hybridV1PitchBias{true};
    std::atomic<bool> hybridV1DurationBias{true};
    std::atomic<bool> hybridV1OnsetGapBias{true};
    std::atomic<bool> hybridV1ShortGapPenalty{true};
    std::atomic<bool> hybridV1LogitTd{true};

protected:
    std::atomic<bool> pitchBiasEnabled{false};
    std::atomic<bool> durationBiasEnabled{false};
    std::atomic<bool> onsetGapBiasEnabled{false};
    std::atomic<float> pitchTau{DensePitchBias::TauBase};
    std::atomic<float> pitchTauPrime{DensePitchBias::TauBase};
    std::atomic<float> durationTau{DenseDurationBias::TauBase};
    std::atomic<float> durationTauPrime{DenseDurationBias::TauPrimeBase};
    std::atomic<float> onsetGapTau{DenseOnsetGapBias::TauBase};
    std::atomic<bool> shortGapPenalty{false};
    std::atomic<bool> logitTd{false};
    std::atomic<float> waitMinPlay{kDefaultWaitMinPlay};
};
