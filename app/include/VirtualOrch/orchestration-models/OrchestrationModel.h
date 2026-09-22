#pragma once

#include <array>
#include <string>
#include <vector>

#include "VirtualOrch/ModelSamplingAlert.h"
#include "VirtualOrch/MusicTransformer.h"
#include "VirtualOrch/OrchestrationBalanceTracker.h"
#include "VirtualOrch/OrchLoopProfile.h"
#include "VirtualOrch/InstrumentLogitSnapshot.h"

struct OrchestrationNote {
    Token token;
    int32_t velocity = 0;
    int32_t localInstrumentId = 0;
};

/**
 * Base class for orchestration models (non-threaded computation only).
 */
class OrchestrationModel {
public:
    static constexpr size_t CONDITIONING_SIGNAL_DIM = 64;
    using ConditioningSignal = std::array<float, CONDITIONING_SIGNAL_DIM>;

    virtual ~OrchestrationModel() = default;

    [[nodiscard]] virtual auto getName() const -> std::string = 0;

    [[nodiscard]] virtual auto tokenHistory() const -> const std::vector<OrchestrationNote> & = 0;

    /**
     * @param balance  When non-null, record per-note instrument proportions after decode.
     * @param bias     When non-null and bias->apply, adjust family-combo logits before sample.
     */
    [[nodiscard]] virtual auto getOutput(
        const std::vector<Token> &incomingTokens,
        const std::vector<int32_t> &instruments,
        const ConditioningSignal &conditioningSignal,
        OrchestrationBalanceTracker *balance = nullptr,
        const OrchestrationBalanceTracker::BiasView *bias = nullptr)
        -> std::vector<OrchestrationNote> = 0;

    int32_t maxContextLength = 256;
    std::vector<int32_t> tokenHistorySequence;

    /** Set by AppSession; used to surface ORT / sampling failures in the Mode widget. */
    ModelSamplingAlert *samplingAlert = nullptr;

    /** Accumulator for getOutput step timings; OT resets each iteration. */
    OrchLoopStepMs *getOutputTimings = nullptr;

    /**
     * Optional sink for post-bias singleton instrument logits (debug bar chart).
     * Written by InstrumentCombinations after masking/bias, before sample.
     */
    InstrumentLogitSnapshot *instrumentLogitSink = nullptr;

protected:
    auto reportSamplingError(const juce::String &detail) -> void {
        if (samplingAlert != nullptr)
            samplingAlert->report(juce::String(getName()), detail);
    }
};
