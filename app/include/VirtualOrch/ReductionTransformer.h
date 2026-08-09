#pragma once

#include <JuceHeader.h>

#include <functional>
#include <vector>

#include "Fifo.h"
#include "VirtualOrch/Clock.h"
#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/ui/ModelConfigurationComponent.h"

enum class MusicModelArch : uint8_t {
    Amt,
    Dense
};

/**
 * Shared base for MusicTransformer (AMT) and DenseMusicTransformer.
 */
class ReductionTransformer : public juce::Thread {
public:
    ReductionTransformer(const juce::String &threadName, ModelConfig &modelConfigIn);

    ~ReductionTransformer() override = default;

    [[nodiscard]] virtual auto isModelLoaded() const -> bool = 0;

    CircularFifo<Token> inputTokenQueue;
    CircularFifo<Token> inputConditioningQueue;
    CircularFifo<Token> outputTokenQueue;
    CircularFifo<TokenUpdate> updatesFromFilter;

    CircularFifo<Token> *orchestrationMidiIncoming = nullptr;
    CircularFifo<Token> *orchestrationConditioningIncoming = nullptr;
    CircularFifo<Token> *orchestrationReductionIncoming = nullptr;
    CircularFifo<TokenUpdate> *orchestrationUpdatesIncoming = nullptr;

    [[nodiscard]] auto getInputData() const -> std::vector<int32_t> { return inputData; }

    [[nodiscard]] auto getCurrentTime() const -> int32_t { return currentTime; }

    /** Called on the message thread whenever inputData changes. */
    std::function<void(std::vector<int32_t>)> onInputDataChanged;

    juce::Atomic<bool> directInputBlock = false;

    /** Soft pause: drain/apply input continues; model generation is skipped. */
    juce::Atomic<bool> paused{false};

    /** Wall clock for ahead-of-realtime throttle (set by AppSession). */
    Clock *clock = nullptr;

    /** Invoked on the message thread when pause changes due to ahead throttle. */
    std::function<void()> onPausedChanged;

    /** Append-only copy of tokens pushed to outputTokenQueue (Step 4 UI). */
    [[nodiscard]] auto getOutputHistory() const -> std::vector<Token>;

    auto clearOutputHistory() -> void;

protected:
    [[nodiscard]] auto isGeneratedTooFarAhead(int32_t generatedTime) const -> bool;

    /** Soft-pause for modelConfig.outputAheadThrottleSeconds; paused path drains input. */
    auto startAheadThrottle() -> void;

    /** Restore pause state when ahead-throttle window ends. */
    auto finishAheadThrottleIfDue() -> void;

    auto notifyPausedChanged() -> void;

    auto notifyInputDataChanged() -> void;

    auto clearInputTokenQueue() -> void;

    auto clearInputConditioningQueue() -> void;

    auto clearOutputTokenQueue() -> void;

    auto clearUpdatesFromFilter() -> void;

    /** Push to outputTokenQueue and outputHistory. */
    auto pushOutputToken(const Token &token) -> void;

    ModelConfig &modelConfig;
    std::vector<int32_t> inputData;
    int32_t currentTime = 0;

    uint32_t aheadThrottleUntilMs = 0;
    bool pausedBeforeAheadThrottle = false;

    mutable juce::CriticalSection outputHistoryLock;
    std::vector<Token> outputHistory;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ReductionTransformer)
};
