#pragma once

#include <JuceHeader.h>

#include <atomic>
#include <functional>
#include <memory>
#include <vector>

#include "VirtualOrch/DurationLogitSnapshot.h"
#include "VirtualOrch/Fifo.h"
#include "VirtualOrch/Clock.h"
#include "VirtualOrch/ModelSamplingAlert.h"
#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/OrchLoopProfile.h"
#include "VirtualOrch/ui/ModelConfigurationComponent.h"

class VoiceSeparation;

enum class MusicModelArch : uint8_t {
    Amt,
    DenseV1,
    DenseV2,
    /** Piano + piano-reduction (2 instruments): see ReductionTransformerPianoReduction. */
    DensePianoReduction
};

[[nodiscard]] inline auto isDenseMusicArch(MusicModelArch arch) -> bool {
    return arch == MusicModelArch::DenseV1 || arch == MusicModelArch::DenseV2
        || arch == MusicModelArch::DensePianoReduction;
}

/**
 * Shared base for MusicTransformer (AMT) and dense ReductionTransformerV1/V2/PianoReduction.
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

    CircularFifo<Token> *orchestrationReductionIncoming = nullptr;
    CircularFifo<TokenUpdate> *orchestrationUpdatesIncoming = nullptr;

    [[nodiscard]] auto getInputData() const -> std::vector<int32_t> { return inputData; }

    [[nodiscard]] auto getCurrentTime() const -> int32_t { return currentTime; }

    /** Called on the message thread whenever inputData changes. */
    std::function<void(std::vector<int32_t>)> onInputDataChanged;

    juce::Atomic<bool> directInputBlock = false;

    /**
     * Manual pause (Launchpad CC105 / Mode widget / sustain pedal). Stops all reduction
     * generation. Independent of overflowPause — clearing overflow never clears this.
     */
    juce::Atomic<bool> generationPause{false};

    /** Set manual generation pause and sync LEDs/UI when the value changes. */
    auto setGenerationPause(bool paused) -> void;

    /** Soft pause while generated output is too far ahead of the clock. */
    juce::Atomic<bool> overflowPause{false};

    /** True when either pause flag blocks model generation. */
    [[nodiscard]] auto isGenerationStopped() const -> bool {
        return generationPause.get() || overflowPause.get();
    }

    /** Most recent threadRun iteration duration in milliseconds. */
    [[nodiscard]] auto getLastThreadLoopMs() const -> float {
        return lastThreadLoopMs.load(std::memory_order_relaxed);
    }

    [[nodiscard]] auto getLastThreadLoopMsMax() const -> float {
        return lastThreadLoopMsMax.load(std::memory_order_relaxed);
    }

    /** Last busy reduction iteration + running max (Mode widget). */
    [[nodiscard]] auto getLastReductionProfile() const -> ReductionLoopProfile;

    /** Wall clock for ahead-of-realtime throttle (set by AppSession). */
    Clock *clock = nullptr;

    /** Invoked on the message thread when generationPause changes (LED / UI sync). */
    std::function<void()> onPausedChanged;

    /** Set by AppSession; used to surface ORT / sampling failures in the Mode widget. */
    ModelSamplingAlert *samplingAlert = nullptr;

    /** Optional online vocsep; stamps Token.voiceId inside pushOutputToken when loaded. */
    VoiceSeparation *voiceSeparation = nullptr;

    /** Latest dense duration-field logits (post-mask) for the Duration Logits UI. */
    [[nodiscard]] auto getDurationLogitSnapshot() const -> DurationLogitSnapshot;

    auto reportSamplingError(const juce::String &detail) -> void {
        if (samplingAlert != nullptr)
            samplingAlert->report("Reduction", detail);
    }

    /** Append-only copy of tokens pushed to outputTokenQueue (Step 4 UI). */
    [[nodiscard]] auto getOutputHistory() const -> std::vector<Token>;

    /** Only the tokens still reaching cutoffCs; used by the UI so refresh cost stays bounded. */
    [[nodiscard]] auto getOutputHistorySince(int32_t cutoffCs) const -> std::vector<Token>;

    auto clearOutputHistory() -> void;

    /**
     * Drop any in-flight ahead throttle. Must clear overflowPause together with the deadline:
     * zeroing only the deadline leaves isGenerationStopped() stuck true across stop/restart.
     */
    auto resetAheadThrottle() -> void;

protected:
    [[nodiscard]] auto isGeneratedTooFarAhead(int32_t generatedTime) const -> bool;

    /** Soft-pause for modelConfig.outputAheadThrottleSeconds; paused path drains input. */
    auto startAheadThrottle() -> void;

    /** Clear overflowPause when ahead-throttle window ends (does not touch generationPause). */
    auto finishAheadThrottleIfDue() -> void;

    auto notifyPausedChanged() -> void;

    /** Subclasses: ints per note in `inputData` (AMT 3, dense 4). */
    [[nodiscard]] virtual auto inputEventWidth() const -> size_t { return 3; }

    auto notifyInputDataChanged() -> void;

    auto clearInputTokenQueue() -> void;

    auto clearInputConditioningQueue() -> void;

    auto clearOutputTokenQueue() -> void;

    auto clearUpdatesFromFilter() -> void;

    /** Push to outputTokenQueue and outputHistory. */
    auto pushOutputToken(const Token &token) -> void;

    /**
     * On generationPause false→true, emit ClearQueue at clock now + 1 s so pending
     * playback onsets beyond the grace window are dropped (soft stop).
     */
    auto maybeEmitGenerationPauseSoftStopClear() -> void;

    /** Log first generated onset after generationPause clears (diagnose pedal-up lag). */
    auto logGeneratedTokenIfResumed(int32_t onset) -> void;

    auto recordThreadLoopMs(double loopStartHiResMs) -> void;

    auto resetLoopTiming() -> void;

    auto resetReductionProfile() -> void;
    auto publishBusyReductionProfile(ReductionLoopStepMs step) -> void;

    /** Copy duration-band logits from a full vocab score vector (post-mask). */
    auto publishDurationLogits(const std::vector<float> &scores, int32_t sampledDurToken) -> void;

    ModelConfig &modelConfig;
    std::vector<int32_t> inputData;
    int32_t currentTime = 0;

    /** Edge detect for maybeEmitGenerationPauseSoftStopClear (reset on thread start). */
    bool wasGenerationPaused = false;

    /** Set on pause→run; cleared after the next successful generate is logged. */
    std::atomic<bool> logFirstTokenAfterResume{false};

    uint32_t aheadThrottleUntilMs = 0;
    std::atomic<float> lastThreadLoopMs{0.0f};
    std::atomic<float> lastThreadLoopMsMax{0.0f};

    /**
     * Guards against queueing one prompt refresh per generated note: generation emits far
     * faster than the message thread can redraw, so bursts must collapse into one refresh
     * instead of backing up the message queue. Shared so a late callback outliving `this`
     * is still safe.
     */
    std::shared_ptr<std::atomic<bool>> inputDataNotifyPending =
        std::make_shared<std::atomic<bool>>(false);

    ReductionLoopStepMs loopAccum;
    mutable juce::CriticalSection reductionProfileLock;
    ReductionLoopProfile reductionProfile;

    mutable juce::CriticalSection outputHistoryLock;
    std::vector<Token> outputHistory;

    mutable juce::CriticalSection durationLogitLock;
    DurationLogitSnapshot durationLogitSnapshot;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ReductionTransformer)
};
