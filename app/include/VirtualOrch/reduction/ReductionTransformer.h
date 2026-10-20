#pragma once

#include <JuceHeader.h>

#include <atomic>
#include <optional>
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
    DensePianoReduction,
    /** Anticipation-style piano -> reduction (RoPE + KV): TwoInstrumentAnticipatedReductionTransformer. */
    DenseAnticipatedPiano,
    /** mb melody first, then double-anticipation reduction: MelodyFirstReductionTransformer. */
    DenseMelodyFirst
};

[[nodiscard]] inline auto isDenseMusicArch(MusicModelArch arch) -> bool {
    return arch == MusicModelArch::DenseV1 || arch == MusicModelArch::DenseV2
        || arch == MusicModelArch::DensePianoReduction
        || arch == MusicModelArch::DenseAnticipatedPiano
        || arch == MusicModelArch::DenseMelodyFirst;
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

    /** Where vocsep re-stamps of already-sent notes go (the OT's restampsIncoming). */
    CircularFifo<VoiceRestamp> *orchestrationRestampsIncoming = nullptr;

    [[nodiscard]] auto getInputData() const -> std::vector<int32_t> { return inputData; }

    [[nodiscard]] auto getCurrentTime() const -> int32_t { return currentTime; }

    /** Called on the message thread whenever inputData changes. */
    std::function<void(std::vector<int32_t>)> onInputDataChanged;

    juce::Atomic<bool> directInputBlock = false;

    /**
     * Manual pause (Launchpad CC105 / Mode widget / pedal-keyboard CC64). Stops all
     * reduction generation. Independent of overflowPause — clearing overflow never clears this.
     */
    juce::Atomic<bool> generationPause{false};

    /** Set manual generation pause and sync LEDs/UI when the value changes. */
    auto setGenerationPause(bool paused) -> void;

    /**
     * Sustain-pedal press at clock time `pedalTimeCs` (MIDI thread). The reduction thread turns
     * it into a ClearQueue token stamped with that time (see PedalClear.h), once per press,
     * also when generation was already paused. Call before setGenerationPause(true).
     */
    auto requestPedalClear(int32_t pedalTimeCs) -> void {
        pedalClearTimeCs.store(pedalTimeCs, std::memory_order_release);
    }

    /**
     * Pedal / pause interrupted the live bias clocks. Dense V2 / Piano Reduction
     * override to snap τ schedules back to base so the first note after resume
     * is not hit with a silence-aged ramp.
     */
    virtual auto onGenerationPauseChanged(bool /*paused*/) -> void {}

    /** Soft pause while generated output is too far ahead of the clock. */
    juce::Atomic<bool> overflowPause{false};

    /** True when either pause flag blocks model generation. */
    [[nodiscard]] auto isGenerationStopped() const -> bool {
        return generationPause.get() || overflowPause.get();
    }

    /**
     * True when live MIDI notes in inputData still carry the provisional hold duration
     * (inputDuration). Generation should wait until note-offs snap them — e.g. after
     * sustain pedal up while held notes are still unresolved.
     */
    [[nodiscard]] auto hasUnresolvedProvisionalInputNotes() const -> bool;

    /** Accompaniment mode: at most this many generated notes sound at once (0 = no cap). */
    auto setMaxPolyphony(int32_t cap) -> void { maxPolyphony.store(cap); }
    [[nodiscard]] auto getMaxPolyphony() const -> int32_t { return maxPolyphony.load(); }

    /**
     * Accompaniment mode: live input is added to the context without rewinding the generated
     * notes after it, and without the input-triggered ClearQueue. Set on every mode change.
     */
    auto setKeepQueueOnInput(bool keep) -> void {
        keepQueueOnInput.store(keep, std::memory_order_relaxed);
    }
    [[nodiscard]] auto keepsQueueOnInput() const -> bool {
        return keepQueueOnInput.load(std::memory_order_relaxed);
    }

    /**
     * Leaving Accompaniment at clock time `timeCs` (message / MIDI thread): the reduction thread
     * drops its generated notes from that time on and sends the cut marker downstream
     * (a PedalClear ClearQueue token, which ends OutputPlayback's drop window).
     */
    auto requestModeCut(int32_t timeCs) -> void {
        modeCutTimeCs.store(timeCs, std::memory_order_release);
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
     * Earliest onset the next generated note may have: currentTime, raised by the accompaniment
     * polyphony cap (generated notes still sounding at currentTime, see PolyphonyCap.h).
     */
    [[nodiscard]] auto onsetLowerBoundCs() const -> int32_t;

    /**
     * Vocsep changed the voice of the output-history note at `historyIndex`: rewrite it in
     * outputHistory and tell the orchestration side, which got that note earlier.
     */
    auto forwardRestamp(size_t historyIndex, int32_t newVoiceId) -> void;

    std::atomic<int32_t> nextNoteId{0};

    /**
     * On generationPause false→true, emit ClearQueue at clock now + 1 s so pending
     * playback onsets beyond the grace window are dropped (soft stop).
     */
    auto maybeEmitGenerationPauseSoftStopClear() -> void;

    /**
     * Reduction thread: a pending mode cut, if any (consumed). Drops outputHistory notes from
     * the cut time on and pushes the cut marker; the caller rewinds its own context.
     */
    [[nodiscard]] auto consumeModeCut() -> std::optional<int32_t>;

    /**
     * Drop the events of the strided inputData (inputEventWidth()) whose onset is at or after
     * `timeCs` (the generated notes ahead of the clock; live input is never ahead), and move
     * currentTime back to `timeCs`.
     */
    auto rewindInputDataFrom(int32_t timeCs) -> void;

    /** Remove the `width`-wide events of `strided` whose onset (first field) is >= timeCs. */
    static auto eraseStridedFrom(std::vector<int32_t> &strided, size_t width, int32_t timeCs)
        -> void;

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
    std::atomic<int32_t> maxPolyphony{0};

    /** Edge detect for maybeEmitGenerationPauseSoftStopClear (reset on thread start). */
    bool wasGenerationPaused = false;

    /** Set on pause→run; cleared after the next successful generate is logged. */
    std::atomic<bool> logFirstTokenAfterResume{false};

    /** Pending pedal press (cs), -1 when none; consumed by maybeEmitGenerationPauseSoftStopClear. */
    std::atomic<int32_t> pedalClearTimeCs{-1};

    std::atomic<bool> keepQueueOnInput{false};
    /** Pending mode cut (cs), -1 when none; consumed by consumeModeCut. */
    std::atomic<int32_t> modeCutTimeCs{-1};

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
