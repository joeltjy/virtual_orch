#pragma once

#include <JuceHeader.h>

#include <atomic>
#include <memory>
#include <string>

#include "VirtualOrch/reduction/EventEngine.h"
#include "VirtualOrch/reduction/EventReductionTransformer.h"
#include "VirtualOrch/reduction/RtLogitAdjustments.h"
#include "VirtualOrch/reduction/StreamingKvCache.h"

/**
 * Hybrid v1.3: the event-tokenized pair (event_pair_proposal.txt). Model 1 event (m1e, continuation_events_m1e_*,
 * vocab 12024, OFF0 11768) while the pianist is silent; model 2 event (m2e, simultaneous_events_chordtokens_m2e_*,
 * vocab 14328, OFF0 14071, STACK_OFF 14327, the wait3 ids) while they play. The switching rules are v1.1's
 * (HybridInformativeWaitReductionTransformer):
 *   - a pedal release (or un-pause) switches to model 1 at once;
 *   - pedal up and the user playing for 1 s: model 2;
 *   - on model 2 and the user silent for 0.5 s: every note generated from now on is cut and model 1 takes over.
 * (v1.1's "2 s of predicted silence" rule is left out: a v1.3 note's length is only known at its OFF.)
 * Both engines get every piano event and each other's orchestra events, so both see the same music.
 *
 * Model 2 runs EventEngine::Settings::chordTokens: [WAIT d][K k,+1], the chord block at a fire (decoded with the
 * preset's voicing bank), STACK_OFF, [LATE d] / [EXPIRED] at a close, then a re-wait or a give-up. Model 2 draws
 * nothing for the first startDelayMs of a performance, as v1.2's.
 */
class HybridEventReductionTransformer : public EventReductionTransformer, public RtLogitAdjustments {
public:
    /**
     * Logit adjustments (Nov 25 2026, Joel: pitch and onset gap on by default for v1.3, both models, the same
     * toggles as v1.2): model 1 event takes the hybridV1* switches, model 2 event the "model" ones (presets,
     * Settings / Logit Adjustments widgets). Pitch and onset gap only: the event models have no duration token,
     * and short-gap / LogitTD are not wired into the event engine yet.
     */
    [[nodiscard]] auto hasSilentSide() const -> bool override { return true; }
    [[nodiscard]] auto usesWaitMinPlay() const -> bool override { return false; }

    static constexpr int64_t kPlayToWaitMs = 1000;
    static constexpr int64_t kStopMs = 500;
    static constexpr int32_t kSimultaneousVelocityBoost = 16;
    static constexpr int32_t kM2eOff0 = 14071;
    static constexpr int32_t kM2eStackOff = 14327;
    static constexpr int32_t kM2eVocab = 14328;

    enum class Reason { Start, PedalReleased, Playing, Stopped, Accompaniment };

    explicit HybridEventReductionTransformer(ModelConfig &modelConfig);

    void run() override { threadRun(); }
    void threadRun();

    /** Load model 1 event (vocab 12024) and model 2 event (vocab 14328). */
    auto init(const char *silentModelPath, const char *waitModelPath) -> bool;
    /** Tests: other backends (owned). */
    auto initWithBackends(std::unique_ptr<StreamingKv::Backend> silent, std::unique_ptr<StreamingKv::Backend> wait) -> void;
    std::string lastInitError;
    [[nodiscard]] auto isModelLoaded() const -> bool override { return silentBackend != nullptr && waitBackend != nullptr; }

    /** Model 2's voicing bank (CHORD blocks -> band notes); false if unreadable (the chord tones are stacked). */
    auto loadVoicingBank(const std::string &path) -> bool;
    auto setVoicingBank(std::shared_ptr<const VoicingBank> bank) -> void { voicings = std::move(bank); }
    /** Model 2's start delay (ms). */
    auto setStartDelayMs(int32_t ms) -> void { startDelayMs.store(ms); }
    /** Orchestra silent this long: the model's piano tokens are masked (spec section 8). */
    auto setForceAfterMs(int32_t ms) -> void { forceAfterMs.store(ms); }

    [[nodiscard]] auto usesSilentModel() const -> bool { return silentActive.load(); }
    [[nodiscard]] auto reason() const -> Reason { return reasonValue.load(); }
    [[nodiscard]] auto label() const -> const char * { return "Hybrid v1.3"; }

    std::function<int64_t()> wallMsProvider;
    /** One loop iteration without waiting (tests). True when an orchestra event went out. */
    auto stepOnce() -> bool;

    [[nodiscard]] auto silentEngineForTest() const -> const EventEngine & { return silentEngine; }
    [[nodiscard]] auto waitEngineForTest() const -> const EventEngine & { return waitEngine; }

private:
    auto onGenerationPauseChanged(bool paused) -> void override {
        if (! paused)
            pedalReleased.store(true);
    }
    auto applySettings() -> void;
    auto resetStreams(int32_t now) -> void;
    auto feedPiano(int32_t now) -> void;
    auto updateHybrid(int32_t now) -> void;
    auto setSilent(bool silent, Reason why, int32_t now) -> void;
    auto cutGeneratedFrom(int32_t now) -> void;
    auto generate(int32_t now) -> bool;
    [[nodiscard]] auto wallMs() const -> int64_t;

    std::unique_ptr<StreamingKv::Backend> silentBackend;
    std::unique_ptr<StreamingKv::Backend> waitBackend;
    EventEngine silentEngine;
    EventEngine waitEngine;
    std::atomic<bool> silentActive{true};
    std::atomic<Reason> reasonValue{Reason::Start};
    std::atomic<bool> pedalReleased{false};
    std::atomic<int32_t> startDelayMs{3000};
    std::atomic<int32_t> forceAfterMs{2000};
    std::shared_ptr<const VoicingBank> voicings;
    int64_t playingSinceMs = -1;
    int32_t startedAtCs = 0;
    uint64_t silentSeed = 1;
    uint64_t waitSeed = 2;
    bool streamsStarted = false;
};
