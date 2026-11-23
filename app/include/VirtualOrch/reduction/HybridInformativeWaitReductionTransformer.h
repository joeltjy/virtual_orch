#pragma once

#include <JuceHeader.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/reduction/InformativeWaitEngine.h"
#include "VirtualOrch/reduction/ReductionTransformer.h"
#include "VirtualOrch/reduction/StreamingKvCache.h"

/**
 * Hybrid v1.1: overnight model 1 (continuation, vocab 11768) while the pianist is silent, the informative-WAIT
 * model (overnight model 2, vocab 11879) while they play. The switching rules are W3's HybridTransformer's
 * (TwoInstrumentAnticipatedReductionTransformer::Variant::Hybrid):
 *   - a pedal release (or un-pause) switches to model 1 at once;
 *   - pedal up and the user playing for 1 s ("playing" = a key down or the last release < 0.5 s ago): model 2;
 *   - on model 2 and the user silent for 0.5 s: every note generated from now on is cut and model 1 takes over;
 *   - on model 2, not paused, nothing generated sounding now and the next generated onset >= 2 s ahead (or nothing
 *     queued for 2 s): the same cut, model 1 takes over and the 1 s playing timer starts again.
 * Each model has its own engine and token stream (their vocabularies differ); both get every piano note, and each
 * generated note is mirrored into the other model's context, so both always see the same music.
 */
class HybridInformativeWaitReductionTransformer : public ReductionTransformer {
public:
    static constexpr int64_t kPlayToWaitMs = 1000;
    static constexpr int64_t kStopMs = 500;
    static constexpr int32_t kPredictedSilenceCs = 200;
    enum class Reason { Start, PedalReleased, Playing, Stopped, PredictedSilence };

    explicit HybridInformativeWaitReductionTransformer(ModelConfig &modelConfig);

    void run() override {
        threadInit();
        threadRun();
        threadStop();
    }
    void threadInit() {}
    void threadRun();
    void threadStop() {}

    /** Load model 1 (silent side, vocab 11768) and model 2 (playing side, vocab 11879). */
    auto init(const char *silentModelPath, const char *waitModelPath) -> bool;
    /** Tests: other backends (owned). */
    auto initWithBackends(std::unique_ptr<StreamingKv::Backend> silent, std::unique_ptr<StreamingKv::Backend> wait)
        -> void;
    std::string lastInitError;

    [[nodiscard]] auto isModelLoaded() const -> bool override { return silentBackend != nullptr && waitBackend != nullptr; }
    [[nodiscard]] auto inputEventWidth() const -> size_t override { return 4; }

    /** Model 2's mode (unconstrained by default). */
    auto setWaitMode(InformativeWaitEngine::Mode mode) -> void { waitMode.store(static_cast<int>(mode)); }
    auto setPrecompute(bool on) -> void { precompute.store(on); }
    [[nodiscard]] auto usesSilentModel() const -> bool { return silentActive.load(); }
    [[nodiscard]] auto reason() const -> Reason { return reasonValue.load(); }

    /** Clock / wall clock (tests replace them). */
    std::function<int32_t()> nowCsProvider;
    std::function<int64_t()> wallMsProvider;
    /** One loop iteration without waiting (tests). True when notes were produced. */
    auto stepOnce() -> bool;
    std::string silentName = "model 1";
    std::string waitName = "informative WAIT";

private:
    auto onGenerationPauseChanged(bool paused) -> void override {
        if (! paused)
            pedalReleased.store(true);
    }
    auto nowCs() const -> int32_t;
    auto applySettings() -> void;
    auto resetStreams(int32_t now) -> void;
    auto drainInput() -> bool;
    auto updateHybrid(int32_t now) -> void;
    auto setSilent(bool silent, Reason why, int32_t now) -> void;
    auto cutGeneratedFrom(int32_t now) -> void;
    [[nodiscard]] auto predictsSilenceFrom(int32_t now) const -> bool;
    auto generate(int32_t now) -> bool;

    std::unique_ptr<StreamingKv::Backend> silentBackend;
    std::unique_ptr<StreamingKv::Backend> waitBackend;
    InformativeWaitEngine silentEngine;
    InformativeWaitEngine waitEngine;

    std::atomic<bool> silentActive{true};
    std::atomic<Reason> reasonValue{Reason::Start};
    std::atomic<bool> pedalReleased{false};
    std::atomic<int> waitMode{0};
    std::atomic<bool> precompute{true};
    int64_t playingSinceMs = -1;
    int32_t waitSinceCs = INT32_MIN;
    bool streamsStarted = false;
};
