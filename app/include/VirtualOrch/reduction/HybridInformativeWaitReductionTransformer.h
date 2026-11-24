#pragma once

#include <JuceHeader.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/reduction/InformativeWaitEngine.h"
#include "VirtualOrch/reduction/ReductionTransformer.h"
#include "VirtualOrch/reduction/RtLogitAdjustments.h"
#include "VirtualOrch/reduction/StreamingKvCache.h"
#include "VirtualOrch/reduction/TdController.h"
#include "VirtualOrch/reduction/WaitTracker.h"

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
 *
 * Dynamic LogitTD as in W3's Hybrid: one TdController per model ("m1", "m2"), ticked every TdController::kTickMs
 * from the orchestra's measured density, each updated only while its model draws (m2 frozen while it waits on a
 * WAIT); the gamma goes to that model's onset draws (InformativeWaitEngine::Settings::tdGamma). Both go back to
 * gamma 0 on a stream restart and a pedal release. TD AUTO: no shift.
 *
 * Logit adjustments as the legacy Hybrid (RtLogitAdjustments): model 1 takes the silent side's (hybridV1*, all ON,
 * base taus), model 2 the main ones (all OFF, the widget's taus); the BiasSettings switch and silence gate apply to
 * the three biases; both draw within the Launchpad register range.
 */
class HybridInformativeWaitReductionTransformer : public ReductionTransformer, public RtLogitAdjustments {
public:
    static constexpr int64_t kPlayToWaitMs = 1000;
    static constexpr int64_t kStopMs = 500;
    static constexpr int32_t kPredictedSilenceCs = 200;
    enum class Reason { Start, PedalReleased, Playing, Stopped, PredictedSilence, Accompaniment };

    explicit HybridInformativeWaitReductionTransformer(ModelConfig &modelConfig);

    [[nodiscard]] auto hasSilentSide() const -> bool override { return true; }
    [[nodiscard]] auto usesWaitMinPlay() const -> bool override { return false; }

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
    /** Model 2's shortest allowed WAIT (cs; 0 = any): InformativeWaitEngine::Settings::minWaitCs. */
    auto setMinWaitCs(int32_t cs) -> void { minWaitCs.store(cs); }
    /** A short WAIT's orchestra note: at its predicted onset (false) or at a redrawn onset (true). */
    auto setShortWaitRedraw(bool on) -> void { shortWaitRedraw.store(on); }
    [[nodiscard]] auto usesSilentModel() const -> bool { return silentActive.load(); }
    [[nodiscard]] auto reason() const -> Reason { return reasonValue.load(); }
    /** ms the pianist has been playing (-1 = not playing): the Active Model widget's progress to model 2. */
    [[nodiscard]] auto playingForMs() const -> int64_t { return playingForValue.load(); }

    /** Clock / wall clock (tests replace them). */
    std::function<int32_t()> nowCsProvider;
    std::function<int64_t()> wallMsProvider;
    /** One loop iteration without waiting (tests). True when notes were produced. */
    auto stepOnce() -> bool;
    std::string silentName = "model 1";
    /** Launchpad TD value (TemporalDensity::kAuto or a level); tests replace it. */
    std::function<int32_t()> tdValueProvider;
    /** Dynamic LogitTD, per model ("m1" silent side, "m2" playing side): what the TD widget shows. */
    struct TdControlView {
        std::string model;
        TdController::State state;
    };
    [[nodiscard]] auto getTdControlStates() const -> std::vector<TdControlView> {
        const std::lock_guard<std::mutex> lock(tdViewMutex);
        return tdViews;
    }
    /** The playing side's WAITs (WAIT Visualizer); nothing pending while model 1 draws. */
    [[nodiscard]] auto waitState() const -> WaitTracker::Snapshot { return waitTracker.snapshot(); }
    /** The gamma a model draws with now (empty at TD AUTO). */
    [[nodiscard]] auto tdGamma(bool silentModel) const -> std::optional<float>;
    std::string waitName = "informative WAIT";

private:
    auto onGenerationPauseChanged(bool paused) -> void override {
        if (! paused) {
            pedalReleased.store(true);
            tdResetRequested.store(true);
        }
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
    /**
     * After a cut: the generation clock (currentTime, and with it the polyphony lower bound) goes to the
     * newest generated note left, or `now`. Left at the furthest onset ever drawn, model 1 was forced
     * tens of seconds ahead and model 2 was switched out at once (Nov 24 2026).
     */
    auto resetGenerationClock(int32_t now) -> void;
    /** A mode cut: the generated notes from `cut` on leave both contexts (the rest stays). */
    auto rewindFrom(int32_t cut) -> void;
    /** Paused or stopped: both engines take in the input (nothing is drawn). */
    auto absorbWhileStopped(int32_t now) -> void;
    auto tickTdControl() -> void;
    [[nodiscard]] auto wallMs() const -> int64_t;

    std::unique_ptr<StreamingKv::Backend> silentBackend;
    std::unique_ptr<StreamingKv::Backend> waitBackend;
    InformativeWaitEngine silentEngine;
    InformativeWaitEngine waitEngine;

    std::atomic<bool> silentActive{true};
    std::atomic<Reason> reasonValue{Reason::Start};
    std::atomic<bool> pedalReleased{false};
    std::atomic<int> waitMode{0};
    std::atomic<bool> precompute{true};
    std::atomic<int32_t> minWaitCs{0};
    std::atomic<bool> shortWaitRedraw{false};
    int64_t playingSinceMs = -1;
    std::atomic<int64_t> playingForValue{-1};
    uint64_t silentSeed = 0; // random per run (threadRun); fixed in tests
    uint64_t waitSeed = 1;
    int32_t waitSinceCs = INT32_MIN;
    bool streamsStarted = false;

    WaitTracker waitTracker;
    TdController tdControlSilent, tdControlWait;
    std::atomic<bool> tdResetRequested{false};
    int64_t lastTdTickMs = -1;
    mutable std::mutex tdViewMutex;
    std::vector<TdControlView> tdViews;
};
