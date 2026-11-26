#pragma once

#include <JuceHeader.h>

#include <atomic>
#include <functional>
#include <optional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/reduction/GroupTokenEngine.h"
#include "VirtualOrch/reduction/ReductionTransformer.h"
#include "VirtualOrch/reduction/RtLogitAdjustments.h"
#include "VirtualOrch/reduction/StreamingKvCache.h"
#include "VirtualOrch/reduction/TdController.h"
#include "VirtualOrch/reduction/WaitTracker.h"

/**
 * HYBRID v1.4 (2026-11-26): HybridChordTokenReductionTransformer (v1.2) copied, with the v1.4 group-token models
 * (orchestration docs/model_specs/v14_group_tokens.txt): model 1 = m1_v4 (rt_continuation_chordal.onnx, vocab 12096),
 * model 2 = m2_v4 (rt_simultaneous_chordal.onnx, vocab 12239), both run by GroupTokenEngine. A group (and a WAIT
 * block) goes to the OT and playback as ONE Token (bass pitch + chordShape), decoded there with the v4 voicing bank
 * (GroupChord::expand; the bank is loadVoicingBank's, shared through GroupChord::setSharedBank). The mirrored note
 * keeps its group, so each model sees the other's groups as groups. The v1.2 comment follows.
 *
 * Hybrid v1.2, its own reduction transformer (Joel, 2026-11-25: "each one inherits from the base reduction
 * Transformer class so that they are all independent of each other. i don't want v1.2's changes to affect v1.1").
 * A copy of HybridInformativeWaitReductionTransformer / InformativeWaitEngine as of v1.2 (GroupTokenEngine), so
 * nothing here changes v1.1. The original comment follows.
 *
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
 * WAIT); the gamma goes to that model's onset draws (GroupTokenEngine::Settings::tdGamma). Both go back to
 * their warm-start gamma on a stream restart; a pedal press only drops a negative gamma. TD AUTO: no shift.
 *
 * Hybrid v1.2 (2026-11-24): the same, with the chord-token model 2 (m2c, vocab 14071; picked by the ONNX vocab): its
 * engine runs GroupTokenEngine::Settings::chordTokens (CHORD blocks decoded for the OT with the preset's voicing
 * bank, [LATE d], [DUR?]). Model 1 needs nothing new: it never reads model 2's tokens; each model-2 note it mirrors
 * is a decoded chord note (or the leftover), so model 1 sees the chord as plain quads, and its own piano durations
 * stay its own estimate / release rewrite.
 *
 * Logit adjustments as the legacy Hybrid (RtLogitAdjustments): model 1 takes the silent side's (hybridV1*, all ON,
 * base taus), model 2 the main ones (all OFF, the widget's taus); the BiasSettings switch and silence gate apply to
 * the three biases; both draw within the Launchpad register range.
 */
class HybridGroupReductionTransformer : public ReductionTransformer, public RtLogitAdjustments {
public:
    static constexpr int64_t kPlayToWaitMs = 1000;
    static constexpr int64_t kStopMs = 500;
    static constexpr int32_t kPredictedSilenceCs = 200;
    /** Added to model 2's output velocities (capped at 127): heard over the pianist. */
    static constexpr int32_t kSimultaneousVelocityBoost = 16;

    enum class Reason { Start, PedalReleased, Playing, Stopped, PredictedSilence, Accompaniment };

    explicit HybridGroupReductionTransformer(ModelConfig &modelConfig);

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

    /** Load model 1 (silent side, vocab 11768) and the chord-token model 2 (playing side, vocab 14071). */
    auto init(const char *silentModelPath, const char *waitModelPath) -> bool;
    /** v1.4: the voicing bank v4 (orchestration/overnight/build_voicing_bank_v4.py, voicing_bank_v4.json); false if unreadable. */
    auto loadVoicingBank(const std::string &path) -> bool;
    /** Tests: the bank directly, and the chord-token mode without an ONNX vocab check. */
    auto setVoicingBank(std::shared_ptr<const GroupChord::VoicingBankV4> bank) -> void { voicings = std::move(bank); GroupChord::setSharedBank(voicings); }
    auto setChordModel(bool on) -> void { chordModel.store(on); }
    [[nodiscard]] auto isChordModel() const -> bool { return chordModel.load(); }
    [[nodiscard]] auto label() const -> const char * { return "Hybrid v1.4"; }
    /** Tests: other backends (owned). */
    auto initWithBackends(std::unique_ptr<StreamingKv::Backend> silent, std::unique_ptr<StreamingKv::Backend> wait)
        -> void;
    std::string lastInitError;

    [[nodiscard]] auto isModelLoaded() const -> bool override { return silentBackend != nullptr && waitBackend != nullptr; }
    [[nodiscard]] auto inputEventWidth() const -> size_t override { return 4; }

    /** Model 2's mode (unconstrained by default). */
    auto setWaitMode(GroupTokenEngine::Mode mode) -> void { waitMode.store(static_cast<int>(mode)); }
    auto setPrecompute(bool on) -> void { precompute.store(on); }
    /** Model 2's shortest allowed WAIT (cs; 0 = any): GroupTokenEngine::Settings::minWaitCs. */
    auto setMinWaitCs(int32_t cs) -> void { minWaitCs.store(cs); }
    /** Model 2's start delay (ms; 0 = none): GroupTokenEngine::Settings::startDelayCs. */
    auto setStartDelayMs(int32_t ms) -> void { startDelayMs.store(ms); }
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
    /**
     * The orchestra's TD measured AFTER the OT (OrchestrationTransformer::orchestraTd: model notes as they will sound,
     * tremolo at 8 events a second), read at every TD tick; empty (or unset) = the RT's own count of its generated
     * onsets, as before. Set by AppSession (Nov 25 2026).
     */
    std::function<std::optional<double>()> measuredTdProvider;
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
    /** The operation log: an RT_DRAW per generated note, an RT_WAIT per WAIT event of model 2. */
    auto logDraw(const GroupTokenEngine &engine, const char *model, const GroupTokenEngine::OutNote &n,
                 int32_t noteId, int32_t now) -> void;
    auto logWaitEvent(const GroupTokenEngine::Event &e) -> void;

    auto onGenerationPauseChanged(bool paused) -> void override {
        if (! paused) {
            pedalReleased.store(true);
            // only drop a negative gamma: a full reset here zeroed it on every pedal press (Nov 25 2026)
            tdSoftResetRequested.store(true);
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
    GroupTokenEngine silentEngine;
    GroupTokenEngine waitEngine;

    std::atomic<bool> silentActive{true};
    std::atomic<Reason> reasonValue{Reason::Start};
    std::atomic<bool> pedalReleased{false};
    std::atomic<int> waitMode{0};
    std::atomic<bool> precompute{true};
    std::atomic<int32_t> minWaitCs{0};
    std::atomic<int32_t> startDelayMs{0};
    std::atomic<bool> shortWaitRedraw{false};
    std::atomic<bool> chordModel{true}; // v1.2: always the chord-token model 2 (tests may switch it off)
    std::shared_ptr<const GroupChord::VoicingBankV4> voicings;
    int64_t playingSinceMs = -1;
    std::atomic<int64_t> playingForValue{-1};
    uint64_t silentSeed = 0; // random per run (threadRun); fixed in tests
    uint64_t waitSeed = 1;
    int32_t waitSinceCs = INT32_MIN;
    bool streamsStarted = false;

    WaitTracker waitTracker;
    TdController tdControlSilent, tdControlWait;
    std::atomic<bool> tdResetRequested{false};
    std::atomic<bool> tdSoftResetRequested{false};
    int64_t lastTdTickMs = -1;
    mutable std::mutex tdViewMutex;
    std::vector<TdControlView> tdViews;
};
