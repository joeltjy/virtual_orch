#pragma once

#include <JuceHeader.h>

#include <atomic>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <memory>
#include <optional>
#include <vector>

#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/TdTokens.h"
#include "VirtualOrch/reduction/AnticipatedStream.h"
#include "VirtualOrch/reduction/AnticipationSampler.h"
#include "VirtualOrch/reduction/AnticipationStage.h"
#include "VirtualOrch/reduction/DensePitchBias.h"
#include "VirtualOrch/reduction/DenseTypes.h"
#include "VirtualOrch/reduction/ReductionTransformer.h"
#include "VirtualOrch/reduction/StreamingKvCache.h"
#include "VirtualOrch/reduction/TdController.h"

/**
 * Piano -> piano reduction with the T3'' model (maestro replay, dense vocab 11776, RoPE with a
 * StreamingLLM KV cache: 4 sink + 508 recent rows).
 *
 *   - The user's piano notes are GIVEN events on instrument 0, in onset order with the generated
 *     notes (instrument 1). A given note is never sampled; it is inserted once the event clock
 *     (onset of the last generated note) has reached its onset, or earlier when a sampled onset
 *     lies at or after it (it goes in first and the onset is drawn again).
 *   - Tie order at a shared onset: the piano (given) note comes FIRST. This was chosen over the
 *     T3'' training order (first generated note first), which setTieFirstGenerated(true) restores.
 *   - Temporal-density (TD) tokens come from the Launchpad (AUTO repeats the last level, the
 *     dense vocab has no "unknown").
 *   - A held key is used for inference with the duration it has so far (frozen), but is not
 *     written into the committed cache until its note-off or the hold limit (see
 *     AnticipatedStream): the cache is split into committed + scratch (StreamingKvCache.h).
 *   - The shared AnticipationSampler draws every field (onset-gap, duration and pitch biases, masks,
 *     top-p 0.98, temperature 0.5), q5v8 with the app's 10 cs duration floor and A0..C8 pitches.
 *   - The polyphony cap of accompaniment mode raises the earliest allowed onset
 *     (ReductionTransformer::onsetLowerBoundCs).
 *   - Accompaniment: late input is held back and written in onset order at most once a second,
 *     with the generated notes kept (AnticipationStage, Params::reorderLate).
 *   - Every logit adjustment of the older models can be switched on (pitch / duration / onset-gap
 *     biases with their tau, the short-gap penalty, LogitTD); all start OFF.
 *
 * Variant::Wait = W2/W3 (simultaneous_model_v1 / v2, vocab 11777, WAIT = 11776): WAIT before every
 * piano onset group, given notes merged within 5 cs, ascending pitch at a shared onset, the WAIT
 * threshold rule (P(play) >= 0.30 plays) and the live 1 s wait; onset temperature 1.0, the other
 * fields 0.5 (simultaneous_model_sampling.txt).
 *
 * Variant::Hybrid = HybridTransformer: two models with one shared context, v1 (continuation_model, the
 * T3'' rules, `silentStage`) and W2/W3 (the WAIT rules, `stage`). Which one draws (updateHybrid):
 *   - a pedal release (or RT un-pause) switches to v1 at once;
 *   - pedal up and the user playing for kHybridPlayToWaitMs: switch to W2/W3 ("playing" = a key
 *     down or the last release less than kHybridStopMs ago);
 *   - on W2/W3 and the user silent for more than kHybridStopMs: every note generated from now on
 *     is deleted (playback and both contexts) and v1 takes over at once;
 *   - on W2/W3, not paused, nothing generated sounding now and the next generated onset at least
 *     kHybridPredictedSilenceCs ahead (W2/W3 predicts 2 s of silence): the same cut, v1 takes
 *     over, and the 1 s playing timer starts again. Nothing queued at all counts too once the
 *     last generated note (or the switch to W2/W3) is kHybridPredictedSilenceCs old: W2/W3
 *     waiting on every group while the user keeps playing never stays silent longer than that.
 * Every given note goes to both streams, and each generated note is also written into the other
 * model's stream (mirrorEvent).
 * The settings below apply to the W2/W3 side. The v1 side has its own (hybridV1*): temperature
 * 0.5 and every logit adjustment ON by default (pitch / duration / onset-gap biases at their base
 * tau, under the global BiasSettings switch and silence gate; the short-gap penalty; LogitTD).
 */
class TwoInstrumentAnticipatedReductionTransformer : public ReductionTransformer {
public:
    enum class Variant { T3, Wait, Hybrid };
    static constexpr int64_t kHybridPlayToWaitMs = 1000;
    static constexpr int64_t kHybridStopMs = 500;
    static constexpr int32_t kHybridPredictedSilenceCs = 200;
    /** Why the hybrid uses its current model (the Active Model widget). */
    enum class HybridReason { Start, PedalReleased, Playing, Stopped, PredictedSilence };
    static constexpr int32_t kWaitToken = 11776;
    static constexpr float kDefaultWaitMinPlay = 0.30f;

    explicit TwoInstrumentAnticipatedReductionTransformer(ModelConfig &modelConfig,
                                                          Variant variant = Variant::T3);
    [[nodiscard]] auto getVariant() const -> Variant { return variant; }

    void run() override {
        threadInit();
        threadRun();
        threadStop();
    }

    /** Load the KV export. Returns false (and reports the error) if it cannot be used. */
    auto init(const char *modelPath) -> bool;
    /** Why the last init failed (empty after a successful one). */
    std::string lastInitError;

    /** Tests: use another backend (vocabSize = its logits width). */
    auto initWithBackend(std::unique_ptr<StreamingKv::Backend> newBackend, size_t vocabSize) -> void;

    [[nodiscard]] auto isModelLoaded() const -> bool override {
        return stage.isLoaded() && (variant != Variant::Hybrid || silentStage.isLoaded());
    }

    /** Hybrid: load v1 (T3'', vocab 11776) and W2/W3 (vocab 11777). False + lastInitError if not. */
    auto initHybrid(const char *silentModelPath, const char *waitModelPath) -> bool;
    /** Tests: hybrid with other backends. */
    auto initHybridWithBackends(std::unique_ptr<StreamingKv::Backend> silentBackend, size_t silentVocab,
                                std::unique_ptr<StreamingKv::Backend> waitBackend, size_t waitVocab)
        -> void;
    /** Hybrid: true when the last note came from v1. */
    [[nodiscard]] auto lastNoteFromSilentModel() const -> bool { return lastFromSilent; }
    /** Hybrid: v1 (true) or W2/W3 (false) draws now, and why (any thread). */
    [[nodiscard]] auto hybridUsesV1() const -> bool { return hybridV1.load(); }
    [[nodiscard]] auto hybridReason() const -> HybridReason { return hybridReasonValue.load(); }
    /** Hybrid: ms the user has been playing (-1 = not playing), for the widget. */
    [[nodiscard]] auto hybridPlayingForMs() const -> int64_t { return hybridPlayingFor.load(); }
    /** Dynamic LogitTD (TdController), per model: what the TD widget shows. */
    struct TdControlView {
        std::string model; // "W", "T3" or "v1"
        TdController::State state;
    };
    /** The controllers' last states (any thread). */
    [[nodiscard]] auto getTdControlStates() const -> std::vector<TdControlView> {
        const std::lock_guard<std::mutex> lock(tdViewMutex);
        return tdViews;
    }
    /** Hybrid, v1 side: its own logit adjustments, all ON by default. */
    std::atomic<bool> hybridV1PitchBias{true};
    std::atomic<bool> hybridV1DurationBias{true};
    std::atomic<bool> hybridV1OnsetGapBias{true};
    std::atomic<bool> hybridV1ShortGapPenalty{true};
    std::atomic<bool> hybridV1LogitTd{true};
    /** Hybrid: the two model files (stems), set by initHybrid. */
    std::string hybridV1Name = "v1";
    std::string hybridWaitName = "W2/W3";
    /** Wall clock in ms for the hybrid rules (tests replace it). */
    std::function<int64_t()> wallMsProvider;
    [[nodiscard]] auto inputEventWidth() const -> size_t override { return 4; }

    void threadInit() {}
    void threadRun();
    void threadStop() {}

    struct FieldSampling {
        float topP = 1.0f;
        float temperature = 1.0f;
    };
    [[nodiscard]] auto samplingForStep(int stepIdx) const -> FieldSampling;

    [[nodiscard]] auto getOnsetTemperature() const -> float { return onsetTemperature.load(); }
    [[nodiscard]] auto getDurationTemperature() const -> float { return durationTemperature.load(); }
    [[nodiscard]] auto getNoteTemperature() const -> float { return noteTemperature.load(); }
    [[nodiscard]] auto getVelocityTemperature() const -> float { return velocityTemperature.load(); }
    auto setOnsetTemperature(float t) -> void { onsetTemperature.store(t); }
    auto setDurationTemperature(float t) -> void { durationTemperature.store(t); }
    auto setNoteTemperature(float t) -> void { noteTemperature.store(t); }
    auto setVelocityTemperature(float t) -> void { velocityTemperature.store(t); }

    // Logit adjustments (all OFF by default; the BiasSettings gate applies to the three biases).
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

    /**
     * T3'' training order at a shared onset: the first generated note before the given notes.
     * Off by default: the piano note comes first.
     */
    auto setTieFirstGenerated(bool on) -> void { stage.tieFirstGenerated = on; }
    [[nodiscard]] auto getTieFirstGenerated() const -> bool { return stage.tieFirstGenerated; }

    /** App clock in cs (defaults to the Clock set by AppSession); tests replace it. */
    std::function<int32_t()> nowCsProvider;
    /** Launchpad TD value (TemporalDensity::kAuto or a level); tests replace it. */
    std::function<int32_t()> tdValueProvider;

    /** One loop iteration without waiting: drain input, then generate one note if allowed. */
    auto stepOnce() -> bool;

    /** What one pass of the live loop did (the thread waits 5 ms after Idle / NothingGenerated). */
    enum class LiveStep { Idle, StoppedWithInput, NothingGenerated, Generated };
    /**
     * Exactly one pass of the thread's loop (threadRun calls it): throttle release, mode cut,
     * clock jump, input drain + Jam cut, hybrid rules, then one note unless stopped. The offline
     * simulator calls beginLive() once and then liveStep() itself, with no thread.
     */
    auto liveStep() -> LiveStep;
    /** What threadRun does before its loop (clear queues and history, reset the stream). */
    auto beginLive() -> void;

    /** Replay (LogReplay): the samplers that draw notes (W2/W3 or T3'' first, then v1 for Hybrid). */
    auto samplersForReplay() -> std::vector<AnticipationSampler *> {
        std::vector<AnticipationSampler *> out{&stage.sampler};
        if (variant == Variant::Hybrid)
            out.push_back(&silentStage.sampler);
        return out;
    }
    /** The last pass stopped before drawing because a replay sampler held the draw. */
    [[nodiscard]] auto drawWasHeld() const -> bool {
        return stage.lastHeld() || (variant == Variant::Hybrid && silentStage.lastHeld());
    }
    /** The drawing model is waiting for the soloist (a W2/W3 WAIT, not timed out yet). */
    [[nodiscard]] auto isWaitingForSoloist() const -> bool {
        return (useSilentStage() ? silentStage : stage).lastWaited();
    }

    auto setWaitMaxAheadForTest(int32_t cs) -> void { stage.maxAheadCs = cs; }

    /** Operation log: "W" or "v1" for a note this transformer drew. */
    [[nodiscard]] auto modelOfNote(int32_t noteId) -> std::string override;
    [[nodiscard]] auto streamForTest() const -> const AnticipatedStream & { return stage.stream; }
    [[nodiscard]] auto silentStreamForTest() const -> const AnticipatedStream & { return silentStage.stream; }
    [[nodiscard]] auto cacheForTest() const -> const StreamingKv::CommittedScratchCache * {
        return stage.cache.get();
    }

private:
    auto nowCs() const -> int32_t;
    /** Apply queued user notes and note-offs. Returns true when a new note arrived. */
    auto drainInput(int32_t now) -> bool;
    auto generateOne(int32_t now) -> std::optional<Token>;
    auto resetStream(int32_t now) -> void;
    /** Leaving Accompaniment: drop the generated notes from `cutCs` on (stream, windows, UI). */
    auto rewindGeneratedFrom(int32_t cutCs) -> void;
    /**
     * Jam (not Accompaniment): new input drops the notes generated from `now` on, from playback
     * (a ClearQueue at now) and from the model, which regenerates from now with the input in
     * context, as the other reductions do. Without it T3'' kept generating seconds ahead.
     */
    auto cutFutureOnInput(int32_t now) -> void;
    /**
     * The clock jumped back (rewind, restart): the queued notes go (a ClearQueue at now), the
     * context and histories are cleared and the stream starts again at the new clock.
     */
    auto restartOnClockJumpBack(int32_t now) -> void;

    /** Stream, KV cache, sampler and the draw loop of the T3'' model (dense, given on instr. 0). */
    AnticipationStage stage{AnticipatedStream::Vocab::Dense, /*visibleRests*/ false};
    /** Hybrid only: the v1 (T3'') model used while the piano is silent. */
    AnticipationStage silentStage{AnticipatedStream::Vocab::Dense, /*visibleRests*/ false};
    bool lastFromSilent = false;
    /** Both stages for the hybrid, the one stage otherwise. */
    template <typename F>
    auto forEachStage(F &&f) -> void {
        f(stage);
        if (variant == Variant::Hybrid)
            f(silentStage);
    }
    [[nodiscard]] auto useSilentStage() const -> bool {
        return variant == Variant::Hybrid && hybridV1.load();
    }
    /** Hybrid: apply the switching rules (each loop iteration, also while idle or waiting). */
    auto updateHybrid(int32_t nowCs) -> void;
    auto setHybridModel(bool v1, HybridReason why, int32_t nowCs) -> void;
    /** Drop every generated note from nowCs on (playback, history, both contexts). */
    auto cutGeneratedFrom(int32_t nowCs, const std::string &reason = {}) -> void;

    // ---- dynamic LogitTD: one gamma per model, updated every TdController::kTickMs
    TdController tdControlMain, tdControlV1;
    double lastTdTickMs = -1.0;
    /** Set on a stream restart or pedal release; tickTdControl resets both controllers. */
    std::atomic<bool> tdResetRequested{false};
    mutable std::mutex tdViewMutex;
    std::vector<TdControlView> tdViews;
    /** Tick the controllers when one is due: measure, update, publish, log TD_CTRL. */
    auto tickTdControl(int32_t now) -> void;
    /** The gamma a model draws with now (empty at TD AUTO). */
    [[nodiscard]] auto tdGamma(bool silentModel) const -> std::optional<float>;

    // ---- operation log (OpLog): the context as a diff after every pass that changed it
    struct ContextLog {
        std::vector<AnticipatedStream::ItemView> prev;
        int32_t base = INT32_MIN;
    };
    ContextLog contextLogMain, contextLogV1;
    std::string removalReason;        // why the next removals from the context happen
    std::vector<int32_t> drainedKeys; // keys taken in by the last drain
    std::map<int32_t, std::string> noteModels;
    std::mutex noteModelsMutex;
    auto logContext(const AnticipationStage &st, ContextLog &log, const char *model) -> void;
    auto logContexts() -> void;
    auto logCheckpoint(int32_t now) -> void;
    /**
     * No generated note sounds at nowCs, and the next one starts >= 2 s later; with none queued,
     * the last one ended (or W2/W3 took over) >= 2 s ago.
     */
    [[nodiscard]] auto predictsSilenceFrom(int32_t nowCs) const -> bool;
    auto onGenerationPauseChanged(bool paused) -> void override {
        if (! paused) {
            pedalReleased.store(true);
            tdResetRequested.store(true);
        }
    }
    std::atomic<bool> hybridV1{true};
    std::atomic<HybridReason> hybridReasonValue{HybridReason::Start};
    std::atomic<int64_t> hybridPlayingFor{-1};
    std::atomic<bool> pedalReleased{false};
    int64_t playingSinceMs = -1;
    int32_t hybridWaitSinceCs = INT32_MIN; // when W2/W3 last took over (clock)
    auto loadKvBackend(const char *path, std::unique_ptr<StreamingKv::OnnxBackend> &out) -> bool;
    /**
     * Session log "RT_IN": the exact token ids the model's context holds, as they change: after
     * each generated note, the tokens from the first one that differs from the last logged copy
     * (so a rewind / reorder / rebase shows as a rewrite from that index).
     */
    auto logModelInput(const AnticipationStage &st, std::vector<int32_t> &lastLogged, const char *who) -> void;
    std::vector<int32_t> loggedMain;
    std::vector<int32_t> loggedV1;

    std::atomic<float> onsetTemperature{DenseSampling::OnsetTemperature};
    std::atomic<float> durationTemperature{DenseSampling::DurationTemperature};
    std::atomic<float> noteTemperature{DenseSampling::NoteTemperature};
    std::atomic<float> velocityTemperature{DenseSampling::OnsetTemperature}; // 0.5: same as the other fields
    Variant variant;
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
