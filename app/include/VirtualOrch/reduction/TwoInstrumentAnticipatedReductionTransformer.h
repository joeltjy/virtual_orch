#pragma once

#include <JuceHeader.h>

#include <atomic>
#include <functional>
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
 */
class TwoInstrumentAnticipatedReductionTransformer : public ReductionTransformer {
public:
    explicit TwoInstrumentAnticipatedReductionTransformer(ModelConfig &modelConfig);

    void run() override {
        threadInit();
        threadRun();
        threadStop();
    }

    /** Load the KV export. Returns false (and reports the error) if it cannot be used. */
    auto init(const char *modelPath) -> bool;

    /** Tests: use another backend (vocabSize = its logits width). */
    auto initWithBackend(std::unique_ptr<StreamingKv::Backend> newBackend, size_t vocabSize) -> void;

    [[nodiscard]] auto isModelLoaded() const -> bool override { return stage.isLoaded(); }
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

    /** Pitch anti-repetition bias (tau 0.5 as in the training sampler). */
    [[nodiscard]] auto isPitchBiasEnabled() const -> bool { return pitchBiasEnabled.load(); }
    auto setPitchBiasEnabled(bool on) -> void { pitchBiasEnabled.store(on); }

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

    [[nodiscard]] auto streamForTest() const -> const AnticipatedStream & { return stage.stream; }
    [[nodiscard]] auto cacheForTest() const -> const StreamingKv::CommittedScratchCache * {
        return stage.cache.get();
    }

private:
    auto nowCs() const -> int32_t;
    /** Apply queued user notes and note-offs. Returns true when a new note arrived. */
    auto drainInput(int32_t now) -> bool;
    auto generateOne(int32_t now) -> std::optional<Token>;
    auto resetStream(int32_t now) -> void;

    /** Stream, KV cache, sampler and the draw loop of the T3'' model (dense, given on instr. 0). */
    AnticipationStage stage{AnticipatedStream::Vocab::Dense, /*visibleRests*/ false};

    std::atomic<float> onsetTemperature{DenseSampling::OnsetTemperature};
    std::atomic<float> durationTemperature{DenseSampling::DurationTemperature};
    std::atomic<float> noteTemperature{DenseSampling::NoteTemperature};
    std::atomic<float> velocityTemperature{DenseSampling::OnsetTemperature}; // 0.5: same as the other fields
    std::atomic<bool> pitchBiasEnabled{true};
};
