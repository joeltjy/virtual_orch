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
#include "VirtualOrch/reduction/DensePitchBias.h"
#include "VirtualOrch/reduction/DenseTypes.h"
#include "VirtualOrch/reduction/ReductionTransformer.h"
#include "VirtualOrch/reduction/StreamingKvCache.h"

/**
 * Piano -> piano reduction with an anticipation-style model (maestro_rope_td_T3 export):
 * RoPE with a StreamingLLM KV cache (4 sink + 508 recent rows).
 *
 *   - The user's piano notes are anticipated CONTROLS on instrument 3: never sampled, written
 *     into the stream once the event clock has passed their onset (see AnticipatedStream).
 *   - The model samples instrument-1 reduction notes, one [time, duration, note, velocity]
 *     quad at a time, with the clock = onset of the last generated note.
 *   - Temporal-density (TD) tokens come from the Launchpad (AUTO repeats the last level, the
 *     dense vocab has no "unknown").
 *   - A held key is used for inference with the duration it has so far (frozen), but is not
 *     written into the committed cache until its note-off (or 1 s... see AnticipatedStream::
 *     kHoldCommitCs): the cache is split into committed + scratch (StreamingKvCache.h).
 *   - Everything is q5v8 with the app's 10 cs duration floor and A0..C8 pitch clamp.
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

    [[nodiscard]] auto isModelLoaded() const -> bool override { return backend != nullptr; }
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

    /** App clock in cs (defaults to the Clock set by AppSession); tests replace it. */
    std::function<int32_t()> nowCsProvider;
    /** Launchpad TD value (TemporalDensity::kAuto or a level); tests replace it. */
    std::function<int32_t()> tdValueProvider;

    /** One loop iteration without waiting: drain input, then generate one note if allowed. */
    auto stepOnce() -> bool;

    [[nodiscard]] auto streamForTest() const -> const AnticipatedStream & { return stream; }
    [[nodiscard]] auto cacheForTest() const -> const StreamingKv::CommittedScratchCache * {
        return cache.get();
    }

private:
    auto nowCs() const -> int32_t;
    /** Apply queued user notes and note-offs. Returns true when a new note arrived. */
    auto drainInput(int32_t now) -> bool;
    auto generateOne(int32_t now) -> std::optional<Token>;
    auto sampleField(int field, const std::vector<float> &logits, int32_t lowerModelOnset,
                     const DensePitchBias::PitchWindowStats *pitchWindow, float tau, float tauPrime)
        -> int32_t;
    auto resetStream(int32_t now) -> void;

    std::unique_ptr<StreamingKv::Backend> backend;
    std::unique_ptr<StreamingKv::CommittedScratchCache> cache;
    size_t vocabSize = 0;

    AnticipatedStream stream;
    TdTokens::DenseTdResolver tdResolver;

    /** Generated notes as dense instrument-1 quads (absolute onsets) for the pitch window. */
    std::vector<int32_t> generatedDense;
    DensePitchBias::PitchTauScheduler pitchTauScheduler;

    std::atomic<float> onsetTemperature{DenseSampling::OnsetTemperature};
    std::atomic<float> durationTemperature{DenseSampling::DurationTemperature};
    std::atomic<float> noteTemperature{DenseSampling::NoteTemperature};
    std::atomic<float> velocityTemperature{DenseSampling::VelocityTemperature};
    std::atomic<bool> pitchBiasEnabled{true};
};
