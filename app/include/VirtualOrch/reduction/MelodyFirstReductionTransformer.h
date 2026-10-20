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
#include "VirtualOrch/reduction/MelodyBassState.h"
#include "VirtualOrch/reduction/MelodyStream.h"
#include "VirtualOrch/reduction/ReductionTransformer.h"
#include "VirtualOrch/reduction/StreamingKvCache.h"

/**
 * Melody first, then the reduction.
 *
 *   1. The user's notes go on instrument 0: L = all of them (q5v8), R = their running melody +
 *      bass (algorithm A on the last 2 s, see MelodyBassState).
 *   2. The mb model (T8, learned positions) generates melody + bass events (instrument 1) with R
 *      as anticipated controls and visible 1 s rest quads (MelodyStream). Three notes per loop,
 *      unless the melody is already more than 3 s ahead of the latest reduction note.
 *   3. The double-anticipation model (T7/T6, RoPE + StreamingLLM) samples ONE reduction note
 *      (instrument 1) with the piano L as controls X (instrument 0, delta 0) and the generated
 *      melody as controls M (instrument 2, delta 3 s), hidden 1 s rest steps and Launchpad TD
 *      tokens. Only these reduction notes go downstream.
 *
 * Both models use the committed + scratch KV cache. T8 has learned positions: its stream is cut
 * to 128 tokens whenever it passes 256 and the cache is rebuilt at positions 0..127.
 * The polyphony cap of accompaniment mode applies to the melody notes and to the reduction notes.
 */
class MelodyFirstReductionTransformer : public ReductionTransformer {
public:
    static constexpr int kMelodyNotesPerLoop = 3;
    /** The melody may be at most this far ahead of the latest reduction note. */
    static constexpr int32_t kMaxMelodyAheadCs = 300;
    static constexpr int32_t kMelodyDeltaCs = 300;

    explicit MelodyFirstReductionTransformer(ModelConfig &modelConfig);

    void run() override {
        threadInit();
        threadRun();
        threadStop();
    }

    /** Load the double-anticipation KV export and the mb KV export. */
    auto init(const char *reductionModelPath, const char *melodyModelPath) -> bool;

    /** Tests: other backends (vocab sizes = logits widths). */
    auto initWithBackends(std::unique_ptr<StreamingKv::Backend> melodyBackendIn, size_t melodyVocab,
                          std::unique_ptr<StreamingKv::Backend> reductionBackendIn,
                          size_t reductionVocab) -> void;

    [[nodiscard]] auto isModelLoaded() const -> bool override {
        return melodyBackend != nullptr && reductionBackend != nullptr;
    }
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

    [[nodiscard]] auto isPitchBiasEnabled() const -> bool { return pitchBiasEnabled.load(); }
    auto setPitchBiasEnabled(bool on) -> void { pitchBiasEnabled.store(on); }

    std::function<int32_t()> nowCsProvider;
    std::function<int32_t()> tdValueProvider;

    /** One loop iteration without waiting. Returns true when it generated anything. */
    auto stepOnce() -> bool;

    // ---- for tests ----
    [[nodiscard]] auto stateForTest() const -> const MelodyBassState & { return state; }
    [[nodiscard]] auto melodyStreamForTest() const -> const MelodyStream & { return melodyStream; }
    [[nodiscard]] auto reductionStreamForTest() const -> const AnticipatedStream & {
        return reductionStream;
    }
    [[nodiscard]] auto melodyEventsForTest() const -> const std::vector<Token> & {
        return melodyEvents;
    }
    [[nodiscard]] auto melodyCacheForTest() const -> const StreamingKv::CommittedScratchCache * {
        return melodyCache.get();
    }
    [[nodiscard]] auto reductionCacheForTest() const -> const StreamingKv::CommittedScratchCache * {
        return reductionCache.get();
    }
    [[nodiscard]] auto melodyIsTooFarAhead() const -> bool;

private:
    auto nowCs() const -> int32_t;
    auto resetAll(int32_t now) -> void;
    /**
     * Leaving Accompaniment: drop the generated reduction notes AND the planned melody from
     * `cutCs` on (both streams, their M controls, windows, UI); your notes are kept.
     */
    auto rewindGeneratedFrom(int32_t cutCs) -> void;
    auto drainInput(int32_t now) -> bool;
    auto generateMelodyNote(int32_t now) -> std::optional<MelodyStream::Note>;
    auto generateReductionNote(int32_t now) -> std::optional<Token>;
    auto sampleMelodyField(int field, const std::vector<float> &logits, int32_t lowerModelOnset)
        -> int32_t;
    auto sampleReductionField(int field, const std::vector<float> &logits, int32_t lowerModelOnset,
                              const DensePitchBias::PitchWindowStats *pitchWindow, float tau,
                              float tauPrime) -> int32_t;

    std::unique_ptr<StreamingKv::Backend> melodyBackend;
    std::unique_ptr<StreamingKv::Backend> reductionBackend;
    std::unique_ptr<StreamingKv::CommittedScratchCache> melodyCache;
    std::unique_ptr<StreamingKv::CommittedScratchCache> reductionCache;
    size_t melodyVocab = 0;
    size_t reductionVocab = 0;

    MelodyBassState state;
    MelodyStream melodyStream;
    AnticipatedStream reductionStream;
    TdTokens::DenseTdResolver tdResolver;

    bool aDirty = false;
    bool userHasPlayed = false;
    int32_t latestMelodyOnset = 0;
    int32_t latestReductionOnset = 0;

    /** Generated melody events (absolute onsets) for the polyphony cap. */
    std::vector<Token> melodyEvents;
    /** Generated reduction notes as dense instrument-1 quads for the pitch window. */
    std::vector<int32_t> generatedDense;
    DensePitchBias::PitchTauScheduler pitchTauScheduler;

    std::atomic<float> onsetTemperature{DenseSampling::OnsetTemperature};
    std::atomic<float> durationTemperature{DenseSampling::DurationTemperature};
    std::atomic<float> noteTemperature{DenseSampling::NoteTemperature};
    std::atomic<float> velocityTemperature{DenseSampling::VelocityTemperature};
    std::atomic<bool> pitchBiasEnabled{true};
};
