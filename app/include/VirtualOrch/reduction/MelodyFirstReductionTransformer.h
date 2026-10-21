#pragma once

#include <JuceHeader.h>

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/TdTokens.h"
#include "VirtualOrch/reduction/AnticipationStage.h"
#include "VirtualOrch/reduction/DenseTypes.h"
#include "VirtualOrch/reduction/MelodyBassState.h"
#include "VirtualOrch/reduction/ReductionTransformer.h"
#include "VirtualOrch/reduction/StreamingKvCache.h"

/**
 * Melody first, then the reduction (the T8'' -> T7'' pipeline).
 *
 *   1. The user's notes go on instrument 0: L = all of them (q5v8), R = their running melody +
 *      bass (algorithm A on the last 2 s, see MelodyBassState).
 *   2. T8'' (dense vocab 11776, RoPE) takes R as GIVEN events on instrument 0 and generates a two-
 *      voice reduction melody + bass on instrument 1 (onset floor: at most 2 notes sounding).
 *      Three notes per loop, unless the melody is already more than 3 s ahead of the latest
 *      reduction note.
 *   3. T7'' (AMT-native vocab 55293, RoPE, visible 1 s REST quads, ANTICIPATE first) takes the
 *      piano L as given events on instrument 0 and T8'''s notes as melody+bass CONTROLS anticipated
 *      3 s, with TD tokens from the Launchpad, and samples ONE reduction note (instrument 1, the
 *      reduction minus the melody and bass). Only these notes go downstream.
 *
 * Both models use the committed + scratch KV cache and the 30 s eviction rebase (see
 * AnticipationStage). The polyphony cap of accompaniment mode applies to both stages (the melody
 * stage is held to min(cap, 2)).
 */
class MelodyFirstReductionTransformer : public ReductionTransformer {
public:
    static constexpr int kMelodyNotesPerLoop = 3;
    /** The melody may be at most this far ahead of the latest reduction note. */
    static constexpr int32_t kMaxMelodyAheadCs = 300;
    static constexpr int32_t kMelodyDeltaCs = 300;
    /** T8'' targets are two voices: never more than this many sounding. */
    static constexpr int kMelodyVoices = 2;

    explicit MelodyFirstReductionTransformer(ModelConfig &modelConfig);

    void run() override {
        threadInit();
        threadRun();
        threadStop();
    }

    /** Load the T7'' (reduction) and T8'' (melody) KV exports. */
    auto init(const char *reductionModelPath, const char *melodyModelPath) -> bool;

    /** Tests: other backends (vocab sizes = logits widths). */
    auto initWithBackends(std::unique_ptr<StreamingKv::Backend> melodyBackendIn, size_t melodyVocab,
                          std::unique_ptr<StreamingKv::Backend> reductionBackendIn,
                          size_t reductionVocab) -> void;

    [[nodiscard]] auto isModelLoaded() const -> bool override {
        return melodyStage.isLoaded() && reductionStage.isLoaded();
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
    [[nodiscard]] auto melodyStreamForTest() const -> const AnticipatedStream & {
        return melodyStage.stream;
    }
    [[nodiscard]] auto reductionStreamForTest() const -> const AnticipatedStream & {
        return reductionStage.stream;
    }
    [[nodiscard]] auto melodyEventsForTest() const -> const std::vector<Token> & {
        return melodyEvents;
    }
    [[nodiscard]] auto melodyCacheForTest() const -> const StreamingKv::CommittedScratchCache * {
        return melodyStage.cache.get();
    }
    [[nodiscard]] auto reductionCacheForTest() const -> const StreamingKv::CommittedScratchCache * {
        return reductionStage.cache.get();
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
    /** R changed (or the stream restarted): feed new pieces to T8'' and confirm finished ones. */
    auto syncMelodyGiven() -> void;
    auto generateMelodyNote(int32_t now) -> std::optional<AnticipatedStream::Note>;
    auto generateReductionNote(int32_t now) -> std::optional<Token>;
    [[nodiscard]] auto temperatures() const -> std::array<float, 4>;

    AnticipationStage melodyStage{AnticipatedStream::Vocab::Dense, /*visibleRests*/ false};
    AnticipationStage reductionStage{AnticipatedStream::Vocab::Amt, /*visibleRests*/ true};

    MelodyBassState state;
    using PieceId = std::tuple<int64_t, int32_t, int, int>; // key, onset, pitch, line
    std::set<PieceId> fedPieces;
    std::set<PieceId> heldPieces;

    bool aDirty = false;
    bool userHasPlayed = false;
    int32_t latestMelodyOnset = 0;
    int32_t latestReductionOnset = 0;

    /** Generated melody events (absolute onsets) for the polyphony cap. */
    std::vector<Token> melodyEvents;

    std::atomic<float> onsetTemperature{DenseSampling::OnsetTemperature};
    std::atomic<float> durationTemperature{DenseSampling::DurationTemperature};
    std::atomic<float> noteTemperature{DenseSampling::NoteTemperature};
    std::atomic<float> velocityTemperature{DenseSampling::OnsetTemperature}; // 0.5 like the others
    std::atomic<bool> pitchBiasEnabled{true};
};
