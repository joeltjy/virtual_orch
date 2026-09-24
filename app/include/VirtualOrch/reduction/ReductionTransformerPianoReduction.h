#pragma once

#include <JuceHeader.h>
#include <onnxruntime_cxx_api.h>

#include <atomic>
#include <memory>
#include <vector>

#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/reduction/DenseDurationBias.h"
#include "VirtualOrch/reduction/DenseFirstTokenLogitsAudit.h"
#include "VirtualOrch/reduction/DenseOnsetGapBias.h"
#include "VirtualOrch/reduction/DensePitchBias.h"
#include "VirtualOrch/reduction/DenseTypes.h"
#include "VirtualOrch/reduction/ReductionTransformer.h"

/**
 * Dense reduction model finetuned on paired piano + piano-reduction MIDI (2 real
 * instruments, amt_dense_maestro_v8_piano_reduction). Unlike V1/V2, this checkpoint
 * has two meaningfully-trained instrument bands:
 *   - instrument 0 (piano): the live keyboard input, conditioning only, never sampled.
 *   - instrument 1 (reduction): the only band the model ever generates. Downstream,
 *     OrchestrationTransformer turns this reduction into a full orchestra.
 * Same 4-token dense event ONNX contract as V1/V2 (input_ids -> logits, no past KV),
 * and the same MAESTRO v8 quantization grid as V2 (this checkpoint was finetuned from
 * the same v8 base V2 targets), but with two differences from both:
 *   1. Generation is hard-masked onto instrument 1 (see generationInstrumentLogits) —
 *      not derived from ModelConfig::outputInstruments the way V1's instrLogits is,
 *      because a misconfigured preset could otherwise resample the wrong band.
 *   2. The pre-inference context sort breaks onset ties by ascending instrument
 *      (DensePianoReductionOrder::sortStridedByOnsetThenInstrument), matching how the
 *      offline training data is ordered — see PIANO_REDUCTION_TRANSFORMER.md.
 *      V1/V2's plain onset-only sort would not match training for a two-instrument
 *      stream (their own events are always instrument 0, so it's a no-op for them).
 *
 * Online pitch/duration anti-repetition bias, ported from V2 (see PITCH_BIAS.md), is
 * gated by `biasEnabled` (default on) so it can be turned off from the UI.
 */
class ReductionTransformerPianoReduction : public ReductionTransformer {
public:
    explicit ReductionTransformerPianoReduction(ModelConfig &modelConfig);

    void run() override {
        threadInit();
        threadRun();
        threadStop();
    }

    void init(const char *modelPath);

    [[nodiscard]] auto isModelLoaded() const -> bool override { return session != nullptr; }

    [[nodiscard]] auto inputEventWidth() const -> size_t override { return 4; }

    void threadInit();

    void threadRun();

    void threadStop();

    /** Nucleus p / temperature for event field `stepIdx` (0 time, 1 dur, 2 note, 3 vel). */
    struct FieldSampling {
        float topP = 1.0f;
        float temperature = 1.0f;
    };

    [[nodiscard]] auto samplingForStep(int stepIdx) const -> FieldSampling;

    [[nodiscard]] auto getOnsetTemperature() const -> float {
        return onsetTemperature.load(std::memory_order_relaxed);
    }
    [[nodiscard]] auto getDurationTemperature() const -> float {
        return durationTemperature.load(std::memory_order_relaxed);
    }
    [[nodiscard]] auto getNoteTemperature() const -> float {
        return noteTemperature.load(std::memory_order_relaxed);
    }
    [[nodiscard]] auto getVelocityTemperature() const -> float {
        return velocityTemperature.load(std::memory_order_relaxed);
    }

    auto setOnsetTemperature(float temperature) -> void {
        onsetTemperature.store(temperature, std::memory_order_relaxed);
    }
    auto setDurationTemperature(float temperature) -> void {
        durationTemperature.store(temperature, std::memory_order_relaxed);
    }
    auto setNoteTemperature(float temperature) -> void {
        noteTemperature.store(temperature, std::memory_order_relaxed);
    }
    auto setVelocityTemperature(float temperature) -> void {
        velocityTemperature.store(temperature, std::memory_order_relaxed);
    }

    /** Online pitch/duration/onset-gap anti-repetition bias. Default on. */
    [[nodiscard]] auto isBiasEnabled() const -> bool {
        return isPitchBiasEnabled() || isDurationBiasEnabled() || isOnsetGapBiasEnabled();
    }

    auto setBiasEnabled(bool enabled) -> void {
        setPitchBiasEnabled(enabled);
        setDurationBiasEnabled(enabled);
        setOnsetGapBiasEnabled(enabled);
    }

    [[nodiscard]] auto isPitchBiasEnabled() const -> bool {
        return pitchBiasEnabled.load(std::memory_order_relaxed);
    }
    auto setPitchBiasEnabled(bool enabled) -> void {
        pitchBiasEnabled.store(enabled, std::memory_order_relaxed);
    }

    [[nodiscard]] auto isDurationBiasEnabled() const -> bool {
        return durationBiasEnabled.load(std::memory_order_relaxed);
    }
    auto setDurationBiasEnabled(bool enabled) -> void {
        durationBiasEnabled.store(enabled, std::memory_order_relaxed);
    }

    [[nodiscard]] auto isOnsetGapBiasEnabled() const -> bool {
        return onsetGapBiasEnabled.load(std::memory_order_relaxed);
    }
    auto setOnsetGapBiasEnabled(bool enabled) -> void {
        onsetGapBiasEnabled.store(enabled, std::memory_order_relaxed);
    }

    [[nodiscard]] auto getPitchTauBase() const -> float {
        return pitchTauBase.load(std::memory_order_relaxed);
    }
    auto setPitchTauBase(float value) -> void {
        pitchTauBase.store(value, std::memory_order_relaxed);
    }
    [[nodiscard]] auto getPitchTauPrimeBase() const -> float {
        return pitchTauPrimeBase.load(std::memory_order_relaxed);
    }
    auto setPitchTauPrimeBase(float value) -> void {
        pitchTauPrimeBase.store(value, std::memory_order_relaxed);
    }
    [[nodiscard]] auto getDurationTauBase() const -> float {
        return durationTauBase.load(std::memory_order_relaxed);
    }
    auto setDurationTauBase(float value) -> void {
        durationTauBase.store(value, std::memory_order_relaxed);
    }
    [[nodiscard]] auto getDurationTauPrimeBase() const -> float {
        return durationTauPrimeBase.load(std::memory_order_relaxed);
    }
    auto setDurationTauPrimeBase(float value) -> void {
        durationTauPrimeBase.store(value, std::memory_order_relaxed);
    }
    [[nodiscard]] auto getOnsetGapTauBase() const -> float {
        return onsetGapTauBase.load(std::memory_order_relaxed);
    }
    auto setOnsetGapTauBase(float value) -> void {
        onsetGapTauBase.store(value, std::memory_order_relaxed);
    }

    /**
     * Refresh the 5 s pitch window (π / s_y / S) and the τ / τ′ schedule using
     * now = max(clock, last onset). No-op when bias is disabled. See PITCH_BIAS.md.
     */
    auto refreshPitchTauSchedule() -> void;

    /** Same now rule as pitch; refreshes duration π / s_y and τ_d / τ′_d. No-op when disabled. */
    auto refreshDurationTauSchedule() -> void;

    /** Same now rule as pitch; refreshes onset-gap s_Δ and τ_g. No-op when disabled. */
    auto refreshOnsetGapTauSchedule() -> void;

    auto onGenerationPauseChanged(bool paused) -> void override;

    [[nodiscard]] auto getPitchTau() const -> float {
        return pitchTau.load(std::memory_order_relaxed);
    }
    [[nodiscard]] auto getPitchTauPrime() const -> float {
        return pitchTauPrime.load(std::memory_order_relaxed);
    }

    [[nodiscard]] auto getDurationTau() const -> float {
        return durationTau.load(std::memory_order_relaxed);
    }
    [[nodiscard]] auto getDurationTauPrime() const -> float {
        return durationTauPrime.load(std::memory_order_relaxed);
    }

    [[nodiscard]] auto getOnsetGapTau() const -> float {
        return onsetGapTau.load(std::memory_order_relaxed);
    }

private:
    auto applyQueuedInputToInputData() -> bool;

    auto applyUpdatesFromFilter() -> bool;

    void futureLogits(std::vector<float> &logits, int currentTime, int32_t forceAtTime);

    /** Hard-masks every note-vocab band except instrument 1 (the reduction). Always on. */
    void generationInstrumentLogits(std::vector<float> &logits);

    void velocityLogits(std::vector<float> &logits);

    void safeLogits(std::vector<float> &logits, size_t stepIdx);

    std::vector<float> runModelAndGetLogits(std::vector<int32_t> &tokens);

    Token generateNewToken(int32_t forceAtTime);

    static auto denseMinTime(std::vector<int32_t> &tokens) -> int32_t;

    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<std::string> allocatedInputNames;
    std::vector<std::string> allocatedOutputNames;

    std::atomic<float> onsetTemperature{DenseSampling::OnsetTemperature};
    std::atomic<float> durationTemperature{DenseSampling::DurationTemperature};
    std::atomic<float> noteTemperature{DenseSampling::NoteTemperature};
    std::atomic<float> velocityTemperature{DenseSampling::VelocityTemperature};

    std::atomic<bool> pitchBiasEnabled{true};
    std::atomic<bool> durationBiasEnabled{true};
    std::atomic<bool> onsetGapBiasEnabled{true};

    std::atomic<float> pitchTauBase{DensePitchBias::TauBase};
    std::atomic<float> pitchTauPrimeBase{DensePitchBias::TauBase};
    std::atomic<float> durationTauBase{DenseDurationBias::TauBase};
    std::atomic<float> durationTauPrimeBase{DenseDurationBias::TauPrimeBase};
    std::atomic<float> onsetGapTauBase{DenseOnsetGapBias::TauBase};

    juce::CriticalSection pitchBiasLock;
    DensePitchBias::PitchTauScheduler pitchTauScheduler;
    DensePitchBias::PitchWindowStats lastPitchWindow;
    std::atomic<float> pitchTau{DensePitchBias::TauBase};
    std::atomic<float> pitchTauPrime{DensePitchBias::TauBase};

    juce::CriticalSection durationBiasLock;
    DenseDurationBias::DurationTauScheduler durationTauScheduler;
    DenseDurationBias::DurationWindowStats lastDurationWindow;
    std::atomic<float> durationTau{DenseDurationBias::TauBase};
    std::atomic<float> durationTauPrime{DenseDurationBias::TauPrimeBase};

    juce::CriticalSection onsetGapBiasLock;
    DenseOnsetGapBias::OnsetGapTauScheduler onsetGapTauScheduler;
    DenseOnsetGapBias::OnsetGapWindowStats lastOnsetGapWindow;
    std::atomic<float> onsetGapTau{DenseOnsetGapBias::TauBase};

    DenseFirstTokenLogitsAudit firstTokenLogitsAudit;
};
