#pragma once

#include <JuceHeader.h>
#include <onnxruntime_cxx_api.h>

#include <atomic>
#include <memory>
#include <vector>

#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/reduction/DensePitchBias.h"
#include "VirtualOrch/reduction/DenseTypes.h"
#include "VirtualOrch/reduction/ReductionTransformer.h"

/**
 * Dense reduction V2: same 4 tokens per note dense ONNX as V1, with lighter RT clamps and
 * online pitch-bias sampling (see documentation in PITCH_BIAS.md).
 * What this does: prevent endless repeated notes or scales
 */
class ReductionTransformerV2 : public ReductionTransformer {
public:
    explicit ReductionTransformerV2(ModelConfig &modelConfig);

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

    /**
     * Refresh the 5 s pitch window (π / s_y / S) and the τ / τ′ schedule using
     * now = max(clock, last onset) (amt_causal prefix-relative when ahead of
     * the clock). Safe to call from the UI timer so τ keeps ramping once the
     * clock catches up. See PITCH_BIAS.md.
     */
    auto refreshPitchTauSchedule() -> void;

    [[nodiscard]] auto getPitchTau() const -> float {
        return pitchTau.load(std::memory_order_relaxed);
    }
    [[nodiscard]] auto getPitchTauPrime() const -> float {
        return pitchTauPrime.load(std::memory_order_relaxed);
    }

private:
    auto applyQueuedInputToInputData() -> bool;

    auto applyUpdatesFromFilter() -> bool;

    void futureLogits(std::vector<float> &logits, int currentTime, int32_t forceAtTime);

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
    std::atomic<float> durationTemperature{DenseSampling::V2DurationTemperature};
    std::atomic<float> noteTemperature{DenseSampling::NoteTemperature};
    std::atomic<float> velocityTemperature{DenseSampling::VelocityTemperature};

    juce::CriticalSection pitchBiasLock;
    DensePitchBias::PitchTauScheduler pitchTauScheduler;
    DensePitchBias::PitchWindowStats lastPitchWindow;
    std::atomic<float> pitchTau{DensePitchBias::TauBase};
    std::atomic<float> pitchTauPrime{DensePitchBias::TauBase};
};
