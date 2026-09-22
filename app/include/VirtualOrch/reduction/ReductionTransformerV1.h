#pragma once

#include <JuceHeader.h>
#include <onnxruntime_cxx_api.h>

#include <atomic>
#include <memory>
#include <vector>

#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/reduction/DenseTypes.h"
#include "VirtualOrch/reduction/ReductionTransformer.h"

/**
 * Dense reduction V1 (initial version): 4-wide events (time, dur, note, velocity).
 * ONNX: input_ids -> logits (no past KV / anticipate). Keeps RT clamps (maxFuture,
 * duration ceiling, instrLogits).
 */
class ReductionTransformerV1 : public ReductionTransformer {
public:
    explicit ReductionTransformerV1(ModelConfig &modelConfig);

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

    /**
     * Dynamic ceiling (centiseconds) for the next duration token:
     *   4 * max(75th-percentile duration, 40)
     * over the previous 15 notes of `denseInputData`.
     */
    [[nodiscard]] static auto durationCeilingCs(const std::vector<int32_t> &denseInputData)
        -> int32_t;

private:
    auto applyQueuedInputToInputData() -> bool;

    auto applyUpdatesFromFilter() -> bool;

    void instrLogits(std::vector<float> &logits);

    void futureLogits(std::vector<float> &logits, int currentTime, int32_t forceAtTime);

    void durLogits(std::vector<float> &logits);

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

    const int32_t maximumFuture = 500;
};
