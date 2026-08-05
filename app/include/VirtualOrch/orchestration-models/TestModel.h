#pragma once

#include <memory>
#include <onnxruntime_cxx_api.h>

#include "VirtualOrch/orchestration-models/OrchestrationModel.h"

class TestModel : public OrchestrationModel {
public:
    static constexpr int32_t kDurOffset = 10000;
    static constexpr int32_t kNoteOffset = 11000;
    static constexpr int32_t kVelocityOffset = 11256;
    static constexpr int32_t kDefaultVelocity = 100;
    static constexpr float kSampleTemperature = 0.5f;

    TestModel();

    auto init(const char *modelPath) -> void;

    [[nodiscard]] auto getName() const -> std::string override;

    [[nodiscard]] auto tokenHistory() const -> const std::vector<OrchestrationNote> & override;

    [[nodiscard]] auto getOutput(const std::vector<Token> &incomingTokens,
                                 const std::vector<int32_t> &instruments,
                                 const ConditioningSignal &conditioningSignal)
        -> std::vector<OrchestrationNote> override;

private:
    auto runModelAndGetLogits(std::vector<int32_t> &tokens) -> std::vector<float>;

    std::vector<OrchestrationNote> history;

    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<std::string> allocatedInputNames;
    std::vector<std::string> allocatedOutputNames;
};
