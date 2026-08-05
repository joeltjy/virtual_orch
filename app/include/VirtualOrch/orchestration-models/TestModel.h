#pragma once

#include <memory>
#include <onnxruntime_cxx_api.h>

#include "VirtualOrch/orchestration-models/OrchestrationModel.h"

class TestModel : public OrchestrationModel {
public:
    static constexpr int32_t durOffset = 10000;
    static constexpr int32_t noteOffset = 11000;
    static constexpr int32_t velocityOffset = 11256;
    static constexpr int32_t defaultVelocity = 100;
    static constexpr float sampleTemperature = 0.5f;

    TestModel();

    auto init(const char *modelPath) -> void;

    [[nodiscard]] auto getName() const -> std::string override;

    [[nodiscard]] auto tokenHistory() const -> const std::vector<OrchestrationNote> & override;

    [[nodiscard]] auto getOutput(const std::vector<Token> &incomingTokens,
                                 const std::vector<int32_t> &instruments,
                                 const ConditioningSignal &conditioningSignal)
        -> std::vector<OrchestrationNote> override;

private:
    [[nodiscard]] auto tokensToInput(const std::vector<Token> &incomingTokens) const
        -> std::vector<int32_t>;

    [[nodiscard]] auto prependHistoryToInput(const std::vector<int32_t> &newTokens) const
        -> std::vector<int32_t>;

    auto runModelAndGetLogits(std::vector<int32_t> &tokens) -> std::vector<float>;

    auto maskLogits(std::vector<float> &logits, size_t numNotes, size_t numInstruments) const -> void;

    [[nodiscard]] auto sample(const std::vector<Token> &incomingTokens,
                              std::vector<float> &newNoteLogits,
                              size_t numInstruments) const -> std::vector<OrchestrationNote>;

    auto updateHistories(const std::vector<OrchestrationNote> &notes,
                         const std::vector<int32_t> &newTokens) -> void;

    std::vector<OrchestrationNote> history;

    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<std::string> allocatedInputNames;
    std::vector<std::string> allocatedOutputNames;
};
