#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "VirtualOrch/orchestration-models/IodPretrainedTypes.h"
#include "VirtualOrch/orchestration-models/OrchestrationModel.h"

/** iod_pretrained / instrument_octave_predictions.onnx orchestration backend. */
class IodPretrained : public OrchestrationModel {
public:
    IodPretrained();

    auto init(const char *modelPath) -> void;

    [[nodiscard]] auto getName() const -> std::string override;

    [[nodiscard]] auto tokenHistory() const -> const std::vector<OrchestrationNote> & override;

    [[nodiscard]] auto isModelLoaded() const -> bool { return session != nullptr; }

    /** Logits layout: [note][step][vocab] = nNotesMax × comboLen × comboVocabSize. */
    [[nodiscard]] auto runLogits(IodPretrainedTypes::EncoderWindow &window)
        -> std::vector<float>;

    /** Greedy AR; updates window.comboIn. One packed combo (len 5) per valid note. */
    [[nodiscard]] auto sampleWindowCombos(IodPretrainedTypes::EncoderWindow &window,
                                          const std::vector<int32_t> &instruments)
        -> std::vector<std::array<int32_t, IodPretrainedTypes::comboLen>>;

    [[nodiscard]] auto getOutput(
        const std::vector<Token> &incomingTokens,
        const std::vector<int32_t> &instruments,
        const ConditioningSignal &conditioningSignal,
        OrchestrationBalanceTracker *balance = nullptr,
        const OrchestrationBalanceTracker::BiasView *bias = nullptr)
        -> std::vector<OrchestrationNote> override;

private:
    [[nodiscard]] auto sampleComboForNote(IodPretrainedTypes::EncoderWindow &window,
                                          int32_t noteIdx,
                                          const std::vector<int32_t> &instruments)
        -> std::array<int32_t, IodPretrainedTypes::comboLen>;

    std::vector<OrchestrationNote> history;

    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<std::string> allocatedInputNames;
    std::vector<std::string> allocatedOutputNames;

    size_t modelVocabSize = 0;
};
