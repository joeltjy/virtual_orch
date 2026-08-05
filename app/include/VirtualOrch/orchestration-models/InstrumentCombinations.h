#pragma once

#include <memory>
#include <utility>
#include <onnxruntime_cxx_api.h>

#include "VirtualOrch/orchestration-models/OrchestrationModel.h"

class InstrumentCombinations : public OrchestrationModel {
public:
    static constexpr int32_t numStrings = 8;
    static constexpr int32_t numWoodwinds = 3;
    static constexpr int32_t numBrass = 4;
    static constexpr int32_t numOther = 5;
    static constexpr int32_t durOffset = 10000;
    static constexpr int32_t noteOffset = 11000;
    static constexpr int32_t velocityOffset = 11256;
    static constexpr int32_t groupsOffset = 11384;
    static constexpr int32_t stringsOffset = 11399;
    static constexpr int32_t woodwindOffset = stringsOffset + (1 << numStrings);
    static constexpr int32_t brassOffset = woodwindOffset + (1 << numWoodwinds);
    static constexpr int32_t otherOffset = brassOffset + (1 << numBrass);
    static constexpr int32_t endNote = otherOffset + (1 << numOther);
    static constexpr int32_t sep = endNote + 1;
    static constexpr int32_t mask = sep + 1;
    static constexpr int32_t tokensPerNote = 9;
    static constexpr int32_t defaultVelocity = 100;
    static constexpr float sampleTemperature = 0.5f;

    InstrumentCombinations();

    auto init(const char *modelPath) -> void;

    [[nodiscard]] auto getName() const -> std::string override;

    [[nodiscard]] auto tokenHistory() const -> const std::vector<OrchestrationNote> & override;

    [[nodiscard]] auto getOutput(const std::vector<Token> &incomingTokens,
                                 const std::vector<int32_t> &instruments,
                                 const ConditioningSignal &conditioningSignal)
        -> std::vector<OrchestrationNote> override;

    /** Decode each 9-token group into one OrchestrationNote per active instrument. */
    [[nodiscard]] auto encodedCombinationToInstruments(const std::vector<int32_t> &sampledTokens) const
        -> std::vector<OrchestrationNote>;

    /** Allowed vocab ids for family slot 0=strings..3=other given active local instruments. */
    [[nodiscard]] auto allowedTokensForFamily(size_t familyIndex,
                                              const std::vector<int32_t> &instruments) const
        -> std::vector<int32_t>;

    /** Allowed vocab ids for the groups slot (token index 4) given active local instruments. */
    [[nodiscard]] auto allowedTokensForGroups(const std::vector<int32_t> &instruments) const
        -> std::vector<int32_t>;

private:
    [[nodiscard]] auto tokensToInput(const std::vector<Token> &incomingTokens) const
        -> std::vector<int32_t>;

    [[nodiscard]] auto prependHistoryToInput(const std::vector<int32_t> &newTokens) const
        -> std::vector<int32_t>;

    auto runModelAndGetLogits(std::vector<int32_t> &tokens)
        -> std::pair<std::vector<float>, size_t>;

    /** Mask combination slots (groups + 4 families) from the active instrument list. */
    auto maskLogits(std::vector<float> &logits,
                    size_t numTokens,
                    size_t vocab,
                    const std::vector<int32_t> &instruments) const -> void;

    [[nodiscard]] auto sample(const std::vector<int32_t> &maskedTokens,
                              std::vector<float> &logits,
                              size_t vocab) const -> std::vector<int32_t>;

    auto updateHistories(const std::vector<OrchestrationNote> &notes,
                         const std::vector<int32_t> &sampledTokens) -> void;

    std::vector<OrchestrationNote> history;

    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<std::string> allocatedInputNames;
    std::vector<std::string> allocatedOutputNames;
};
