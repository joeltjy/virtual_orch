#pragma once

#include <memory>
#include <utility>
#include <onnxruntime_cxx_api.h>

#include "VirtualOrch/orchestration-models/OrchestrationModel.h"

class InstrumentCombinations : public OrchestrationModel {
public:
    // Taxonomy (instrument_taxonomy_20): strings 0–6, woodwind 7–10, brass 11–14,
    // other 15–19. Family blocks are 2^familySize wide (combo 0 included but unused).
    static constexpr int32_t numStrings = 7;
    static constexpr int32_t numWoodwinds = 4;
    static constexpr int32_t numBrass = 4;
    static constexpr int32_t numOther = 5;
    static constexpr int32_t numFamilies = 4;
    static constexpr int32_t durOffset = 10000;
    static constexpr int32_t noteOffset = 11000;
    static constexpr int32_t velocityOffset = 11256;
    static constexpr int32_t groupsOffset = 11384;
    /** Groups has no empty-set id: combos 1..15 pack into 15 slots, shifted down by one. */
    static constexpr int32_t numGroupTokens = (1 << numFamilies) - 1;
    static constexpr int32_t stringsOffset = groupsOffset + numGroupTokens;
    static constexpr int32_t woodwindOffset = stringsOffset + (1 << numStrings);
    static constexpr int32_t brassOffset = woodwindOffset + (1 << numWoodwinds);
    static constexpr int32_t otherOffset = brassOffset + (1 << numBrass);
    static constexpr int32_t endNote = otherOffset + (1 << numOther);
    static constexpr int32_t sep = endNote + 1;
    static constexpr int32_t mask = sep + 1;
    /** Vocab implied by the offsets above; must equal the checkpoint's logits width. */
    static constexpr int32_t expectedVocabSize = mask + 1;
    static constexpr int32_t tokensPerNote = 9;
    static constexpr int32_t defaultVelocity = 100;
    static constexpr float sampleTemperature = 0.5f;

    /** Family bitmask (1..15) → groups vocab id. */
    [[nodiscard]] static constexpr auto groupsTokenForCombo(int32_t combo) -> int32_t {
        return groupsOffset + combo - 1;
    }

    /**
     * Groups vocab id → family bitmask. Ids outside the groups block decode to 0
     * (no active families), which emits no orchestration notes.
     */
    [[nodiscard]] static constexpr auto comboFromGroupsToken(int32_t token) -> int32_t {
        const int32_t combo = token - groupsOffset + 1;
        return (combo >= 1 && combo <= numGroupTokens) ? combo : 0;
    }

    InstrumentCombinations();

    auto init(const char *modelPath) -> void;

    [[nodiscard]] auto getName() const -> std::string override;

    [[nodiscard]] auto tokenHistory() const -> const std::vector<OrchestrationNote> & override;

    [[nodiscard]] auto getOutput(
        const std::vector<Token> &incomingTokens,
        const std::vector<int32_t> &instruments,
        const ConditioningSignal &conditioningSignal,
        OrchestrationBalanceTracker *balance = nullptr,
        const OrchestrationBalanceTracker::BiasView *bias = nullptr)
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

    /** Mask combination slots (groups + 4 families) from pitch ∩ allowed instruments. */
    auto maskLogits(std::vector<float> &logits,
                    size_t numTokens,
                    size_t vocab,
                    const std::vector<int32_t> &instruments,
                    const std::vector<int32_t> &newTokens) const -> void;

    /** Subtract Σ τ_i log p_i from groups + family combo logits. */
    auto applyProportionBias(std::vector<float> &logits,
                             size_t numTokens,
                             size_t vocab,
                             const OrchestrationBalanceTracker::BiasView &bias) const -> void;

    /** Fill sink with post-bias singleton combo logits from the last note in the batch. */
    auto publishSingletonInstrumentLogits(const std::vector<float> &logits,
                                          size_t numTokens,
                                          size_t vocab,
                                          bool biasApplied) const -> void;

    [[nodiscard]] auto sample(const std::vector<int32_t> &maskedTokens,
                              std::vector<float> &logits,
                              size_t vocab) const -> std::vector<int32_t>;

    auto updateHistories(const std::vector<OrchestrationNote> &notes,
                         const std::vector<int32_t> &sampledTokens) -> void;

    /** After decode: one recordNote per 9-token group in sampledTokens. */
    auto recordBalanceFromSampled(OrchestrationBalanceTracker &balance,
                                  const std::vector<int32_t> &sampledTokens) const -> void;

    std::vector<OrchestrationNote> history;

    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<std::string> allocatedInputNames;
    std::vector<std::string> allocatedOutputNames;

    /** Logits width of the loaded graph; 0 when unknown (dynamic last dim). */
    size_t modelVocabSize = 0;
};
