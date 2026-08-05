#pragma once

#include <array>
#include <string>
#include <vector>

#include "VirtualOrch/MusicTransformer.h"

struct OrchestrationNote {
    Token token;
    int32_t velocity = 0;
    int32_t localInstrumentId = 0;
};

/**
 * Base class for orchestration models (non-threaded computation only).
 */
class OrchestrationModel {
public:
    static constexpr size_t CONDITIONING_SIGNAL_DIM = 64;
    using ConditioningSignal = std::array<float, CONDITIONING_SIGNAL_DIM>;

    virtual ~OrchestrationModel() = default;

    [[nodiscard]] virtual auto getName() const -> std::string = 0;

    [[nodiscard]] virtual auto tokenHistory() const -> const std::vector<OrchestrationNote> & = 0;

    [[nodiscard]] virtual auto getOutput(const std::vector<Token> &incomingTokens,
                                         const std::vector<int32_t> &instruments,
                                         const ConditioningSignal &conditioningSignal)
        -> std::vector<OrchestrationNote> = 0;

    int32_t maxContextLength = 1024;
    std::vector<int32_t> tokenHistorySequence;
};
