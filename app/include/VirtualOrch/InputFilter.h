#pragma once

#include <memory>
#include <vector>

#include "Fifo.h"
#include "VirtualOrch/MusicTransformer.h"

/**
 * Routes incoming tokens into MusicTransformer's token / conditioning queues.
 * Owns history copies for UI and future history-aware filters.
 * This comes BEFORE the transformer thread: tokensToSend -> inputFilter -> transformer thread.
 */
class InputFilter {
public:
    InputFilter(CircularFifo<Token> &inputTokenQueue,
                CircularFifo<Token> &inputConditioningQueue);

    virtual ~InputFilter() = default;

    virtual void filter(const Token &current) = 0;

    virtual void reset();

    [[nodiscard]] auto getPastTokens() const -> const std::vector<Token> & { return pastTokens; }

    [[nodiscard]] auto getPastConditioning() const -> const std::vector<Token> & {
        return pastConditioning;
    }

protected:
    void pushToken(const Token &token);

    void pushConditioning(const Token &token);

    std::vector<Token> pastTokens;
    std::vector<Token> pastConditioning;

    CircularFifo<Token> &inputTokenQueue;
    CircularFifo<Token> &inputConditioningQueue;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(InputFilter)
};

/** All tokens go to the token queue; conditioning unused. */
class PassthroughInputFilter : public InputFilter {
public:
    using InputFilter::InputFilter;

    void filter(const Token &current) override;
};

/** Pitches in [conditioningLow, conditioningHigh) → conditioning; else → token queue. */
class PitchRangeSplitInputFilter : public InputFilter {
public:
    PitchRangeSplitInputFilter(CircularFifo<Token> &inputTokenQueue,
                               CircularFifo<Token> &inputConditioningQueue,
                               int32_t conditioningLow,
                               int32_t conditioningHigh);

    void filter(const Token &current) override;

private:
    int32_t conditioningLow;
    int32_t conditioningHigh;
};

[[nodiscard]] auto createInputFilter(InputFilterType type,
                                     const ModelConfig &modelConfig,
                                     CircularFifo<Token> &inputTokenQueue,
                                     CircularFifo<Token> &inputConditioningQueue)
    -> std::unique_ptr<InputFilter>;
