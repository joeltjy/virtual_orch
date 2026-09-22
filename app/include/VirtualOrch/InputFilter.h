#pragma once

#include <memory>
#include <vector>

#include "Fifo.h"
#include "VirtualOrch/MusicTransformer.h"

/**
 * Routes incoming tokens into the reduction queues and optionally OT fifos.
 * Owns history copies for UI and future history-aware filters.
 * MIDI/message-thread path: tokensToSend -> inputFilter -> reduction (+ OT).
 */
class InputFilter {
public:
    InputFilter(CircularFifo<Token> &inputTokenQueue,
                CircularFifo<Token> &inputConditioningQueue,
                CircularFifo<TokenUpdate> &updatesFromFilter);

    virtual ~InputFilter() = default;

    virtual void filter(const Token &current) = 0;

    virtual void reset();

    /** Drain updatesFromMain; patch pastInput / pastConditioning; forward to updatesFromFilter.
     *  Returns how many updates were applied to at least one history. */
    auto processUpdates() -> int;

    [[nodiscard]] auto getPastInput() const -> const std::vector<Token> & { return pastInput; }

    [[nodiscard]] auto getPastConditioning() const -> const std::vector<Token> & {
        return pastConditioning;
    }

    CircularFifo<TokenUpdate> updatesFromMain;

    /** Optional: live keyboard tokens for OT (retagged as piano). */
    CircularFifo<Token> *orchestrationMidiOutgoing = nullptr;

    /** Optional: conditioning tokens for OT. */
    CircularFifo<Token> *orchestrationConditioningOutgoing = nullptr;

protected:
    void pushToken(const Token &token);

    void pushConditioning(const Token &token);

    std::vector<Token> pastInput;
    std::vector<Token> pastConditioning;

    CircularFifo<Token> &inputTokenQueue;
    CircularFifo<Token> &inputConditioningQueue;
    CircularFifo<TokenUpdate> &updatesFromFilter;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(InputFilter)
};

/** All tokens go to the token queue; conditioning unused. */
class PassthroughInputFilter : public InputFilter {
public:
    using InputFilter::InputFilter;

    void filter(const Token &current) override;
};

/** Pitches in [conditioningLow, conditioningHigh) are conditioning. */
class PitchRangeSplitInputFilter : public InputFilter {
public:
    PitchRangeSplitInputFilter(CircularFifo<Token> &inputTokenQueue,
                               CircularFifo<Token> &inputConditioningQueue,
                               CircularFifo<TokenUpdate> &updatesFromFilter,
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
                                     CircularFifo<Token> &inputConditioningQueue,
                                     CircularFifo<TokenUpdate> &updatesFromFilter)
    -> std::unique_ptr<InputFilter>;
