#include "VirtualOrch/InputFilter.h"

#include "VirtualOrch/InstrumentConstants.h"

InputFilter::InputFilter(CircularFifo<Token> &inputTokenQueueIn,
                         CircularFifo<Token> &inputConditioningQueueIn,
                         CircularFifo<TokenUpdate> &updatesFromFilterIn)
    : inputTokenQueue(inputTokenQueueIn),
      inputConditioningQueue(inputConditioningQueueIn),
      updatesFromFilter(updatesFromFilterIn) {
}

void InputFilter::reset() {
    pastInput.clear();
    pastConditioning.clear();
    TokenUpdate u{};
    while (updatesFromMain.pull(u)) {
    }
}

auto InputFilter::processUpdates() -> int {
    int applied = 0;
    TokenUpdate update{};
    while (updatesFromMain.pull(update)) {
        const bool updatedInput = applyTokenUpdateToHistory(pastInput, update);
        const bool updatedConditioning = applyTokenUpdateToHistory(pastConditioning, update);
        if (updatedInput || updatedConditioning) {
            updatesFromFilter.push(update);
            ++applied;
        }
    }
    return applied;
}

void InputFilter::pushToken(const Token &token) {
    inputTokenQueue.push(token);
    pastInput.push_back(token);
    if (orchestrationMidiOutgoing != nullptr)
        orchestrationMidiOutgoing->push(
            token.withInstrument(InstrumentConstants::kKeyboardLocalInstrumentId));
}

void InputFilter::pushConditioning(const Token &token) {
    inputConditioningQueue.push(token);
    pastConditioning.push_back(token);
    if (orchestrationConditioningOutgoing != nullptr)
        orchestrationConditioningOutgoing->push(token);
}

void PassthroughInputFilter::filter(const Token &current) {
    pushToken(current);
}

PitchRangeSplitInputFilter::PitchRangeSplitInputFilter(CircularFifo<Token> &inputTokenQueueIn,
                                                       CircularFifo<Token> &inputConditioningQueueIn,
                                                       CircularFifo<TokenUpdate> &updatesFromFilterIn,
                                                       int32_t conditioningLowIn,
                                                       int32_t conditioningHighIn)
    : InputFilter(inputTokenQueueIn, inputConditioningQueueIn, updatesFromFilterIn),
      conditioningLow(conditioningLowIn),
      conditioningHigh(conditioningHighIn) {
}

void PitchRangeSplitInputFilter::filter(const Token &current) {
    const auto pitch = current.getPitch();
    if (pitch >= conditioningLow && pitch < conditioningHigh)
        pushConditioning(current);
    else
        pushToken(current);
}

auto createInputFilter(InputFilterType type,
                       const ModelConfig &modelConfig,
                       CircularFifo<Token> &inputTokenQueue,
                       CircularFifo<Token> &inputConditioningQueue,
                       CircularFifo<TokenUpdate> &updatesFromFilter)
    -> std::unique_ptr<InputFilter> {
    switch (type) {
        case InputFilterType::Passthrough:
            return std::make_unique<PassthroughInputFilter>(inputTokenQueue,
                                                            inputConditioningQueue,
                                                            updatesFromFilter);
        case InputFilterType::PitchRangeSplit:
            return std::make_unique<PitchRangeSplitInputFilter>(
                inputTokenQueue,
                inputConditioningQueue,
                updatesFromFilter,
                modelConfig.filterConditioningLow,
                modelConfig.filterConditioningHigh);
    }

    return std::make_unique<PassthroughInputFilter>(inputTokenQueue,
                                                    inputConditioningQueue,
                                                    updatesFromFilter);
}
