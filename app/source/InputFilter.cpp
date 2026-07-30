#include "VirtualOrch/InputFilter.h"

InputFilter::InputFilter(CircularFifo<Token> &inputTokenQueueIn,
                         CircularFifo<Token> &inputConditioningQueueIn)
    : inputTokenQueue(inputTokenQueueIn),
      inputConditioningQueue(inputConditioningQueueIn) {
}

void InputFilter::reset() {
    pastTokens.clear();
    pastConditioning.clear();
}

void InputFilter::pushToken(const Token &token) {
    inputTokenQueue.push(token);
    pastTokens.push_back(token);
}

void InputFilter::pushConditioning(const Token &token) {
    inputConditioningQueue.push(token);
    pastConditioning.push_back(token);
}

void PassthroughInputFilter::filter(const Token &current) {
    pushToken(current);
}

PitchRangeSplitInputFilter::PitchRangeSplitInputFilter(CircularFifo<Token> &inputTokenQueueIn,
                                                       CircularFifo<Token> &inputConditioningQueueIn,
                                                       int32_t conditioningLowIn,
                                                       int32_t conditioningHighIn)
    : InputFilter(inputTokenQueueIn, inputConditioningQueueIn),
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
                       CircularFifo<Token> &inputConditioningQueue)
    -> std::unique_ptr<InputFilter> {
    switch (type) {
        case InputFilterType::Passthrough:
            return std::make_unique<PassthroughInputFilter>(inputTokenQueue, inputConditioningQueue);
        case InputFilterType::PitchRangeSplit:
            return std::make_unique<PitchRangeSplitInputFilter>(
                inputTokenQueue,
                inputConditioningQueue,
                modelConfig.filterConditioningLow,
                modelConfig.filterConditioningHigh);
    }

    return std::make_unique<PassthroughInputFilter>(inputTokenQueue, inputConditioningQueue);
}
