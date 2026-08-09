#include "VirtualOrch/ReductionTransformer.h"

#include <algorithm>

ReductionTransformer::ReductionTransformer(const juce::String &threadName, ModelConfig &modelConfigIn)
    : Thread(threadName),
      modelConfig(modelConfigIn) {
}

auto ReductionTransformer::isGeneratedTooFarAhead(int32_t generatedTime) const -> bool {
    if (clock == nullptr || modelConfig.outputMaxAheadSeconds <= 0)
        return false;
    const auto maxAheadTicks = modelConfig.outputMaxAheadSeconds * Config::TimeResolution;
    return generatedTime - static_cast<int32_t>(clock->getTime()) > maxAheadTicks;
}

auto ReductionTransformer::notifyPausedChanged() -> void {
    if (! onPausedChanged)
        return;
    juce::MessageManager::callAsync([callback = onPausedChanged] {
        if (callback)
            callback();
    });
}

auto ReductionTransformer::startAheadThrottle() -> void {
    if (aheadThrottleUntilMs != 0)
        return;
    const auto sleepMs =
        static_cast<uint32_t>(std::max(0, modelConfig.outputAheadThrottleSeconds) * 1000);
    if (sleepMs == 0)
        return;
    pausedBeforeAheadThrottle = paused.get();
    paused.set(true);
    aheadThrottleUntilMs = juce::Time::getMillisecondCounter() + sleepMs;
    notifyPausedChanged();
}

auto ReductionTransformer::finishAheadThrottleIfDue() -> void {
    if (aheadThrottleUntilMs == 0)
        return;
    if (juce::Time::getMillisecondCounter() < aheadThrottleUntilMs)
        return;
    aheadThrottleUntilMs = 0;
    paused.set(pausedBeforeAheadThrottle);
    notifyPausedChanged();
}

auto ReductionTransformer::notifyInputDataChanged() -> void {
    auto callback = onInputDataChanged;
    if (! callback)
        return;
    auto snapshot = inputData;
    juce::MessageManager::callAsync(
        [callback = std::move(callback), snapshot = std::move(snapshot)]() mutable {
            callback(std::move(snapshot));
        });
}

auto ReductionTransformer::clearInputTokenQueue() -> void {
    Token t{.time = -1, .duration = -1, .note = -1};
    while (inputTokenQueue.pull(t)) {
    }
}

auto ReductionTransformer::clearInputConditioningQueue() -> void {
    Token t{.time = -1, .duration = -1, .note = -1};
    while (inputConditioningQueue.pull(t)) {
    }
}

auto ReductionTransformer::clearOutputTokenQueue() -> void {
    Token t{.time = -1, .duration = -1, .note = -1};
    while (outputTokenQueue.pull(t)) {
    }
}

auto ReductionTransformer::clearUpdatesFromFilter() -> void {
    TokenUpdate u{};
    while (updatesFromFilter.pull(u)) {
    }
}

auto ReductionTransformer::getOutputHistory() const -> std::vector<Token> {
    const juce::ScopedLock lock(outputHistoryLock);
    return outputHistory;
}

auto ReductionTransformer::clearOutputHistory() -> void {
    const juce::ScopedLock lock(outputHistoryLock);
    outputHistory.clear();
}

auto ReductionTransformer::pushOutputToken(const Token &token) -> void {
    outputTokenQueue.push(token);
    if (orchestrationReductionIncoming != nullptr)
        orchestrationReductionIncoming->push(token);
    const juce::ScopedLock lock(outputHistoryLock);
    outputHistory.push_back(token);
}
