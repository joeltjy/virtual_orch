#include "VirtualOrch/reduction/ReductionTransformer.h"

#include "VirtualOrch/NoteWindow.h"
#include "VirtualOrch/reduction/DenseTypes.h"
#include "VirtualOrch/vocsep/VoiceSeparation.h"

#include <algorithm>
#include <cmath>

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

auto ReductionTransformer::hasUnresolvedProvisionalInputNotes() const -> bool {
    // Dense only (4 ints/note). Live keyboard is always instrument 0; Piano Reduction
    // keeps generated notes on instrument 1 so they are not mistaken for MIDI holds.
    const auto width = static_cast<int>(inputEventWidth());
    if (width < 4)
        return false;
    return DenseSampling::hasProvisionalDurationNotes(
        inputData, modelConfig.inputDuration, width,
        DenseConfig::PianoReductionInputInstrument);
}

auto ReductionTransformer::notifyPausedChanged() -> void {
    if (! onPausedChanged)
        return;
    juce::MessageManager::callAsync([callback = onPausedChanged] {
        if (callback)
            callback();
    });
}

auto ReductionTransformer::setGenerationPause(bool paused) -> void {
    if (generationPause.get() == paused)
        return;
    const bool wasPaused = generationPause.get();
    generationPause.set(paused);
    const int32_t clockCs =
        clock != nullptr ? static_cast<int32_t>(clock->getTime()) : -1;
    juce::Logger::writeToLog(
        "[rt] generationPause " + juce::String(wasPaused ? "true" : "false") + " -> "
        + juce::String(paused ? "true" : "false") + " clockCs=" + juce::String(clockCs)
        + " overflow=" + juce::String(overflowPause.get() ? "true" : "false")
        + " stopped="
        + juce::String(isGenerationStopped() ? "true" : "false")
        + " reason="
        + juce::String(generationPause.get() ? "generationPause"
                      : (overflowPause.get() ? "overflowPause" : "none")));
    if (wasPaused && ! paused)
        logFirstTokenAfterResume.store(true, std::memory_order_relaxed);
    onGenerationPauseChanged(paused);
    notifyPausedChanged();
}

auto ReductionTransformer::logGeneratedTokenIfResumed(int32_t onset) -> void {
    if (! logFirstTokenAfterResume.exchange(false, std::memory_order_relaxed))
        return;
    const int32_t clockCs =
        clock != nullptr ? static_cast<int32_t>(clock->getTime()) : -1;
    juce::Logger::writeToLog(
        "[rt] firstTokenAfterResume onset=" + juce::String(onset)
        + " currentTime=" + juce::String(currentTime)
        + " clockCs=" + juce::String(clockCs)
        + " deltaOnsetMinusClock=" + juce::String(onset - clockCs));
}

auto ReductionTransformer::startAheadThrottle() -> void {
    if (aheadThrottleUntilMs != 0)
        return;
    const auto sleepMs =
        static_cast<uint32_t>(std::max(0, modelConfig.outputAheadThrottleSeconds) * 1000);
    if (sleepMs == 0)
        return;
    overflowPause.set(true);
    aheadThrottleUntilMs = juce::Time::getMillisecondCounter() + sleepMs;
    const int32_t clockCs =
        clock != nullptr ? static_cast<int32_t>(clock->getTime()) : -1;
    juce::Logger::writeToLog(
        "[rt] overflowPause start throttleMs=" + juce::String(static_cast<int>(sleepMs))
        + " clockCs=" + juce::String(clockCs)
        + " currentTime=" + juce::String(currentTime));
}

auto ReductionTransformer::finishAheadThrottleIfDue() -> void {
    if (aheadThrottleUntilMs == 0)
        return;
    if (juce::Time::getMillisecondCounter() < aheadThrottleUntilMs)
        return;
    resetAheadThrottle();
}

auto ReductionTransformer::resetAheadThrottle() -> void {
    const bool wasOverflow = overflowPause.get();
    aheadThrottleUntilMs = 0;
    overflowPause.set(false);
    wasGenerationPaused = generationPause.get();
    if (wasOverflow) {
        const int32_t clockCs =
            clock != nullptr ? static_cast<int32_t>(clock->getTime()) : -1;
        juce::Logger::writeToLog(
            "[rt] overflowPause end clockCs=" + juce::String(clockCs)
            + " generationPause="
            + juce::String(generationPause.get() ? "true" : "false")
            + " stopped="
            + juce::String(isGenerationStopped() ? "true" : "false"));
    }
}

auto ReductionTransformer::maybeEmitGenerationPauseSoftStopClear() -> void {
    const bool paused = generationPause.get();
    if (paused && ! wasGenerationPaused && clock != nullptr) {
        const auto clearAt =
            static_cast<int32_t>(clock->getTime()) + Config::TimeResolution; // now + 1 s
        pushOutputToken(Token{clearAt, static_cast<int32_t>(Vocab::DurOffset),
                              static_cast<int32_t>(Vocab::ClearQueue)});
    }
    wasGenerationPaused = paused;
}

auto ReductionTransformer::notifyInputDataChanged() -> void {
    NoteWindow::trimStridedToLastNotes(inputData, inputEventWidth(),
                                       NoteWindow::maxReductionInputNotes);

    auto callback = onInputDataChanged;
    if (! callback)
        return;
    if (inputDataNotifyPending->exchange(true))
        return;

    auto snapshot = inputData;
    juce::MessageManager::callAsync([callback = std::move(callback),
                                     snapshot = std::move(snapshot),
                                     pending = inputDataNotifyPending]() mutable {
        pending->store(false);
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

auto ReductionTransformer::getOutputHistorySince(int32_t cutoffCs) const -> std::vector<Token> {
    const juce::ScopedLock lock(outputHistoryLock);
    return NoteWindow::filteredTokens(outputHistory, cutoffCs);
}

auto ReductionTransformer::clearOutputHistory() -> void {
    const juce::ScopedLock lock(outputHistoryLock);
    outputHistory.clear();
}

auto ReductionTransformer::pushOutputToken(const Token &tokenIn) -> void {
    Token token = tokenIn;
    if (voiceSeparation != nullptr) {
        std::vector<Token> historyCopy;
        {
            const juce::ScopedLock histLock(outputHistoryLock);
            historyCopy = outputHistory;
        }
        const auto restamps = voiceSeparation->stampVoiceId(token, historyCopy);
        if (! restamps.empty()) {
            const juce::ScopedLock histLock(outputHistoryLock);
            for (const auto &r: restamps) {
                if (r.historyIndex < outputHistory.size())
                    outputHistory[r.historyIndex].voiceId = r.newVoiceId;
            }
        }
    }
    outputTokenQueue.push(token);
    if (orchestrationReductionIncoming != nullptr)
        orchestrationReductionIncoming->push(token);
    const juce::ScopedLock lock(outputHistoryLock);
    outputHistory.push_back(token);
    NoteWindow::trimToLastNotes(outputHistory);
}

auto ReductionTransformer::recordThreadLoopMs(double loopStartHiResMs) -> void {
    const auto elapsed =
        static_cast<float>(juce::Time::getMillisecondCounterHiRes() - loopStartHiResMs);
    lastThreadLoopMs.store(elapsed, std::memory_order_relaxed);
    const auto prevMax = lastThreadLoopMsMax.load(std::memory_order_relaxed);
    if (elapsed > prevMax)
        lastThreadLoopMsMax.store(elapsed, std::memory_order_relaxed);
}

auto ReductionTransformer::resetLoopTiming() -> void {
    lastThreadLoopMs.store(0.0f, std::memory_order_relaxed);
    lastThreadLoopMsMax.store(0.0f, std::memory_order_relaxed);
    resetReductionProfile();
}

auto ReductionTransformer::resetReductionProfile() -> void {
    loopAccum = {};
    const juce::ScopedLock lock(reductionProfileLock);
    reductionProfile = {};
}

auto ReductionTransformer::publishBusyReductionProfile(ReductionLoopStepMs step) -> void {
    const juce::ScopedLock lock(reductionProfileLock);
    reductionProfile.publishBusy(step);
}

auto ReductionTransformer::getLastReductionProfile() const -> ReductionLoopProfile {
    const juce::ScopedLock lock(reductionProfileLock);
    return reductionProfile;
}

auto ReductionTransformer::getDurationLogitSnapshot() const -> DurationLogitSnapshot {
    const juce::ScopedLock lock(durationLogitLock);
    return durationLogitSnapshot;
}

auto ReductionTransformer::publishDurationLogits(const std::vector<float> &scores,
                                                 int32_t sampledDurToken) -> void {
    using namespace DenseVocab;
    if (scores.size() < NoteOffset)
        return;

    DurationLogitSnapshot snap;
    for (size_t cs = 0; cs < DurationLogitSnapshot::kNumBins; ++cs) {
        const float v = scores[DurOffset + cs];
        const bool ok = std::isfinite(v);
        snap.valid[cs] = ok;
        snap.logits[cs] = ok ? v : 0.0f;
    }
    if (sampledDurToken >= static_cast<int32_t>(DurOffset)
        && sampledDurToken < static_cast<int32_t>(NoteOffset))
        snap.sampledCs = sampledDurToken - static_cast<int32_t>(DurOffset);

    const juce::ScopedLock lock(durationLogitLock);
    snap.sequence = durationLogitSnapshot.sequence + 1;
    durationLogitSnapshot = snap;
}
