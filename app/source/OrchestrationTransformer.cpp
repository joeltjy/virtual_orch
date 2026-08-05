#include "VirtualOrch/OrchestrationTransformer.h"

#include "VirtualOrch/InstrumentConstants.h"

#include <algorithm>
#include <iostream>
#include <set>

OrchestrationTransformer::OrchestrationTransformer()
    : Thread("Orchestration Transformer") {
}

void OrchestrationTransformer::threadInit() {
}

auto OrchestrationTransformer::applyInstrumentUpdates() -> void {
    InstrumentUpdate update{};
    while (instrumentUpdates.pull(update)) {
        if (! InstrumentConstants::isValidLocalInstrumentId(update.localInstrumentId))
            continue;

        if (update.state != 0 && update.state != 1)
            continue;

        auto &instruments = update.target == InstrumentUpdateTarget::User ? userInstruments
                                                                          : modelInstruments;

        if (update.state == 0)
            instruments.erase(update.localInstrumentId);
        else
            instruments.insert(update.localInstrumentId);
    }
}

auto OrchestrationTransformer::applyTokenUpdates() -> void {
    TokenUpdate update{};
    while (updatesIncoming.pull(update)) {
        applyTokenUpdateToHistory(midiInputHistory, update);
        applyTokenUpdateToHistory(conditioningHistory, update);

        for (auto &note: outputHistory) {
            if (tokenEquals(note.token, update.oldNote))
                note.token = update.newNote;
        }
        
        // for notes that were given the default duration.
        outputDurationUpdates.push(update);
    }
}

auto OrchestrationTransformer::publishDebugSnapshot(OrchestrationDebugSnapshot snapshot) -> void {
    const juce::ScopedLock lock(debugSnapshotLock);
    debugSnapshot = std::move(snapshot);
}

auto OrchestrationTransformer::getDebugSnapshot() const -> OrchestrationDebugSnapshot {
    const juce::ScopedLock lock(debugSnapshotLock);
    return debugSnapshot;
}

auto OrchestrationTransformer::clearDebugSnapshot() -> void {
    const juce::ScopedLock lock(debugSnapshotLock);
    debugSnapshot = {};
}

auto OrchestrationTransformer::drainIncoming(CircularFifo<Token> &incoming) -> std::vector<Token> {
    std::vector<Token> batch;
    Token token{};
    while (incoming.pull(token))
        batch.push_back(token);
    return batch;
}

auto OrchestrationTransformer::appendToHistory(std::vector<Token> &history,
                                               const std::vector<Token> &batch) -> void {
    history.insert(history.end(), batch.begin(), batch.end());
}

auto OrchestrationTransformer::tokenWithInstrument(Token token, int32_t instrument) -> Token {
    const auto pitch = token.getPitch();
    token.note = static_cast<int32_t>(Vocab::NoteOffset + Config::MaxPitch * instrument + pitch);
    return token;
}

auto OrchestrationTransformer::sortTokensByTimeThenDuration(std::vector<Token> &tokens) -> void {
    std::ranges::sort(tokens, [](const Token &a, const Token &b) {
        if (a.time != b.time)
            return a.time < b.time;
        return a.duration < b.duration;
    });
}

auto OrchestrationTransformer::getConditioningSignal(const std::vector<Token> &midiInput,
                                                     const std::vector<Token> &conditioning)
    -> ConditioningSignal {
    juce::ignoreUnused(midiInput, conditioning);
    return ConditioningSignal{};
}

auto OrchestrationTransformer::getEditOrchestrationOutput(
    const std::vector<Token> &midiInput,
    const std::vector<Token> &reductionInput,
    const std::vector<int32_t> &userInstruments,
    const std::vector<int32_t> &modelInstruments,
    const ConditioningSignal &signal)
    -> std::pair<std::vector<OrchestrationNote>, std::vector<OrchestrationNote>> {
    if (orchestrationModel == nullptr)
        return {};

    auto userNotes = orchestrationModel->getOutput(midiInput, userInstruments, signal);
    auto modelNotes = orchestrationModel->getOutput(reductionInput, modelInstruments, signal);
    return {std::move(userNotes), std::move(modelNotes)};
}

auto OrchestrationTransformer::getJamOrchestrationOutput(
    const std::vector<Token> &orchestrationInput,
    const std::vector<int32_t> &orchestrationInstruments,
    const ConditioningSignal &signal) -> std::vector<OrchestrationNote> {
    if (orchestrationModel == nullptr)
        return {};

    return orchestrationModel->getOutput(orchestrationInput, orchestrationInstruments, signal);
}

auto OrchestrationTransformer::packSortAndPushOutput(const std::vector<OrchestrationNote> &notes)
    -> void {
    auto packed = notes;
    std::ranges::sort(packed, [](const OrchestrationNote &a, const OrchestrationNote &b) {
        if (a.token.time != b.token.time)
            return a.token.time < b.token.time;
        return a.token.duration < b.token.duration;
    });

    for (const auto &note: packed)
        pushOutputNote(note);
}

auto OrchestrationTransformer::pushOutputNote(const OrchestrationNote &note) -> void {
    currentOTTime = note.token.time;
    outputTokenQueue.push(note);
    outputHistory.push_back(note);
}

auto OrchestrationTransformer::clearOutputNotes() -> void {
    OrchestrationNote n{};
    while (outputTokenQueue.pull(n)) {
    }
    outputHistory.clear();
    currentOTTime = 0;
}

void OrchestrationTransformer::threadRun() {
    midiInputHistory.clear();
    conditioningHistory.clear();
    reductionHistory.clear();
    clearMidiInputIncoming();
    clearConditioningIncoming();
    clearReductionIncoming();
    clearOutputNotes();
    clearInstrumentUpdates();
    clearUpdatesIncoming();
    clearOutputDurationUpdates();
    clearDebugSnapshot();

    while (! threadShouldExit()) {
        const double loopStartMs = juce::Time::getMillisecondCounterHiRes();

        applyInstrumentUpdates();

        const auto midiUpdate = drainIncoming(midiInputIncoming);
        const auto conditioningUpdate = drainIncoming(conditioningIncoming);
        const auto reductionUpdate = drainIncoming(reductionIncoming);

        {
            OrchestrationDebugSnapshot snapshot;
            snapshot.midiHistory = midiInputHistory;
            snapshot.midiPending = midiUpdate;
            snapshot.conditioningHistory = conditioningHistory;
            snapshot.conditioningPending = conditioningUpdate;
            snapshot.reductionHistory = reductionHistory;
            snapshot.reductionPending = reductionUpdate;
            snapshot.outputHistory = outputHistory;
            snapshot.userInstruments = userInstruments;
            snapshot.modelInstruments = modelInstruments;
            publishDebugSnapshot(std::move(snapshot));
        }

        const auto signal = getConditioningSignal(midiUpdate, conditioningUpdate);
        appendToHistory(midiInputHistory, midiUpdate);
        appendToHistory(conditioningHistory, conditioningUpdate);
        appendToHistory(reductionHistory, reductionUpdate);
        applyTokenUpdates();

        const std::vector<int32_t> userInstrumentList(userInstruments.begin(), userInstruments.end());
        const std::vector<int32_t> modelInstrumentList(modelInstruments.begin(),
                                                      modelInstruments.end());

        std::vector<OrchestrationNote> toOutput;
        if (mode == OrchestrationMode::Edit) {
            auto [userNotes, modelNotes] = getEditOrchestrationOutput(
                midiUpdate, reductionUpdate, userInstrumentList, modelInstrumentList, signal);
            toOutput.reserve(userNotes.size() + modelNotes.size());
            toOutput.insert(toOutput.end(), userNotes.begin(), userNotes.end());
            toOutput.insert(toOutput.end(), modelNotes.begin(), modelNotes.end());
        } else if (mode == OrchestrationMode::Jam) {
            std::set<int32_t> instrumentUnion = userInstruments;
            instrumentUnion.insert(modelInstruments.begin(), modelInstruments.end());
            const std::vector<int32_t> orchestrationInstruments(instrumentUnion.begin(),
                                                                instrumentUnion.end());
            toOutput = getJamOrchestrationOutput(reductionUpdate, orchestrationInstruments, signal);
        } else {
            juce::Logger::writeToLog("Invalid orchestration mode");
            continue;
        }

        packSortAndPushOutput(toOutput);

        const double loopMs = juce::Time::getMillisecondCounterHiRes() - loopStartMs;
        // std::cout << "OT loop: " << loopMs << " ms"
        //           << " hist(m/c/r)=" << midiInputHistory.size() << '/'
        //           << conditioningHistory.size() << '/' << reductionHistory.size()
        //           << std::endl;
    }
}

void OrchestrationTransformer::threadStop() {
}
