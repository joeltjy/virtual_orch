#include "VirtualOrch/OrchestrationTransformer.h"

#include "VirtualOrch/InstrumentConstants.h"
#include "VirtualOrch/NoteWindow.h"

#include <algorithm>
#include <iostream>
#include <set>

namespace {

auto snapshotHasValidLogit(const InstrumentLogitSnapshot &snap) -> bool {
    return std::any_of(snap.valid.begin(), snap.valid.end(), [](bool v) { return v; });
}

} // namespace

OrchestrationTransformer::OrchestrationTransformer()
    : Thread("Orchestration Transformer") {
}

void OrchestrationTransformer::threadInit() {
}

auto OrchestrationTransformer::applyOneInstrumentUpdate(const InstrumentUpdate &update) -> void {
    if (! InstrumentConstants::isValidLocalInstrumentId(update.localInstrumentId))
        return;
    if (update.state != 0 && update.state != 1)
        return;

    const juce::ScopedLock lock(instrumentsLock);
    const bool isUser = update.target == InstrumentUpdateTarget::User;
    auto &instruments = isUser ? userInstruments : modelInstruments;
    auto &balance = isUser ? userBalance : modelBalance;

    if (update.state == 0) {
        instruments.erase(update.localInstrumentId);
        balance.onInstrumentRemoved(update.localInstrumentId);
    } else {
        instruments.insert(update.localInstrumentId);
        balance.onInstrumentAdded(update.localInstrumentId);
    }
}

auto OrchestrationTransformer::applyInstrumentUpdates() -> void {
    InstrumentUpdate update{};
    while (instrumentUpdates.pull(update))
        applyOneInstrumentUpdate(update);
}

auto OrchestrationTransformer::replaceInstrumentSets(std::set<int32_t> user,
                                                     std::set<int32_t> model) -> void {
    const juce::ScopedLock lock(instrumentsLock);
    userInstruments = std::move(user);
    modelInstruments = std::move(model);
}

auto OrchestrationTransformer::copyInstrumentLists() const
    -> std::pair<std::vector<int32_t>, std::vector<int32_t>> {
    const juce::ScopedLock lock(instrumentsLock);
    return {{userInstruments.begin(), userInstruments.end()},
            {modelInstruments.begin(), modelInstruments.end()}};
}

auto OrchestrationTransformer::applyTokenUpdates() -> void {
    TokenUpdate update{};
    while (updatesIncoming.pull(update)) {
        for (auto &note: outputHistory) {
            // Orch notes store pitch-only note ids; keyboard updates embed instrument.
            if (! tokenEqualsByOnsetPitch(note.token, update.oldNote))
                continue;
            note.token.duration = update.newNote.duration;
            note.token.velocity = update.newNote.velocity;
        }

        // OT midi history is retagged as piano (15); note-off updates use reduction instr 0.
        {
            TokenUpdate midiUpdate = update;
            if (update.oldNote.note >= static_cast<int32_t>(Vocab::NoteOffset)
                && update.oldNote.note < static_cast<int32_t>(Vocab::Rest)) {
                midiUpdate.oldNote =
                    update.oldNote.withInstrument(InstrumentConstants::kKeyboardLocalInstrumentId);
                midiUpdate.newNote =
                    update.newNote.withInstrument(InstrumentConstants::kKeyboardLocalInstrumentId);
            }
            applyTokenUpdateToHistory(midiInputHistory, midiUpdate);
            applyTokenUpdateToHistory(conditioningHistory, midiUpdate);
        }

        // Playback queue also holds pitch-only orch tokens — strip instrument for match.
        TokenUpdate playbackUpdate = update;
        if (update.oldNote.note >= static_cast<int32_t>(Vocab::NoteOffset)
            && update.oldNote.note < static_cast<int32_t>(Vocab::Rest)) {
            playbackUpdate.oldNote = update.oldNote.withInstrument(0);
            playbackUpdate.newNote = update.newNote.withInstrument(0);
        }
        outputDurationUpdates.push(playbackUpdate);
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

auto OrchestrationTransformer::getDebugSnapshotSince(int32_t cutoffCs) const
    -> OrchestrationDebugSnapshot {
    const juce::ScopedLock lock(debugSnapshotLock);
    OrchestrationDebugSnapshot trimmed;
    trimmed.midiHistory = NoteWindow::filteredTokens(debugSnapshot.midiHistory, cutoffCs);
    trimmed.midiPending = NoteWindow::filteredTokens(debugSnapshot.midiPending, cutoffCs);
    trimmed.conditioningHistory =
        NoteWindow::filteredTokens(debugSnapshot.conditioningHistory, cutoffCs);
    trimmed.conditioningPending =
        NoteWindow::filteredTokens(debugSnapshot.conditioningPending, cutoffCs);
    trimmed.reductionHistory = NoteWindow::filteredTokens(debugSnapshot.reductionHistory, cutoffCs);
    trimmed.reductionPending = NoteWindow::filteredTokens(debugSnapshot.reductionPending, cutoffCs);
    trimmed.outputHistory =
        NoteWindow::filtered(debugSnapshot.outputHistory, cutoffCs, [](const OrchestrationNote &n) {
            return std::pair<int32_t, int32_t>{n.token.time, n.token.getRealDuration()};
        });
    trimmed.userInstruments = debugSnapshot.userInstruments;
    trimmed.modelInstruments = debugSnapshot.modelInstruments;
    return trimmed;
}

auto OrchestrationTransformer::publishInstrumentLogits(InstrumentLogitSnapshot snapshot) -> void {
    const juce::ScopedLock lock(instrumentLogitLock);
    snapshot.sequence = instrumentLogitSnapshot.sequence + 1;
    instrumentLogitSnapshot = std::move(snapshot);
}

auto OrchestrationTransformer::getInstrumentLogitSnapshot() const -> InstrumentLogitSnapshot {
    const juce::ScopedLock lock(instrumentLogitLock);
    return instrumentLogitSnapshot;
}

auto OrchestrationTransformer::clearDebugSnapshot() -> void {
    const juce::ScopedLock lock(debugSnapshotLock);
    debugSnapshot = {};
}

auto OrchestrationTransformer::resetOrchProfile() -> void {
    getOutputAccum = {};
    const juce::ScopedLock lock(orchProfileLock);
    orchProfile = {};
}

auto OrchestrationTransformer::publishBusyOrchProfile(OrchLoopStepMs step) -> void {
    const juce::ScopedLock lock(orchProfileLock);
    orchProfile.publishBusy(step);
}

auto OrchestrationTransformer::getLastOrchProfile() const -> OrchLoopProfile {
    const juce::ScopedLock lock(orchProfileLock);
    return orchProfile;
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
    NoteWindow::trimToLastNotes(history);
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

auto OrchestrationTransformer::extractClearQueueTokens(std::vector<Token> &tokens)
    -> std::vector<Token> {
    std::vector<Token> clears;
    std::vector<Token> notes;
    notes.reserve(tokens.size());
    for (const auto &token: tokens) {
        if (token.note == static_cast<int32_t>(Vocab::ClearQueue))
            clears.push_back(token);
        else
            notes.push_back(token);
    }
    tokens = std::move(notes);
    return clears;
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

    const bool applyBias = proportionBiasEnabled.get();
    const auto nowMs = juce::Time::getMillisecondCounter();

    std::vector<OrchestrationNote> userNotes;
    if (! userInstruments.empty()) {
        const auto userBias = userBalance.makeBiasView(applyBias, nowMs);
        InstrumentLogitSnapshot logitScratch;
        orchestrationModel->instrumentLogitSink = &logitScratch;
        userNotes = orchestrationModel->getOutput(midiInput,
                                                  userInstruments,
                                                  signal,
                                                  &userBalance,
                                                  &userBias);
        orchestrationModel->instrumentLogitSink = nullptr;
        if (snapshotHasValidLogit(logitScratch))
            publishInstrumentLogits(std::move(logitScratch));
    }

    std::vector<OrchestrationNote> modelNotes;
    if (! modelInstruments.empty()) {
        const auto modelBias = modelBalance.makeBiasView(applyBias, nowMs);
        InstrumentLogitSnapshot logitScratch;
        orchestrationModel->instrumentLogitSink = &logitScratch;
        modelNotes = orchestrationModel->getOutput(reductionInput,
                                                   modelInstruments,
                                                   signal,
                                                   &modelBalance,
                                                   &modelBias);
        orchestrationModel->instrumentLogitSink = nullptr;
        // Prefer model stream for the bar chart when both run.
        if (snapshotHasValidLogit(logitScratch))
            publishInstrumentLogits(std::move(logitScratch));
    }

    return {std::move(userNotes), std::move(modelNotes)};
}

auto OrchestrationTransformer::getJamOrchestrationOutput(
    const std::vector<Token> &orchestrationInput,
    const std::vector<int32_t> &orchestrationInstruments,
    const ConditioningSignal &signal) -> std::vector<OrchestrationNote> {
    if (orchestrationModel == nullptr || orchestrationInstruments.empty())
        return {};

    const bool applyBias = proportionBiasEnabled.get();
    const auto nowMs = juce::Time::getMillisecondCounter();
    std::set<int32_t> userSnap;
    std::set<int32_t> modelSnap;
    {
        const juce::ScopedLock lock(instrumentsLock);
        userSnap = userInstruments;
        modelSnap = modelInstruments;
    }
    // Jam counts live on modelBalance (reduction stream); τ comes from the owning pad set.
    auto &counts = modelSnap.empty() ? userBalance : modelBalance;
    const auto jamBias = OrchestrationBalanceTracker::makeJamBiasView(userBalance,
                                                                      modelBalance,
                                                                      counts,
                                                                      userSnap,
                                                                      modelSnap,
                                                                      applyBias,
                                                                      nowMs);
    InstrumentLogitSnapshot logitScratch;
    orchestrationModel->instrumentLogitSink = &logitScratch;
    auto notes = orchestrationModel->getOutput(orchestrationInput,
                                               orchestrationInstruments,
                                               signal,
                                               &counts,
                                               &jamBias);
    orchestrationModel->instrumentLogitSink = nullptr;
    if (snapshotHasValidLogit(logitScratch))
        publishInstrumentLogits(std::move(logitScratch));
    return notes;
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
    NoteWindow::trimToLastNotes(outputHistory);
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
    {
        const juce::ScopedLock lock(instrumentLogitLock);
        instrumentLogitSnapshot = {};
    }
    // Prefer Launchpad effective pads as source of truth. Clearing the FIFO without
    // applying would drop pad/side changes made while generation was stopped.
    clearInstrumentUpdates();
    if (rebuildInstrumentsFromPads)
        rebuildInstrumentsFromPads();
    userBalance.reset();
    modelBalance.reset();
    {
        const juce::ScopedLock lock(instrumentsLock);
        userBalance.syncActive(userInstruments);
        modelBalance.syncActive(modelInstruments);
    }
    resetOrchProfile();
    if (orchestrationModel != nullptr)
        orchestrationModel->getOutputTimings = &getOutputAccum;

    while (! threadShouldExit()) {
        const double loopStartMs = juce::Time::getMillisecondCounterHiRes();
        OrchLoopStepMs step{};
        getOutputAccum = {};

        applyInstrumentUpdates();

        std::vector<Token> midiUpdate;
        std::vector<Token> conditioningUpdate;
        std::vector<Token> reductionUpdate;
        {
            const ScopedMs drainMs(&step.drain);
            midiUpdate = drainIncoming(midiInputIncoming);
            conditioningUpdate = drainIncoming(conditioningIncoming);
            reductionUpdate = drainIncoming(reductionIncoming);
        }
        step.midiIn = static_cast<int>(midiUpdate.size());
        step.reductionIn = static_cast<int>(reductionUpdate.size());

        {
            const ScopedMs snapMs(&step.snap);
            OrchestrationDebugSnapshot snapshot;
            snapshot.midiHistory = midiInputHistory;
            snapshot.midiPending = midiUpdate;
            snapshot.conditioningHistory = conditioningHistory;
            snapshot.conditioningPending = conditioningUpdate;
            snapshot.reductionHistory = reductionHistory;
            snapshot.reductionPending = reductionUpdate;
            snapshot.outputHistory = outputHistory;
            {
                const juce::ScopedLock lock(instrumentsLock);
                snapshot.userInstruments = userInstruments;
                snapshot.modelInstruments = modelInstruments;
            }
            publishDebugSnapshot(std::move(snapshot));
        }

        std::vector<Token> clearFromMidi;
        std::vector<Token> clearFromReduction;
        OrchestrationTransformer::ConditioningSignal signal{};
        {
            const ScopedMs histMs(&step.hist);
            clearFromMidi = extractClearQueueTokens(midiUpdate);
            clearFromReduction = extractClearQueueTokens(reductionUpdate);
            signal = getConditioningSignal(midiUpdate, conditioningUpdate);
            appendToHistory(midiInputHistory, midiUpdate);
            appendToHistory(conditioningHistory, conditioningUpdate);
            appendToHistory(reductionHistory, reductionUpdate);
        }

        std::vector<OrchestrationNote> toOutput;
        if (! paused.get()) {
            const auto [userInstrumentList, modelInstrumentList] = copyInstrumentLists();

            const auto mode = getMode();
            if (mode == OrchestrationMode::Edit) {
                const ScopedMs modelMs(&step.model);
                auto [userNotes, modelNotes] = getEditOrchestrationOutput(
                    midiUpdate, reductionUpdate, userInstrumentList, modelInstrumentList, signal);
                toOutput.reserve(userNotes.size() + modelNotes.size() + clearFromMidi.size()
                                 + clearFromReduction.size());
                toOutput.insert(toOutput.end(), userNotes.begin(), userNotes.end());
                toOutput.insert(toOutput.end(), modelNotes.begin(), modelNotes.end());
            } else if (mode == OrchestrationMode::Jam) {
                const ScopedMs modelMs(&step.model);
                std::set<int32_t> instrumentUnion(userInstrumentList.begin(),
                                                  userInstrumentList.end());
                instrumentUnion.insert(modelInstrumentList.begin(), modelInstrumentList.end());
                const std::vector<int32_t> orchestrationInstruments(instrumentUnion.begin(),
                                                                    instrumentUnion.end());
                toOutput = getJamOrchestrationOutput(reductionUpdate, orchestrationInstruments,
                                                     signal);
                toOutput.reserve(toOutput.size() + clearFromMidi.size() + clearFromReduction.size());
            } else {
                juce::Logger::writeToLog("Invalid orchestration mode");
                step.total = static_cast<float>(
                    juce::Time::getMillisecondCounterHiRes() - loopStartMs);
                lastThreadLoopMs.store(step.total, std::memory_order_relaxed);
                continue;
            }
        } else {
            toOutput.reserve(clearFromMidi.size() + clearFromReduction.size());
        }

        step.prep = getOutputAccum.prep;
        step.onnx = getOutputAccum.onnx;
        step.logits = getOutputAccum.logits;
        step.mask = getOutputAccum.mask;
        step.sample = getOutputAccum.sample;
        step.decode = getOutputAccum.decode;
        step.getOutputCalls = getOutputAccum.getOutputCalls;
        step.contextTokens = getOutputAccum.contextTokens;

        {
            const ScopedMs packMs(&step.pack);
            // Bypass orch model: ClearQueue must reach OutputPlayback as-is.
            for (const auto &token: clearFromMidi)
                toOutput.push_back(OrchestrationNote{
                    .token = token,
                    .velocity = 0,
                    .localInstrumentId = InstrumentConstants::kReductionPlaybackLocalId});
            for (const auto &token: clearFromReduction)
                toOutput.push_back(OrchestrationNote{
                    .token = token,
                    .velocity = 0,
                    .localInstrumentId = InstrumentConstants::kReductionPlaybackLocalId});

            packSortAndPushOutput(toOutput);
        }

        // After notes are queued so OutputPlayback cannot drain a duration update
        // before the matching note exists (same-loop note-on + note-off race).
        {
            const ScopedMs updMs(&step.upd);
            applyTokenUpdates();
        }

        step.total =
            static_cast<float>(juce::Time::getMillisecondCounterHiRes() - loopStartMs);
        lastThreadLoopMs.store(step.total, std::memory_order_relaxed);

        const bool busy = step.midiIn > 0 || step.reductionIn > 0 || step.getOutputCalls > 0
                          || ! conditioningUpdate.empty();
        if (busy)
            publishBusyOrchProfile(step);
    }
}

void OrchestrationTransformer::threadStop() {
}
