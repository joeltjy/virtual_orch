#include "VirtualOrch/OrchestrationTransformer.h"
#include "VirtualOrch/TemporalDensity.h"
#include "VirtualOrch/OpLog.h"
#include "VirtualOrch/AppTime.h"
#include "VirtualOrch/PedalClear.h"
#include "VirtualOrch/SessionLog.h"

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


namespace {
auto otItemId(const Token &t) -> std::string {
    if (t.note == static_cast<int32_t>(Vocab::ClearQueue))
        return "CLEAR c" + std::to_string(t.time);
    if (t.keyId >= 0)
        return OpLog::id('k', t.keyId);
    return OpLog::id('r', t.noteId);
}
auto otIds(const std::vector<Token> &tokens) -> OpLog::Strings {
    OpLog::Strings ids;
    ids.reserve(tokens.size());
    for (const auto &t: tokens)
        ids.push_back(otItemId(t));
    return ids;
}
auto modeName(OrchestrationMode m) -> const char * {
    // "Step-in" is the user-facing name of OrchestrationMode::Edit (docs/OPLOGS.txt).
    return m == OrchestrationMode::Edit ? "Step-in" : m == OrchestrationMode::Jam ? "Jam" : "Accompaniment";
}
} // namespace

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

auto OrchestrationTransformer::applyVoiceRestamps(std::vector<Token> &reductionBatch) -> void {
    VoiceRestamp restamp{};
    while (restampsIncoming.pull(restamp)) {
        applyVoiceRestampToTokens(reductionBatch, restamp);
        applyVoiceRestampToTokens(reductionHistory, restamp);
        holdBack.applyRestamp(restamp);
        if (orchestrationModel != nullptr)
            orchestrationModel->applyVoiceRestamp(restamp.noteId, restamp.newVoiceId);
    }
}

auto OrchestrationTransformer::stampInputVoices(std::vector<Token> &midiBatch) -> void {
    if (inputVoiceSeparation == nullptr || ! inputVoiceSeparation->isLoaded())
        return;
    for (auto &token: midiBatch) {
        if (token.note < static_cast<int32_t>(Vocab::NoteOffset) || token.note >= static_cast<int32_t>(Vocab::Rest)
            || token.voiceId >= 0)
            continue;
        if (token.noteId < 0)
            token.noteId = nextInputNoteId++;
        if (const auto length = releasedLengthOf(token.keyId))
            token.duration = *length;
        const auto restamps = inputVoiceSeparation->stampVoiceId(token, inputVocsepHistory);
        for (const auto &r: restamps) {
            if (r.historyIndex >= inputVocsepHistory.size())
                continue;
            auto &old = inputVocsepHistory[r.historyIndex];
            old.voiceId = r.newVoiceId;
            const VoiceRestamp restamp{old.noteId, r.newVoiceId};
            applyVoiceRestampToTokens(midiBatch, restamp);
            if (orchestrationModel != nullptr)
                orchestrationModel->applyVoiceRestamp(restamp.noteId, restamp.newVoiceId);
        }
        inputVocsepHistory.push_back(token);
    }
    constexpr size_t kKeep = static_cast<size_t>(VoiceSeparation::WindowNotes) * 2;
    if (inputVocsepHistory.size() > kKeep)
        inputVocsepHistory.erase(inputVocsepHistory.begin(),
                                 inputVocsepHistory.end() - static_cast<std::ptrdiff_t>(kKeep));
}

auto OrchestrationTransformer::rememberReleasedKey(const TokenUpdate &update) -> void {
    if (update.oldNote.keyId < 0)
        return;
    std::erase_if(releasedKeyLengths, [&](const auto &e) { return e.first == update.oldNote.keyId; });
    releasedKeyLengths.emplace_back(update.oldNote.keyId, update.newNote.duration);
    if (releasedKeyLengths.size() > kReleasedKeysKept)
        releasedKeyLengths.erase(releasedKeyLengths.begin());
}

auto OrchestrationTransformer::releasedLengthOf(int32_t keyId) const -> std::optional<int32_t> {
    if (keyId < 0)
        return std::nullopt;
    for (auto it = releasedKeyLengths.rbegin(); it != releasedKeyLengths.rend(); ++it)
        if (it->first == keyId)
            return it->second;
    return std::nullopt;
}

auto OrchestrationTransformer::applyTokenUpdates() -> void {
    TokenUpdate update{};
    while (updatesIncoming.pull(update)) {
        rememberReleasedKey(update); // for notes made from this key later (see pushOutputNote)
        // The stamped user notes keep their real lengths for vocsep (matched by keyId).
        for (auto &t: inputVocsepHistory)
            if (tokenMatchesUpdateSource(t, update.oldNote))
                t.duration = update.newNote.duration;
        for (auto &note: outputHistory) {
            // Orch notes store pitch-only note ids; keyboard updates embed instrument.
            // Keyboard notes match by keyId (onset / pitch / velocity may have changed).
            if (! tokenMatchesUpdateSource(note.token, update.oldNote))
                continue;
            note.token.duration = update.newNote.duration;
            if (update.oldNote.keyId < 0)
                note.token.velocity = update.newNote.velocity;
        }

        // Hybrid v1.3: a reduction note's real length (the model's OFF; it went out with a provisional one) for every
        // note the OT made from it, whatever pitch / velocity it was given (matched by the source note id), and for
        // the post-OT TD meter (a provisional 5 s tremolo would count 40 events).
        if (update.oldNote.keyId < 0 && update.oldNote.noteId >= 0) {
            for (auto &note: outputHistory) {
                if (note.sourceNoteId != update.oldNote.noteId || note.token.duration == update.newNote.duration)
                    continue;
                TokenUpdate own{note.token, note.token};
                own.newNote.duration = update.newNote.duration;
                note.token.duration = update.newNote.duration;
                outputDurationUpdates.push(own);
            }
            tdMeter.setDuration(update.oldNote.noteId, update.newNote.getRealDuration());
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

namespace {
/** 30 ms at Config::TimeResolution (100 cs/s → 10 ms/cs). */
constexpr int32_t kOnsetClusterTolCs = 3;

[[nodiscard]] auto isSoundingNoteToken(const Token &token) -> bool {
    return token.note >= static_cast<int32_t>(Vocab::NoteOffset)
           && token.note < static_cast<int32_t>(Vocab::Rest);
}
} // namespace

auto OrchestrationTransformer::snapNearbyOnsets(std::vector<Token> &tokens) -> void {
    std::vector<size_t> idxs;
    idxs.reserve(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (isSoundingNoteToken(tokens[i]))
            idxs.push_back(i);
    }
    if (idxs.size() < 2)
        return;

    std::ranges::sort(idxs, [&](size_t a, size_t b) {
        if (tokens[a].time != tokens[b].time)
            return tokens[a].time < tokens[b].time;
        return a < b;
    });

    size_t clusterBegin = 0;
    while (clusterBegin < idxs.size()) {
        size_t clusterEnd = clusterBegin + 1;
        while (clusterEnd < idxs.size()
               && tokens[idxs[clusterEnd]].time - tokens[idxs[clusterEnd - 1]].time
                      <= kOnsetClusterTolCs) {
            ++clusterEnd;
        }
        const size_t n = clusterEnd - clusterBegin;
        if (n >= 2) {
            int64_t sum = 0;
            for (size_t i = clusterBegin; i < clusterEnd; ++i)
                sum += tokens[idxs[i]].time;
            // Round mean to nearest 10 ms (1 cs) grid.
            const auto snapped =
                static_cast<int32_t>((sum + static_cast<int64_t>(n) / 2) / static_cast<int64_t>(n));
            for (size_t i = clusterBegin; i < clusterEnd; ++i)
                tokens[idxs[i]].time = snapped;
        }
        clusterBegin = clusterEnd;
    }
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
    const auto nowMs = AppTime::counterMs();

    std::vector<OrchestrationNote> userNotes;
    if (! userInstruments.empty()) {
        auto snappedMidi = midiInput;
        snapNearbyOnsets(snappedMidi);
        const auto userBias = userBalance.makeBiasView(applyBias, nowMs);
        InstrumentLogitSnapshot logitScratch;
        orchestrationModel->instrumentLogitSink = &logitScratch;
        orchestrationModel->octaveDeltaSink = &octaveDeltaTracker;
        orchestrationModel->currentStream = OrchestrationModel::InputStream::User;
        userNotes = orchestrationModel->getOutput(snappedMidi,
                                                  userInstruments,
                                                  signal,
                                                  &userBalance,
                                                  &userBias);
        orchestrationModel->instrumentLogitSink = nullptr;
        orchestrationModel->octaveDeltaSink = nullptr;
        if (snapshotHasValidLogit(logitScratch))
            publishInstrumentLogits(std::move(logitScratch));
    }

    std::vector<OrchestrationNote> modelNotes;
    if (! modelInstruments.empty()) {
        auto snappedReduction = reductionInput;
        snapNearbyOnsets(snappedReduction);
        const auto modelBias = modelBalance.makeBiasView(applyBias, nowMs);
        InstrumentLogitSnapshot logitScratch;
        orchestrationModel->instrumentLogitSink = &logitScratch;
        orchestrationModel->octaveDeltaSink = &octaveDeltaTracker;
        orchestrationModel->currentStream = OrchestrationModel::InputStream::Reduction;
        modelNotes = orchestrationModel->getOutput(snappedReduction,
                                                   modelInstruments,
                                                   signal,
                                                   &modelBalance,
                                                   &modelBias);
        orchestrationModel->instrumentLogitSink = nullptr;
        orchestrationModel->octaveDeltaSink = nullptr;
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
    const auto nowMs = AppTime::counterMs();
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
    orchestrationModel->octaveDeltaSink = &octaveDeltaTracker;
    orchestrationModel->currentStream = OrchestrationModel::InputStream::Reduction;
    auto snappedInput = orchestrationInput;
    snapNearbyOnsets(snappedInput);
    auto notes = orchestrationModel->getOutput(snappedInput,
                                               orchestrationInstruments,
                                               signal,
                                               &counts,
                                               &jamBias);
    orchestrationModel->instrumentLogitSink = nullptr;
    orchestrationModel->octaveDeltaSink = nullptr;
    if (snapshotHasValidLogit(logitScratch))
        publishInstrumentLogits(std::move(logitScratch));
    return notes;
}

auto OrchestrationTransformer::packSortAndPushOutput(const std::vector<OrchestrationNote> &notes)
    -> void {
    auto packed = notes;
    PedalClear::sortForOutput(packed);

    for (const auto &note: packed)
        pushOutputNote(note);
}

auto OrchestrationTransformer::pushOutputNote(const OrchestrationNote &noteIn) -> void {
    if (holdOutput && holdOutput(noteIn))
        return; // the replay delivers it later (deliverOutputNote)
    deliverOutputNote(noteIn);
}

auto OrchestrationTransformer::deliverOutputNote(const OrchestrationNote &noteIn) -> void {
    OrchestrationNote note = noteIn;
    // The key was already released: the note takes its real length, not the provisional one.
    if (const auto length = releasedLengthOf(note.token.keyId))
        note.token.duration = *length;
    if (OpLog::on()) {
        if (note.token.note == static_cast<int32_t>(Vocab::ClearQueue)) {
            OpLog::emit(OpLog::Src::Ot, "OT_CLEAR", {{"from", note.token.time}});
            OpLog::emit(OpLog::Src::Ot, "Q_PUSH", {{"queue", "ot→play"}, {"item", otItemId(note.token)}});
        } else {
            note.outputId = OpLog::nextOutputId();
            const auto o = OpLog::id('o', note.outputId);
            OpLog::emit(OpLog::Src::Ot, "OT_NOTE",
                        {{"o", o},
                         {"from", note.token.keyId >= 0 ? OpLog::id('k', note.token.keyId) : OpLog::id('r', note.sourceNoteId)},
                         {"instrument", note.localInstrumentId}, {"pitch", note.token.getPitch()}, {"vel", note.velocity},
                         {"onset", note.token.time}, {"dur", note.token.getRealDuration()}, {"voice", note.token.voiceId}});
            OpLog::emit(OpLog::Src::Ot, "Q_PUSH", {{"queue", "ot→play"}, {"item", o}});
        }
    }
    // the orchestra's TD after the OT: the model's notes only (keyId < 0), not the pianist's orchestrated keys
    if (note.token.note == static_cast<int32_t>(Vocab::ClearQueue))
        tdMeter.clearFrom(note.token.time);
    else if (note.token.keyId < 0)
        tdMeter.add(note.token.time, note.token.getRealDuration(),
                    note.localInstrumentId == OrchestraTdMeter::kTremoloLocalId, note.sourceNoteId);
    currentOTTime = note.token.time;
    SessionLog::add("OT", juce::String(note.token.toUnderstandableString()) + " instr "
                              + juce::String(note.localInstrumentId) + " vel "
                              + juce::String(note.velocity) + " voice "
                              + juce::String(note.token.voiceId));
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
    tdMeter.clear();
    tremoloBias.reset();
    tremoloBiasValue.store(0.0f);
    if (orchestrationModel != nullptr)
        orchestrationModel->setTremoloBias(0.0f);
}

auto OrchestrationTransformer::orchestraTd() const -> std::optional<OrchestraTdMeter::Reading> {
    const auto nowCs = clock != nullptr ? std::optional<int32_t>(static_cast<int32_t>(clock->getTime())) : std::nullopt;
    auto r = tdMeter.read(nowCs);
    if (! r.has_value() || r->events <= 0.0)
        return std::nullopt; // nothing sounding or starting in the last 3 s (paused, silent): the RT counts itself
    return r;
}

auto OrchestrationTransformer::tickTremoloBias() -> void {
    const double nowMs = juce::Time::getMillisecondCounterHiRes();
    if (lastTremoloTickMs >= 0.0 && nowMs - lastTremoloTickMs < TremoloBiasController::kTickMs)
        return;
    lastTremoloTickMs = nowMs;
    const bool running = ! paused.get() && (clock == nullptr || clock->isRunning());
    const auto reading = running ? orchestraTd() : std::nullopt;
    const int32_t target = TemporalDensity::get();
    const double bonus = tremoloBias.tick(target, reading.has_value() ? std::optional<double>(reading->td) : std::nullopt);
    tremoloBiasValue.store(static_cast<float>(bonus));
    if (orchestrationModel != nullptr)
        orchestrationModel->setTremoloBias(static_cast<float>(bonus));
    if (OpLog::on() && target != TemporalDensity::kAuto)
        OpLog::emit(OpLog::Src::Ot, "OT_TD",
                    {{"td", target}, {"measured", reading.has_value() ? reading->td : -1.0},
                     {"events", reading.has_value() ? reading->events : 0.0},
                     {"tremolo_events", reading.has_value() ? reading->tremoloEvents : 0.0},
                     {"tremolo_bias", bonus}});
}

void OrchestrationTransformer::threadRun() {
    beginSteps();
    while (! threadShouldExit())
        stepOnce();
}

auto OrchestrationTransformer::beginSteps() -> void {
    holdBack.clear();
    heldCountValue.store(0);
    midiInputHistory.clear();
    inputVocsepHistory.clear();
    releasedKeyLengths.clear();
    nextInputNoteId = kInputNoteIdBase;
    if (inputVoiceSeparation != nullptr)
        inputVoiceSeparation->reset();
    conditioningHistory.clear();
    reductionHistory.clear();
    clearMidiInputIncoming();
    clearConditioningIncoming();
    clearReductionIncoming();
    clearOutputNotes();
    clearInstrumentUpdates();
    clearUpdatesIncoming();
    clearRestampsIncoming();
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
    octaveDeltaTracker.reset();
    {
        const juce::ScopedLock lock(instrumentsLock);
        userBalance.syncActive(userInstruments);
        modelBalance.syncActive(modelInstruments);
    }
    resetOrchProfile();
    if (orchestrationModel != nullptr) {
        orchestrationModel->resetState();
        orchestrationModel->getOutputTimings = &getOutputAccum;
    }
}

auto OrchestrationTransformer::stepOnce() -> void {
    const double loopStartMs = juce::Time::getMillisecondCounterHiRes();
    OrchLoopStepMs step{};
    getOutputAccum = {};

    applyInstrumentUpdates();
    tickTremoloBias();

    std::vector<Token> midiUpdate;
    std::vector<Token> conditioningUpdate;
    std::vector<Token> reductionUpdate;
    {
        const ScopedMs drainMs(&step.drain);
        midiUpdate = drainIncoming(midiInputIncoming);
        conditioningUpdate = drainIncoming(conditioningIncoming);
        reductionUpdate = drainIncoming(reductionIncoming);
    }
    applyVoiceRestamps(reductionUpdate);
    const bool logging = OpLog::on();
    if (logging) {
        for (const auto &t: midiUpdate)
            OpLog::emit(OpLog::Src::Ot, "Q_POP", {{"queue", "input→ot"}, {"item", otItemId(t)}});
        for (const auto &t: reductionUpdate)
            OpLog::emit(OpLog::Src::Ot, "Q_POP", {{"queue", "rt→ot"}, {"item", otItemId(t)}});
    }
    const auto heldBefore = logging ? otIds(holdBack.notes()) : OpLog::Strings{};
    // Reduction notes wait here (un-orchestrated) until their onset is within the hold-back
    // window; clears erase what they cover, in arrival order. Only the released notes reach
    // the model, with the instrument list that is current now.
    std::vector<Token> reductionForModel;
    {
        holdBack.accept(reductionUpdate);
        const int32_t nowCs = clock != nullptr ? static_cast<int32_t>(clock->getTime()) : -1;
        if (paused.get()) {
            // Paused: nothing is released, and notes that are already past are dropped.
            if (nowCs >= 0)
                holdBack.dropPast(nowCs);
            else
                holdBack.clear();
        } else {
            reductionForModel = holdBack.release(nowCs, holdBackWindowCs.load());
        }
        heldCountValue.store(holdBack.size());
        if (logging) {
            // hold-back diff: what arrived, what was released, what was dropped (and why)
            std::set<std::string> released;
            for (const auto &t: reductionForModel)
                released.insert(otItemId(t));
            const auto heldAfter = otIds(holdBack.notes());
            const std::set<std::string> after(heldAfter.begin(), heldAfter.end());
            std::set<std::string> before(heldBefore.begin(), heldBefore.end());
            std::string lastClear;
            OpLog::Strings droppedByClear, droppedPaused;
            for (const auto &t: reductionUpdate) {
                if (t.note == static_cast<int32_t>(Vocab::ClearQueue)) {
                    lastClear = otItemId(t);
                    continue;
                }
                const auto id = otItemId(t);
                if (! after.count(id) && ! released.count(id) && ! paused.get())
                    droppedByClear.push_back(id); // arrived and was cleared in the same batch
                OpLog::emit(OpLog::Src::Ot, "OT_HOLD", {{"id", id}, {"onset", t.time}});
                before.insert(id);
            }
            for (const auto &id: heldBefore)
                if (! after.count(id) && ! released.count(id))
                    (paused.get() ? droppedPaused : droppedByClear).push_back(id);
            if (! droppedByClear.empty())
                OpLog::emit(OpLog::Src::Ot, "OT_DROP", {{"ids", std::move(droppedByClear)}, {"reason", lastClear.empty() ? std::string("cleared") : lastClear}});
            if (! droppedPaused.empty())
                OpLog::emit(OpLog::Src::Ot, "OT_DROP", {{"ids", std::move(droppedPaused)}, {"reason", "paused, past"}});
            if (! reductionForModel.empty())
                OpLog::emit(OpLog::Src::Ot, "OT_RELEASE", {{"ids", otIds(reductionForModel)}, {"window_cs", holdBackWindowCs.load()}});
        }
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
        if (logging && (! midiUpdate.empty() || ! reductionForModel.empty())) {
            const auto userList = OpLog::Ints(userInstrumentList.begin(), userInstrumentList.end());
            const auto modelList = OpLog::Ints(modelInstrumentList.begin(), modelInstrumentList.end());
            OpLog::emit(OpLog::Src::Ot, "OT_CALL",
                        {{"mode", modeName(mode)},
                         {"user_inputs", mode == OrchestrationMode::Edit ? otIds(midiUpdate) : OpLog::Strings{}},
                         {"reduction_inputs", otIds(reductionForModel)},
                         {"user_instruments", userList}, {"model_instruments", modelList}});
        }
        if (mode == OrchestrationMode::Edit) {
            const ScopedMs modelMs(&step.model);
            stampInputVoices(midiUpdate); // the user's notes through vocsep too
            auto [userNotes, modelNotes] = getEditOrchestrationOutput(
                midiUpdate, reductionForModel, userInstrumentList, modelInstrumentList, signal);
            toOutput.reserve(userNotes.size() + modelNotes.size() + clearFromMidi.size()
                             + clearFromReduction.size());
            toOutput.insert(toOutput.end(), userNotes.begin(), userNotes.end());
            toOutput.insert(toOutput.end(), modelNotes.begin(), modelNotes.end());
        } else if (isJamLike(mode)) {
            const ScopedMs modelMs(&step.model);
            // Jam: the reduction plays the model instruments only (the user instruments are
            // for the user's own notes, in Edit). Accompaniment keeps both.
            std::set<int32_t> instrumentUnion(modelInstrumentList.begin(), modelInstrumentList.end());
            if (mode != OrchestrationMode::Jam)
                instrumentUnion.insert(userInstrumentList.begin(), userInstrumentList.end());
            const std::vector<int32_t> orchestrationInstruments(instrumentUnion.begin(),
                                                                instrumentUnion.end());
            toOutput = getJamOrchestrationOutput(reductionForModel, orchestrationInstruments,
                                                 signal);
            toOutput.reserve(toOutput.size() + clearFromMidi.size() + clearFromReduction.size());
        } else {
            juce::Logger::writeToLog("Invalid orchestration mode");
            step.total = static_cast<float>(
                juce::Time::getMillisecondCounterHiRes() - loopStartMs);
            lastThreadLoopMs.store(step.total, std::memory_order_relaxed);
            return;
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

    if (logging) {
        const double t = AppTime::hiResMs();
        if (OpLog::checkpointDue(OpLog::Src::Ot, t)) {
            OpLog::markCheckpoint(OpLog::Src::Ot, t);
            const auto [userList, modelList] = copyInstrumentLists();
            OpLog::Obj ot;
            ot.add("holdback", otIds(holdBack.notes())).add("paused", paused.get())
                .add("mode", std::string(modeName(getMode())))
                .add("user_instruments", OpLog::Ints(userList.begin(), userList.end()))
                .add("model_instruments", OpLog::Ints(modelList.begin(), modelList.end()));
            OpLog::emit(OpLog::Src::State, "CHECKPOINT", {{"part", "ot"}, {"ot", ot.raw()}});
        }
    }

    const bool busy = step.midiIn > 0 || step.reductionIn > 0 || step.getOutputCalls > 0
                      || ! conditioningUpdate.empty();
    if (busy)
        publishBusyOrchProfile(step);
}

void OrchestrationTransformer::threadStop() {
}
