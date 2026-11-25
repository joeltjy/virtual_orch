#include "VirtualOrch/OutputPlayback.h"
#include "VirtualOrch/AppTime.h"
#include "VirtualOrch/OpLog.h"
#include "VirtualOrch/Sympathetic.h"
#include "VirtualOrch/PerformanceActivity.h"

#include "VirtualOrch/InstrumentConstants.h"
#include "VirtualOrch/NoteWindow.h"
#include "VirtualOrch/PedalClear.h"

#include <set>

namespace {

auto orchestrationNoteOnsetLess(const OrchestrationNote &a, const OrchestrationNote &b) -> bool {
    if (a.token.time != b.token.time)
        return a.token.time < b.token.time;
    if (a.token.duration != b.token.duration)
        return a.token.duration < b.token.duration;
    if (a.localInstrumentId != b.localInstrumentId)
        return a.localInstrumentId < b.localInstrumentId;
    return a.token.note < b.token.note;
}

auto orchestrationNoteEndLess(const OrchestrationNote &a, const OrchestrationNote &b) -> bool {
    const auto aEnd = a.token.time + a.token.getRealDuration();
    const auto bEnd = b.token.time + b.token.getRealDuration();
    if (aEnd != bEnd)
        return aEnd < bEnd;
    return orchestrationNoteOnsetLess(a, b);
}

[[nodiscard]] auto isClearQueueToken(const Token &token) -> bool {
    return token.note == static_cast<int32_t>(Vocab::ClearQueue);
}

[[nodiscard]] auto isRestToken(const Token &token) -> bool {
    return token.note == static_cast<int32_t>(Vocab::Rest);
}

[[nodiscard]] auto isSoundingNoteToken(const Token &token) -> bool {
    return token.note != static_cast<int32_t>(Vocab::BarSeparator)
           && token.note >= static_cast<int32_t>(Vocab::NoteOffset)
           && token.note < static_cast<int32_t>(Vocab::Rest)
           && ! isRestToken(token);
}

[[nodiscard]] auto reductionMonitorNote(const Token &token) -> OrchestrationNote {
    auto scheduled = token;
    // Avoid instant note-on/note-off in the same tick (inaudible in many hosts).
    if (scheduled.getRealDuration() <= 0)
        scheduled.duration = static_cast<int32_t>(Vocab::DurOffset + 1);
    return OrchestrationNote{
        .token = scheduled,
        .velocity = token.velocity > 0 ? token.velocity : 100,
        .localInstrumentId = InstrumentConstants::kReductionPlaybackLocalId,
    };
}

/** Operation log id of a scheduled note: its OT output (o) or, when monitoring the RT, the RT note (r). */
auto playId(const OrchestrationNote &note) -> std::string {
    return note.outputId >= 0 ? OpLog::id('o', note.outputId) : OpLog::id('r', note.token.noteId);
}

auto applyClearQueue(std::multiset<OrchestrationNote, decltype(&orchestrationNoteOnsetLess)> &nextTokens,
                     int32_t clearTime) -> void {
    for (auto it = nextTokens.begin(); it != nextTokens.end();) {
        if (it->token.time >= clearTime)
            it = nextTokens.erase(it);
        else
            ++it;
    }
}

} // namespace

OutputPlayback::OutputPlayback(Clock &clock,
                               OrchestrationTransformer &orchestrationTransformer,
                               std::unique_ptr<OutputProcessor> &outputProcessor,
                               std::unique_ptr<OSCBufferOutputProcessor> &bufferOutputProcessor,
                               int32_t &visualizationBufferSize,
                               std::atomic<PlaybackSource> &playbackSourceIn)
    : Thread("Output Playback"),
      clock(clock),
      orchestrationTransformer(orchestrationTransformer),
      outputProcessor(outputProcessor),
      bufferOutputProcessor(bufferOutputProcessor),
      visualizationBufferSize(visualizationBufferSize),
      playbackSource(playbackSourceIn) {
}

bool OutputPlayback::handleNoteForOutput(const OrchestrationNote &note, TokenNoteType tokenEventType,
                                         bool reductionMonitor) {
    if (outputProcessor == nullptr)
        return false;

    const int32_t instrument = note.localInstrumentId;
    const int32_t pitch = note.token.getPitch();
    // Sympathetic: uncued parts stay silent (safety net behind the model mask, e.g. a
    // string-ensemble note routed to an uncued solo string); Select/Expression set the
    // velocity. Both are no-ops while the gesture layer is off.
    auto played = note;
    if (tokenEventType == TokenNoteOn) {
        const auto &sym = Sympathetic::state();
        if (! sym.maySound(instrument, pitch, reductionMonitor))
            return false;
        played.velocity = sym.playbackVelocity(note.velocity > 0 ? note.velocity : 100, instrument, pitch);
    }
    if (OpLog::on()) {
        const auto channel = outputProcessor->channelForInstrument(instrument, pitch);
        OpLog::emit(OpLog::Src::Play, "MIDI_OUT",
                    {{"kind", tokenEventType == TokenNoteOn ? "on" : "off"}, {"o", playId(note)},
                     {"instrument", instrument},
                     {"channel", channel.has_value() ? OpLog::Value(static_cast<int64_t>(*channel)) : OpLog::Value()},
                     {"pitch", pitch},
                     {"vel", tokenEventType == TokenNoteOn ? OpLog::Value(static_cast<int64_t>(played.velocity)) : OpLog::Value()}});
    }
    switch (tokenEventType) {
        case TokenNoteOn: {
            const float midiVelocity = static_cast<float>(played.velocity) / 127.0f;
            NoteOnEvent noteOnEvent = {
                .instrument = instrument, .note = pitch, .velocity = midiVelocity};
            outputProcessor->send(noteOnEvent);
            if (const auto channel = outputProcessor->channelForInstrument(instrument, pitch))
                recordNoteOn(played, *channel);
            break;
        }
        case TokenNoteOff: {
            NoteOffEvent noteOffEvent = {.instrument = instrument, .note = pitch};
            outputProcessor->send(noteOffEvent);
            break;
        }
    }
    return true;
}

auto OutputPlayback::recordNoteOn(const OrchestrationNote &note, int32_t channel) -> void {
    PlaybackNoteOnRecord record;
    record.time = clock.getTime();
    record.onsetCs = note.token.time;
    record.durationCs = note.token.getRealDuration();
    record.pitch = note.token.getPitch();
    record.velocity = note.velocity > 0 ? note.velocity : 100;
    record.localInstrumentId = note.localInstrumentId;
    record.channel = channel;
    record.keyId = note.token.keyId;
    if (activityLog != nullptr)
        activityLog->noteOn(AppTime::nowMs(), record.localInstrumentId,
                            record.pitch, static_cast<int64_t>(record.durationCs) * 10);
    const juce::ScopedLock lock(noteOnHistoryLock);
    noteOnHistory.push_back(record);
    NoteWindow::trimToLastNotes(noteOnHistory);
}

auto OutputPlayback::updateNoteOnHistoryDuration(const Token &oldNote, const Token &newNote) -> void {
    if (oldNote.note < static_cast<int32_t>(Vocab::NoteOffset)
        || oldNote.note >= static_cast<int32_t>(Vocab::Rest))
        return;

    const auto oldDur = oldNote.getRealDuration();
    const auto newDur = newNote.getRealDuration();
    const auto pitch = oldNote.getPitch();
    const juce::ScopedLock lock(noteOnHistoryLock);
    for (auto &record: noteOnHistory) {
        const bool matches = oldNote.keyId >= 0
                                 ? record.keyId == oldNote.keyId
                                 : record.onsetCs == oldNote.time && record.pitch == pitch
                                       && record.durationCs == oldDur;
        if (matches)
            record.durationCs = newDur;
    }
}

auto OutputPlayback::getNoteOnHistory() const -> std::vector<PlaybackNoteOnRecord> {
    const juce::ScopedLock lock(noteOnHistoryLock);
    return noteOnHistory;
}

auto OutputPlayback::getNoteOnHistorySince(int32_t cutoffCs) const
    -> std::vector<PlaybackNoteOnRecord> {
    const juce::ScopedLock lock(noteOnHistoryLock);
    return NoteWindow::filtered(noteOnHistory, cutoffCs, [](const PlaybackNoteOnRecord &record) {
        return std::pair<int32_t, int32_t>{record.onsetCs, record.durationCs};
    });
}

auto OutputPlayback::clearNoteOnHistory() -> void {
    const juce::ScopedLock lock(noteOnHistoryLock);
    noteOnHistory.clear();
}

/** Loop state of the playback thread (one per run, or per simulator session). */
struct OutputPlayback::StepState {
    std::multiset<OrchestrationNote, decltype(&orchestrationNoteOnsetLess)> nextTokens{orchestrationNoteOnsetLess};
    std::multiset<OrchestrationNote, decltype(&orchestrationNoteEndLess)> currentlyPlaying{orchestrationNoteEndLess};
    // Unmatched duration updates retry until the note appears or its default end has passed.
    std::vector<TokenUpdate> pendingDurationUpdates;
    int32_t playbackTime = 0;
    PlaybackSource lastSource = PlaybackSource::Orchestration;
    PedalClear::DropWindow pedalDrop;
};

OutputPlayback::~OutputPlayback() = default;

void OutputPlayback::run() {
    beginSteps();
    while (! threadShouldExit())
        stepOnce();
}

auto OutputPlayback::beginSteps() -> void {
    clearNoteOnHistory();
    state = std::make_unique<StepState>();
    state->lastSource = playbackSource.load(std::memory_order_relaxed);
}

auto OutputPlayback::stepOnce() -> void {
    if (state == nullptr)
        beginSteps();
    auto &nextTokens = state->nextTokens;
    auto &currentlyPlaying = state->currentlyPlaying;
    auto &pendingDurationUpdates = state->pendingDurationUpdates;
    auto &playbackTime = state->playbackTime;
    auto &lastSource = state->lastSource;
    auto &pedalDrop = state->pedalDrop;

    const bool logging = OpLog::on();
    // Operation log: notes dropped from the schedule by `op` (compares the schedule before / after).
    auto scheduled = [&]() {
        std::vector<std::string> ids;
        if (logging)
            for (const auto &n: nextTokens)
                ids.push_back(playId(n));
        return ids;
    };
    auto logDrops = [&](const std::vector<std::string> &before, const std::string &reason) {
        if (! logging || before.empty())
            return;
        std::multiset<std::string> after;
        for (const auto &n: nextTokens)
            after.insert(playId(n));
        for (const auto &id: before) {
            const auto it = after.find(id);
            if (it != after.end())
                after.erase(it);
            else
                OpLog::emit(OpLog::Src::Play, "PLAY_DROP", {{"o", id}, {"reason", reason}});
        }
    };

    auto flushModelSchedule = [&]() {
        for (const auto &playing: currentlyPlaying)
            handleNoteForOutput(playing, TokenNoteType::TokenNoteOff);
        currentlyPlaying.clear();
        nextTokens.clear();
        pendingDurationUpdates.clear();
    };

    auto scheduleNote = [&](const OrchestrationNote &note) {
        if (outputProcessor != nullptr)
            outputProcessor->ensureInstrument(note.localInstrumentId);
        nextTokens.insert(note);
        if (logging)
            OpLog::emit(OpLog::Src::Play, "PLAY_SCHEDULE", {{"o", playId(note)}, {"onset", note.token.time}});
    };

    // Keyboard notes are matched by keyId and keep their onset (chord snapping) and their
    // instrument-scaled velocity: only the duration changes.
    auto revise = [](OrchestrationNote note, const TokenUpdate &update) {
        note.token.duration = update.newNote.duration;
        if (update.oldNote.keyId < 0)
            note.token.velocity = update.newNote.velocity;
        return note;
    };

    auto applyOneDurationUpdate = [&](const TokenUpdate &durationUpdate) -> int {
        int matched = 0;
        for (auto it = nextTokens.begin(); it != nextTokens.end();) {
            if (! tokenMatchesUpdateSource(it->token, durationUpdate.oldNote)) {
                ++it;
                continue;
            }
            ++matched;
            const OrchestrationNote revised = revise(*it, durationUpdate);
            if (logging)
                OpLog::emit(OpLog::Src::Play, "PLAY_LENGTH", {{"o", playId(revised)}, {"dur", revised.token.getRealDuration()}});
            it = nextTokens.erase(it);
            nextTokens.insert(revised);
        }

        for (auto it = currentlyPlaying.begin(); it != currentlyPlaying.end();) {
            if (! tokenMatchesUpdateSource(it->token, durationUpdate.oldNote)) {
                ++it;
                continue;
            }
            ++matched;
            const OrchestrationNote revised = revise(*it, durationUpdate);
            if (logging)
                OpLog::emit(OpLog::Src::Play, "PLAY_LENGTH", {{"o", playId(revised)}, {"dur", revised.token.getRealDuration()}, {"sounding", true}});
            it = currentlyPlaying.erase(it);

            if (revised.token.time + revised.token.getRealDuration() <= playbackTime)
                handleNoteForOutput(revised, TokenNoteType::TokenNoteOff);
            else
                currentlyPlaying.insert(revised);
        }
        return matched;
    };


    bool clearedQueueThisRun = false;
    playbackTime = clock.getTime();
    const auto time = playbackTime;
    const bool reductionPaused = isReductionPaused && isReductionPaused();
    if (orchestrationTransformer.paused.get() || reductionPaused) {
        progress = 0.0;
    } else {
        progress = (static_cast<double>(orchestrationTransformer.getCurrentOTTime())
                    - static_cast<double>(time))
                   / 1000.0;
    }

    {
        // Sustain pedal: drop every note that has not sounded, end sounding notes by
        // pedal time + 0.3 s, then keep dropping late arrivals until the upstream clear.
        int32_t pedalTime = 0;
        while (pedalClearRequests.pull(pedalTime)) {
            const auto before = scheduled();
            PedalClear::apply(nextTokens, currentlyPlaying, pedalTime, time,
                              [&](const OrchestrationNote &note) {
                                  handleNoteForOutput(note, TokenNoteType::TokenNoteOff);
                              });
            pendingDurationUpdates.clear();
            logDrops(before, "pedal c" + std::to_string(pedalTime));
            pedalDrop.start(pedalTime, time);
            clearedQueueThisRun = true;
            DBG("Pedal clear at time " + juce::String(pedalTime));
        }

        // Leaving Accompaniment: drop every note that has not started; notes already
        // sounding play to their end (their note-off updates still apply).
        int32_t cutTime = 0;
        while (modeCutRequests.pull(cutTime)) {
            const auto before = scheduled();
            nextTokens.clear();
            logDrops(before, "mode cut c" + std::to_string(cutTime));
            pedalDrop.start(cutTime, time);
            clearedQueueThisRun = true;
            DBG("Mode cut at time " + juce::String(cutTime));
        }
    }

    const auto source = playbackSource.load(std::memory_order_relaxed);
    if (source != lastSource) {
        const auto before = scheduled();
        flushModelSchedule();
        logDrops(before, "playback source changed");
        lastSource = source;
        // While Out: OT is selected, RT FIFO tokens are drained and discarded. Seed from
        // outputHistory so Out: RT is not silent until the ahead-throttle window ends.
        if (source == PlaybackSource::Reduction && getReductionOutputHistory) {
            for (const auto &histToken: getReductionOutputHistory()) {
                if (! isSoundingNoteToken(histToken))
                    continue;
                if (histToken.time + histToken.getRealDuration() <= time)
                    continue;
                scheduleNote(reductionMonitorNote(histToken));
            }
        }
    }

    // Always drain OT output (schedule only when Orchestration is selected).
    OrchestrationNote note{};
    while (orchestrationTransformer.outputTokenQueue.pull(note)) {
        if (logging)
            OpLog::emit(OpLog::Src::Play, "Q_POP",
                        {{"queue", "ot→play"},
                         {"item", isClearQueueToken(note.token) ? "CLEAR c" + std::to_string(note.token.time) : playId(note)}});
        if (source != PlaybackSource::Orchestration)
            continue;
        if (isRestToken(note.token))
            continue;
        if (isClearQueueToken(note.token)) {
            clearedQueueThisRun = true;
            DBG("Clearing queue at time " + juce::String(note.token.time));
            const auto before = scheduled();
            applyClearQueue(nextTokens, note.token.time);
            logDrops(before, "Clear Queue c" + std::to_string(note.token.time));
            if (PedalClear::isPedalClear(note.token))
                pedalDrop.finish();
            continue;
        }
        if (pedalDrop.drops(note.token.time, time)) {
            if (logging)
                OpLog::emit(OpLog::Src::Play, "PLAY_DROP", {{"o", playId(note)}, {"reason", "pedal (late arrival)"}});
            continue;
        }
        scheduleNote(note);
    }

    // Always drain RT outputTokenQueue (not OT reductionIncoming).
    if (reductionOutputQueue != nullptr) {
        Token rtToken{};
        while (reductionOutputQueue->pull(rtToken)) {
            if (logging)
                OpLog::emit(OpLog::Src::Play, "Q_POP",
                            {{"queue", "rt→play"},
                             {"item", isClearQueueToken(rtToken) ? "CLEAR c" + std::to_string(rtToken.time) : OpLog::id('r', rtToken.noteId)}});
            if (source != PlaybackSource::Reduction)
                continue;
            if (isRestToken(rtToken))
                continue;
            if (isClearQueueToken(rtToken)) {
                clearedQueueThisRun = true;
                DBG("Clearing RT monitor queue at time " + juce::String(rtToken.time));
                const auto before = scheduled();
                applyClearQueue(nextTokens, rtToken.time);
                logDrops(before, "Clear Queue c" + std::to_string(rtToken.time));
                if (PedalClear::isPedalClear(rtToken))
                    pedalDrop.finish();
                continue;
            }
            if (! isSoundingNoteToken(rtToken))
                continue;
            if (pedalDrop.drops(rtToken.time, time))
                continue;

            scheduleNote(reductionMonitorNote(rtToken));
        }
    }

    TokenUpdate durationUpdate{};
    while (orchestrationTransformer.outputDurationUpdates.pull(durationUpdate)) {
        if (source == PlaybackSource::Orchestration)
            pendingDurationUpdates.push_back(durationUpdate);
    }

    if (source == PlaybackSource::Orchestration) {
        for (auto it = pendingDurationUpdates.begin(); it != pendingDurationUpdates.end();) {
            const int matched = applyOneDurationUpdate(*it);

            const auto defaultEnd = it->oldNote.time + it->oldNote.getRealDuration();
            if (matched > 0) {
                updateNoteOnHistoryDuration(it->oldNote, it->newNote);
                it = pendingDurationUpdates.erase(it);
            } else if (time > defaultEnd) {
                it = pendingDurationUpdates.erase(it);
            } else {
                ++it;
            }
        }
    }

    const bool reductionMonitor = lastSource == PlaybackSource::Reduction;
    const auto tangibleMask = Sympathetic::state().tangibleMask.load(std::memory_order_relaxed);
    if (! reductionMonitor && tangibleMask != Sympathetic::kTangibleInactive) {
        for (auto it = currentlyPlaying.begin(); it != currentlyPlaying.end();) {
            if (Sympathetic::State::tangibleAllows(tangibleMask, it->localInstrumentId, it->token.getPitch())) {
                ++it;
            } else {
                handleNoteForOutput(*it, TokenNoteType::TokenNoteOff);
                it = currentlyPlaying.erase(it);
            }
        }
    }

    for (auto it = nextTokens.begin(); it != nextTokens.end();) {
        if (it->token.time <= time) {
            if (handleNoteForOutput(*it, TokenNoteType::TokenNoteOn, reductionMonitor))
                currentlyPlaying.insert(*it);
            it = nextTokens.erase(it);
        } else {
            break;
        }
    }

    if (bufferOutputProcessor != nullptr) {
        bufferOutputProcessor->setTime(time);
        if (clearedQueueThisRun) {
            Token clearQueue{static_cast<int32_t>(time), Vocab::DurOffset, Vocab::ClearQueue};
            bufferOutputProcessor->addToBuffer(clearQueue);
        }
        for (auto it = nextTokens.begin(); it != nextTokens.end(); ++it) {
            if (it->token.time <= time + visualizationBufferSize) {
                bufferOutputProcessor->addToBuffer(it->token);
            } else {
                break;
            }
        }
        bufferOutputProcessor->sendBuffer(time);
    }

    for (auto it = currentlyPlaying.begin(); it != currentlyPlaying.end();) {
        if ((it->token.time + it->token.getRealDuration()) <= time) {
            handleNoteForOutput(*it, TokenNoteType::TokenNoteOff);
            it = currentlyPlaying.erase(it);
        } else {
            break;
        }
    }

    if (logging) {
        const double t = AppTime::hiResMs();
        if (OpLog::checkpointDue(OpLog::Src::Play, t)) {
            OpLog::markCheckpoint(OpLog::Src::Play, t);
            OpLog::Strings waiting, sounding;
            for (const auto &n: nextTokens)
                waiting.push_back(playId(n));
            for (const auto &n: currentlyPlaying)
                sounding.push_back(playId(n));
            OpLog::Obj play;
            play.add("scheduled", std::move(waiting)).add("playing", std::move(sounding))
                .add("source", std::string(source == PlaybackSource::Orchestration ? "orchestration" : "reduction"))
                .add("pedal_drop_active", pedalDrop.active);
            OpLog::emit(OpLog::Src::State, "CHECKPOINT", {{"part", "play"}, {"play", play.raw()}});
        }
    }
}
