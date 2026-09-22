#include "VirtualOrch/OutputPlayback.h"

#include "VirtualOrch/InstrumentConstants.h"
#include "VirtualOrch/NoteWindow.h"

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

void OutputPlayback::handleNoteForOutput(const OrchestrationNote &note, TokenNoteType tokenEventType) {
    if (outputProcessor == nullptr) {
        return;
    }

    const int32_t instrument = note.localInstrumentId;
    const int32_t pitch = note.token.getPitch();
    switch (tokenEventType) {
        case TokenNoteOn: {
            const float midiVelocity =
                note.velocity > 0 ? static_cast<float>(note.velocity) / 127.0f : 100.0f / 127.0f;
            NoteOnEvent noteOnEvent = {
                .instrument = instrument, .note = pitch, .velocity = midiVelocity};
            outputProcessor->send(noteOnEvent);
            if (const auto channel = outputProcessor->channelForInstrument(instrument, pitch))
                recordNoteOn(note, *channel);
            break;
        }
        case TokenNoteOff: {
            NoteOffEvent noteOffEvent = {.instrument = instrument, .note = pitch};
            outputProcessor->send(noteOffEvent);
            break;
        }
    }
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
        if (record.onsetCs == oldNote.time && record.pitch == pitch && record.durationCs == oldDur)
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

void OutputPlayback::run() {
    clearNoteOnHistory();

    std::multiset<OrchestrationNote, decltype(&orchestrationNoteOnsetLess)> nextTokens(
        orchestrationNoteOnsetLess);

    std::multiset<OrchestrationNote, decltype(&orchestrationNoteEndLess)> currentlyPlaying(
        orchestrationNoteEndLess);

    // Unmatched duration updates retry until the note appears or its default end has passed.
    std::vector<TokenUpdate> pendingDurationUpdates;
    int32_t playbackTime = 0;
    auto lastSource = playbackSource.load(std::memory_order_relaxed);

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
    };

    auto applyOneDurationUpdate = [&](const TokenUpdate &durationUpdate) -> int {
        int matched = 0;
        for (auto it = nextTokens.begin(); it != nextTokens.end();) {
            if (! tokenEqualsByOnsetPitch(it->token, durationUpdate.oldNote)) {
                ++it;
                continue;
            }
            ++matched;
            OrchestrationNote revised = *it;
            revised.token.duration = durationUpdate.newNote.duration;
            revised.token.velocity = durationUpdate.newNote.velocity;
            it = nextTokens.erase(it);
            nextTokens.insert(revised);
        }

        for (auto it = currentlyPlaying.begin(); it != currentlyPlaying.end();) {
            if (! tokenEqualsByOnsetPitch(it->token, durationUpdate.oldNote)) {
                ++it;
                continue;
            }
            ++matched;
            OrchestrationNote revised = *it;
            revised.token.duration = durationUpdate.newNote.duration;
            revised.token.velocity = durationUpdate.newNote.velocity;
            it = currentlyPlaying.erase(it);

            if (revised.token.time + revised.token.getRealDuration() <= playbackTime)
                handleNoteForOutput(revised, TokenNoteType::TokenNoteOff);
            else
                currentlyPlaying.insert(revised);
        }
        return matched;
    };

    while (! threadShouldExit()) {
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

        const auto source = playbackSource.load(std::memory_order_relaxed);
        if (source != lastSource) {
            flushModelSchedule();
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
            if (source != PlaybackSource::Orchestration)
                continue;
            if (isRestToken(note.token))
                continue;
            if (isClearQueueToken(note.token)) {
                clearedQueueThisRun = true;
                DBG("Clearing queue at time " + juce::String(note.token.time));
                applyClearQueue(nextTokens, note.token.time);
                continue;
            }
            scheduleNote(note);
        }

        // Always drain RT outputTokenQueue (not OT reductionIncoming).
        if (reductionOutputQueue != nullptr) {
            Token rtToken{};
            while (reductionOutputQueue->pull(rtToken)) {
                if (source != PlaybackSource::Reduction)
                    continue;
                if (isRestToken(rtToken))
                    continue;
                if (isClearQueueToken(rtToken)) {
                    clearedQueueThisRun = true;
                    DBG("Clearing RT monitor queue at time " + juce::String(rtToken.time));
                    applyClearQueue(nextTokens, rtToken.time);
                    continue;
                }
                if (! isSoundingNoteToken(rtToken))
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

        for (auto it = nextTokens.begin(); it != nextTokens.end();) {
            if (it->token.time <= time) {
                handleNoteForOutput(*it, TokenNoteType::TokenNoteOn);
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
    }
}
