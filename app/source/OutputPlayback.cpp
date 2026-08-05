#include "VirtualOrch/OutputPlayback.h"

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

} // namespace

OutputPlayback::OutputPlayback(Clock &clock,
                               OrchestrationTransformer &orchestrationTransformer,
                               std::unique_ptr<OutputProcessor> &outputProcessor,
                               std::unique_ptr<OSCBufferOutputProcessor> &bufferOutputProcessor,
                               int32_t &visualizationBufferSize)
    : Thread("Output Playback"),
      clock(clock),
      orchestrationTransformer(orchestrationTransformer),
      outputProcessor(outputProcessor),
      bufferOutputProcessor(bufferOutputProcessor),
      visualizationBufferSize(visualizationBufferSize) {
}

void OutputPlayback::handleNoteForOutput(const OrchestrationNote &note, TokenNoteType tokenEventType) {
    if (outputProcessor == nullptr) {
        return;
    }

    const int32_t instrument = note.localInstrumentId;
    const int32_t pitch = note.token.getPitch();
    switch (tokenEventType) {
        case TokenNoteOn: {
            const float velocity =
                note.velocity > 0 ? static_cast<float>(note.velocity) / 127.0f : 1.0f;
            NoteOnEvent noteOnEvent = {.instrument = instrument, .note = pitch, .velocity = velocity};
            outputProcessor->send(noteOnEvent);
            break;
        }
        case TokenNoteOff: {
            NoteOffEvent noteOffEvent = {.instrument = instrument, .note = pitch};
            outputProcessor->send(noteOffEvent);
            break;
        }
    }
}

void OutputPlayback::run() {
    std::multiset<OrchestrationNote, decltype(&orchestrationNoteOnsetLess)> nextTokens(
        orchestrationNoteOnsetLess);

    std::multiset<OrchestrationNote, decltype(&orchestrationNoteEndLess)> currentlyPlaying(
        orchestrationNoteEndLess);

    while (!threadShouldExit()) {
        bool clearedQueueThisRun = false;
        auto time = clock.getTime();
        progress = (static_cast<double>(orchestrationTransformer.getCurrentOTTime())
                    - static_cast<double>(time))
                   / 1000.0;

        OrchestrationNote note{};
        while (orchestrationTransformer.outputTokenQueue.pull(note)) {
            if (note.token.note == Vocab::Rest) {
                continue;
            }

            if (note.token.note == Vocab::ClearQueue) {
                clearedQueueThisRun = true;
                DBG("Clearing queue!");
                while (!nextTokens.empty() && nextTokens.begin()->token.time >= note.token.time) {
                    nextTokens.erase(nextTokens.begin());
                }
                continue;
            }

            if (outputProcessor != nullptr)
                outputProcessor->ensureInstrument(note.localInstrumentId);

            nextTokens.insert(note);
        }

        // Apply after ingesting new notes so same-loop duration corrections still match.
        TokenUpdate durationUpdate{};
        while (orchestrationTransformer.outputDurationUpdates.pull(durationUpdate)) {
            for (auto it = nextTokens.begin(); it != nextTokens.end();) {
                if (! tokenEquals(it->token, durationUpdate.oldNote)) {
                    ++it;
                    continue;
                }
                OrchestrationNote revised = *it;
                revised.token = durationUpdate.newNote;
                it = nextTokens.erase(it);
                nextTokens.insert(revised);
            }

            for (auto it = currentlyPlaying.begin(); it != currentlyPlaying.end();) {
                if (! tokenEquals(it->token, durationUpdate.oldNote)) {
                    ++it;
                    continue;
                }
                OrchestrationNote revised = *it;
                revised.token = durationUpdate.newNote;
                it = currentlyPlaying.erase(it);

                if (revised.token.time + revised.token.getRealDuration() <= time)
                    handleNoteForOutput(revised, TokenNoteType::TokenNoteOff);
                else
                    currentlyPlaying.insert(revised);
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
