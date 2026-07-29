#include "VirtualOrch/OutputPlayback.h"

#include <set>

OutputPlayback::OutputPlayback(Clock &clock,
                               MusicTransformer &musicTransformer,
                               std::unique_ptr<OutputProcessor> &outputProcessor,
                               std::unique_ptr<OSCBufferOutputProcessor> &bufferOutputProcessor,
                               int32_t &visualizationBufferSize)
    : Thread("Output Playback"),
      clock(clock),
      musicTransformer(musicTransformer),
      outputProcessor(outputProcessor),
      bufferOutputProcessor(bufferOutputProcessor),
      visualizationBufferSize(visualizationBufferSize) {
}

void OutputPlayback::handleNoteForOutput(Token token, TokenNoteType tokenEventType) {
    if (outputProcessor == nullptr) {
        return;
    }

    int32_t instrument = (token.note - Vocab::NoteOffset) / Config::MaxPitch;
    int32_t pitch = (token.note - Vocab::NoteOffset) % Config::MaxPitch;
    switch (tokenEventType) {
        case TokenNoteOn: {
            NoteOnEvent noteOnEvent = {.instrument = instrument, .note = pitch, .velocity = 1.0F};
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
    std::multiset<Token> nextTokens;

    auto compareDuration = [](Token a, Token b) {
        return a.time + a.getRealDuration() < b.time + b.getRealDuration();
    };
    std::multiset<Token, decltype(compareDuration)> currentlyPlaying;

    while (!threadShouldExit()) {
        bool clearedQueueThisRun = false;
        auto time = clock.getTime();
        progress = (static_cast<double>(musicTransformer.getCurrentTime()) - static_cast<double>(time)) / 1000.0;

        Token token = {-1, -1, -1};
        while (musicTransformer.outputTokenQueue.pull(token)) {
            if (token.note == Vocab::Rest) {
                continue;
            }

            if (token.note == Vocab::ClearQueue) {
                clearedQueueThisRun = true;
                DBG("Clearing queue!");
                while (!nextTokens.empty() && nextTokens.begin()->time >= token.time) {
                    nextTokens.erase(nextTokens.begin());
                }
                continue;
            }

            nextTokens.insert(token);
        }

        for (auto it = nextTokens.begin(); it != nextTokens.end();) {
            if (it->time <= time) {
                DBG("Playing note at " + std::to_string(time) + " : " + it->toUnderstandableString());
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
                if (it->time <= time + visualizationBufferSize) {
                    bufferOutputProcessor->addToBuffer(*it);
                } else {
                    break;
                }
            }
            bufferOutputProcessor->sendBuffer(time);
        }

        for (auto it = currentlyPlaying.begin(); it != currentlyPlaying.end();) {
            if ((it->time + it->getRealDuration()) <= time) {
                handleNoteForOutput(*it, TokenNoteType::TokenNoteOff);
                it = currentlyPlaying.erase(it);
            } else {
                break;
            }
        }
    }
}
