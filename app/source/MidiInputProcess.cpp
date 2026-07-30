#include "VirtualOrch/MidiInputProcess.h"

#include <iostream>

MidiInputProcess::MidiInputProcess(Clock &clock,
                                   MusicTransformer &musicTransformer,
                                   ModelConfig &modelConfig,
                                   std::unique_ptr<OutputProcessor> &outputProcessor,
                                   std::unique_ptr<InputFilter> &inputFilter,
                                   juce::String &selectedMidiInputIdentifier,
                                   juce::String &selectedMidiInput2Identifier,
                                   juce::String &selectedMtcClockIdentifier,
                                   bool &mtcClockActive)
    : clock(clock),
      musicTransformer(musicTransformer),
      modelConfig(modelConfig),
      outputProcessor(outputProcessor),
      inputFilter(inputFilter),
      selectedMidiInputIdentifier(selectedMidiInputIdentifier),
      selectedMidiInput2Identifier(selectedMidiInput2Identifier),
      selectedMtcClockIdentifier(selectedMtcClockIdentifier),
      mtcClockActive(mtcClockActive) {
}

void MidiInputProcess::resetForStart() {
    tokensToSend.clear();
    notesOnToSend.clear();
    if (modelConfig.inputMode == InputMode::Direct && modelConfig.directInputHoldBass
        && modelConfig.directInputInitialBass != -1) {
        bassHeld = modelConfig.directInputInitialBass;
    } else {
        bassHeld = std::nullopt;
    }
}

void MidiInputProcess::sendTokensToMusicTransformer(const std::optional<int32_t> atTime) {
    for (auto &token: tokensToSend) {
        std::cout << "considering token: " << token.note << std::endl;
        if (atTime.has_value()) {
            token.time = atTime.value();
        }
        std::cout << "input token: " << token.toUnderstandableString() << std::endl;
        if (inputFilter != nullptr)
            inputFilter->filter(token);
    }
    tokensToSend.clear();
}

void MidiInputProcess::timerCallback() {
    if (modelConfig.inputMode != InputMode::Direct) {
        throw std::runtime_error("Timer should not be running in non-direct input mode.");
    }

    uint32_t time = clock.getTime();
    if ((!notesOnToSend.empty() || (modelConfig.directInputSendNoteOffs && !tokensToSend.empty()))
        && time - lastTokenAddedTime > modelConfig.directInputWindowLength) {
        if (modelConfig.directInputHoldBass && bassHeld.has_value()) {
            DBG("Holding bass " + std::to_string(bassHeld.value()));
            Token bassToken = {
                .time = static_cast<int32_t>(lastTokenAddedTime),
                .duration = static_cast<int32_t>(Vocab::DurOffset + modelConfig.inputDuration),
                .note = bassHeld.value()
            };
            tokensToSend.push_back(bassToken);
        }
        for (auto &note: notesOnToSend) {
            Token inputToken = {
                .time = static_cast<int32_t>(lastTokenAddedTime),
                .duration = static_cast<int32_t>(Vocab::DurOffset + modelConfig.inputDuration),
                .note = static_cast<int32_t>(Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument
                                             + note.first)
            };
            tokensToSend.push_back(inputToken);
        }

        if (modelConfig.directInputStartOnInput && musicTransformer.directInputBlock.get()) {
            sendTokensToMusicTransformer(lastTokenAddedTime + modelConfig.directInputStartDelay);
            musicTransformer.directInputBlock = false;
        } else {
            sendTokensToMusicTransformer(std::nullopt);
        }
        stopTimer();
    }
}

void MidiInputProcess::handleIncomingMidiMessage(juce::MidiInput *source, const juce::MidiMessage &message) {
    if (message.isController()) {
        std::cout << "midi in: cc " << message.getControllerNumber() << " value "
                  << message.getControllerValue() << std::endl;
    } else if (message.isNoteOn()) {
        std::cout << "midi in: note on " << message.getNoteNumber() << " vel "
                  << message.getVelocity() << std::endl;
    } else if (message.isNoteOff()) {
        std::cout << "midi in: note off " << message.getNoteNumber() << std::endl;
    }

    if (inputThruEnabled && source->getIdentifier() == selectedMidiInputIdentifier
        && outputProcessor != nullptr) {
        outputProcessor->relayMidi(message);
    }

    if (mtcClockActive && source->getIdentifier() == selectedMtcClockIdentifier && message.isQuarterFrame()) {
        const int value = message.getQuarterFrameValue();
        switch (message.getQuarterFrameSequenceNumber()) {
            case 0: frames = (frames & 0xf0) | value;
                break;
            case 1: frames = (frames & 0x0f) | (value << 4);
                break;
            case 2: seconds = (seconds & 0xf0) | value;
                break;
            case 3: seconds = (seconds & 0x0f) | (value << 4);
                break;
            case 4: minutes = (minutes & 0xf0) | value;
                break;
            case 5: minutes = (minutes & 0x0f) | (value << 4);
                break;
            case 6: hours = (hours & 0xf0) | value;
                break;
            case 7: hours = (hours & 0x0f) | ((value << 4) & 0x10);
                const uint32_t time = (seconds + (frames / 30.0)) * 100 + minutes * 6000;
                clock.setMtcTime(time);
                break;
        }
    } else if (message.isController() && source->getIdentifier() == selectedMidiInput2Identifier) {
        DBG("Controller: " + std::to_string(message.getControllerNumber()) + " - Value: " +
            std::to_string(message.getControllerValue()));
        handleContinuousControl(message.getControllerNumber(), message.getControllerValue());
    } else if (message.isNoteOn()) {
        DBG("Note On: " + std::to_string(message.getNoteNumber()) + " - Vel: " +
            std::to_string(message.getVelocity()));
        handleNoteOn(message.getNoteNumber(), message.getVelocity());
    } else if (message.isNoteOff()) {
        handleNoteOff(message.getNoteNumber());
    }
}

void MidiInputProcess::handleContinuousControl(int controllerNumber, int controllerValue) {
    if (!musicTransformer.isThreadRunning()) {
        std::cout << "input token dropped: thread not running (cc)" << std::endl;
        return;
    }
    juce::ignoreUnused(controllerNumber, controllerValue);
}

void MidiInputProcess::handleNoteOn(int midiNoteNumber, float velocity) {
    juce::ignoreUnused(velocity);

    if (!musicTransformer.isThreadRunning()) {
        std::cout << "input token dropped: thread not running (note on)" << std::endl;
        return;
    }

    if (midiNoteNumber < modelConfig.inputLow || midiNoteNumber > modelConfig.inputHigh) {
        std::cout << "input token dropped: note out of input range " << midiNoteNumber << std::endl;
        return;
    }

    uint32_t time = clock.getTime();
    Token inputToken = {
        static_cast<int32_t>(time), static_cast<int32_t>(Vocab::DurOffset + modelConfig.inputDuration),
        static_cast<int32_t>(Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument + midiNoteNumber)
    };

    if (modelConfig.inputMode == InputMode::Direct) {
        if (modelConfig.directInputHoldBass &&
            (inputToken.note >= Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument +
             modelConfig.directInputBassLow
             && inputToken.note < Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument +
             modelConfig.directInputBassHigh)) {
            bassHeld = inputToken.note;
            DBG("Setting bass to " + std::to_string(bassHeld.value()));
        } else {
            notesOnToSend[midiNoteNumber] = time;
        }
        lastTokenAddedTime = time;
        if (!isTimerRunning()) {
            std::cout << "starting timer: 10ms" << std::endl;
            startTimer(10);
        } else {
            std::cout << "timer not started" << std::endl;
        }
    } else if (modelConfig.inputMode == InputMode::Buffer) {
        if (inputToken.note >= Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument +
            modelConfig.bufferInputBassLow
            && inputToken.note < Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument +
            modelConfig.bufferInputBassHigh) {
            tokensToSend.push_front(inputToken);
            sendTokensToMusicTransformer(std::nullopt);
        } else if (inputToken.note >= Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument +
                   modelConfig.bufferInputLow
                   && inputToken.note < Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument +
                   modelConfig.bufferInputHigh) {
            tokensToSend.push_back(inputToken);
        }

        if (tokensToSend.size() > static_cast<size_t>(modelConfig.bufferInputSize)) {
            tokensToSend.pop_front();
        }
    }
}

void MidiInputProcess::handleNoteOff(int midiNoteNumber) {
    if (!musicTransformer.isThreadRunning()) {
        std::cout << "input token dropped: thread not running (note off)" << std::endl;
        return;
    }

    if (midiNoteNumber < modelConfig.inputLow || midiNoteNumber > modelConfig.inputHigh) {
        std::cout << "input token dropped: note off out of input range " << midiNoteNumber << std::endl;
        return;
    }

    uint32_t time = clock.getTime();

    if (modelConfig.inputMode == InputMode::Direct) {
        if (notesOnToSend.contains(midiNoteNumber)) {
            if (modelConfig.directInputSendNoteOffs) {
                const double onsetTime = notesOnToSend[midiNoteNumber];
                Token inputToken = {
                    static_cast<int32_t>(onsetTime), static_cast<int32_t>(Vocab::DurOffset + (time - onsetTime)),
                    static_cast<int32_t>(Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument +
                                         midiNoteNumber)
                };
                tokensToSend.push_back(inputToken);
            }
            notesOnToSend.erase(midiNoteNumber);
        }
    }
}
