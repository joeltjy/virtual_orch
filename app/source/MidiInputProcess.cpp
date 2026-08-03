#include "VirtualOrch/MidiInputProcess.h"

#include <algorithm>
#include <iostream>

MidiInputProcess::MidiInputProcess(Clock &clock,
                                   MusicTransformer &musicTransformer,
                                   ModelConfig &modelConfig,
                                   std::unique_ptr<OutputProcessor> &outputProcessor,
                                   std::unique_ptr<InputFilter> &inputFilter,
                                   juce::String &selectedMidiInputIdentifier,
                                   juce::String &selectedLaunchpadMidiIdentifier,
                                   juce::String &selectedMtcClockIdentifier,
                                   bool &mtcClockActive)
    : clock(clock),
      musicTransformer(musicTransformer),
      modelConfig(modelConfig),
      outputProcessor(outputProcessor),
      inputFilter(inputFilter),
      selectedMidiInputIdentifier(selectedMidiInputIdentifier),
      selectedLaunchpadMidiIdentifier(selectedLaunchpadMidiIdentifier),
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

auto MidiInputProcess::hasUnflushedDirectNotes() const -> bool {
    for (const auto &[pitch, note]: notesOnToSend) {
        juce::ignoreUnused(pitch);
        if (! note.flushedTime.has_value())
            return true;
    }
    return false;
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
    if ((! hasUnflushedDirectNotes() && ! (modelConfig.directInputSendNoteOffs && ! tokensToSend.empty()))
        || time - lastTokenAddedTime <= modelConfig.directInputWindowLength) {
        return;
    }

    std::optional<int32_t> atTime;
    if (modelConfig.directInputStartOnInput && musicTransformer.directInputBlock.get()) {
        atTime = static_cast<int32_t>(lastTokenAddedTime + modelConfig.directInputStartDelay);
        musicTransformer.directInputBlock = false;
    }
    const int32_t stamp = atTime.value_or(static_cast<int32_t>(lastTokenAddedTime));
    const int32_t defaultDur = static_cast<int32_t>(Vocab::DurOffset + modelConfig.inputDuration);

    if (modelConfig.directInputHoldBass && bassHeld.has_value()) {
        DBG("Holding bass " + std::to_string(bassHeld.value()));
        tokensToSend.push_back({
            .time = stamp,
            .duration = defaultDur,
            .note = bassHeld.value()
        });
    }

    for (auto &[pitch, note]: notesOnToSend) {
        if (note.flushedTime.has_value())
            continue;
        tokensToSend.push_back({
            .time = stamp,
            .duration = defaultDur,
            .note = static_cast<int32_t>(Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument
                                         + pitch)
        });
        note.flushedTime = stamp;
    }

    sendTokensToMusicTransformer(std::nullopt);
    stopTimer();
}

void MidiInputProcess::logIncomingMidiMessage(juce::MidiInput *source, const juce::MidiMessage &message) {
    const bool fromLaunchpad = source != nullptr
                               && selectedLaunchpadMidiIdentifier.isNotEmpty()
                               && source->getIdentifier() == selectedLaunchpadMidiIdentifier;
    const char *tag = fromLaunchpad ? "[launchpad]" : "[midi]";

    if (message.isNoteOn(true)) {
        const auto vel = message.getVelocity();
        if (vel <= 0) {
            std::cout << tag << " note off (vel 0) note=" << message.getNoteNumber()
                      << " ch=" << message.getChannel() << std::endl;
        } else {
            std::cout << tag << " note on note=" << message.getNoteNumber()
                      << " vel=" << vel << " ch=" << message.getChannel() << std::endl;
        }
    } else if (message.isNoteOff()) {
        std::cout << tag << " note off note=" << message.getNoteNumber()
                  << " ch=" << message.getChannel() << std::endl;
    } else if (message.isController()) {
        std::cout << tag << " cc num=" << message.getControllerNumber()
                  << " value=" << message.getControllerValue()
                  << " ch=" << message.getChannel() << std::endl;
    } else if (message.isProgramChange()) {
        std::cout << tag << " program change=" << message.getProgramChangeNumber()
                  << " ch=" << message.getChannel() << std::endl;
    } else if (message.isPitchWheel()) {
        std::cout << tag << " pitch wheel=" << message.getPitchWheelValue()
                  << " ch=" << message.getChannel() << std::endl;
    } else if (message.isAftertouch()) {
        std::cout << tag << " aftertouch note=" << message.getNoteNumber()
                  << " value=" << message.getAfterTouchValue()
                  << " ch=" << message.getChannel() << std::endl;
    } else if (message.isChannelPressure()) {
        std::cout << tag << " channel pressure=" << message.getChannelPressureValue()
                  << " ch=" << message.getChannel() << std::endl;
    } else if (message.isSysEx()) {
        std::cout << tag << " sysex size=" << message.getSysExDataSize() << std::endl;
    } else if (message.isQuarterFrame()) {
        std::cout << tag << " mtc quarter frame seq=" << message.getQuarterFrameSequenceNumber()
                  << " value=" << message.getQuarterFrameValue() << std::endl;
    } else {
        std::cout << tag << " other raw=" << message.getDescription() << std::endl;
    }
}

void MidiInputProcess::handleLaunchpadMessage(juce::MidiInput *source, const juce::MidiMessage &message) {
    juce::ignoreUnused(source, message);
}

void MidiInputProcess::handleIncomingMidiMessage(juce::MidiInput *source, const juce::MidiMessage &message) {
    logIncomingMidiMessage(source, message);

    if (source != nullptr
        && selectedLaunchpadMidiIdentifier.isNotEmpty()
        && source->getIdentifier() == selectedLaunchpadMidiIdentifier) {
        handleLaunchpadMessage(source, message);
        return;
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
    } else if (message.isNoteOn()) {
        DBG("Note On: " + std::to_string(message.getNoteNumber()) + " - Vel: " +
            std::to_string(message.getVelocity()));
        handleNoteOn(message.getNoteNumber(), message.getVelocity());
    } else if (message.isNoteOff()) {
        handleNoteOff(message.getNoteNumber());
    }
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
            notesOnToSend[midiNoteNumber] = HeldDirectNote{.onset = time, .flushedTime = std::nullopt};
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

    if (modelConfig.inputMode != InputMode::Direct)
        return;

    const auto it = notesOnToSend.find(midiNoteNumber);
    if (it == notesOnToSend.end())
        return;

    if (modelConfig.directInputSendNoteOffs && inputFilter != nullptr) {
        const auto onset = it->second.onset;
        const int32_t noteId = static_cast<int32_t>(
            Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument + midiNoteNumber);
        const int32_t heldDur = std::max<int32_t>(1, static_cast<int32_t>(time - onset));
        const int32_t defaultDur = static_cast<int32_t>(Vocab::DurOffset + modelConfig.inputDuration);
        const int32_t correctDur = static_cast<int32_t>(Vocab::DurOffset + heldDur);

        if (it->second.flushedTime.has_value()) {
            const int32_t stamp = *it->second.flushedTime;
            inputFilter->updatesFromMain.push({
                .oldNote = Token{stamp, defaultDur, noteId},
                .newNote = Token{stamp, correctDur, noteId}
            });
            inputFilter->processUpdates();
        } else {
            inputFilter->filter(Token{static_cast<int32_t>(onset), correctDur, noteId});
        }
    }

    notesOnToSend.erase(it);

    if (! hasUnflushedDirectNotes() && tokensToSend.empty())
        stopTimer();
}
