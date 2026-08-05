#include "VirtualOrch/MidiOutputProcessor.h"

MidiOutputProcessor::MidiOutputProcessor(MidiOutputType midiOutputType,
                                         const juce::String &midiOutputName)
    : outputType(midiOutputType) {
    switch (midiOutputType) {
        case VIRTUAL:
            midiOutput = juce::MidiOutput::createNewDevice(midiOutputName);
            break;
        case HARDWARE:
            midiOutput = juce::MidiOutput::openDevice(midiOutputName);
            break;
    }
}

MidiOutputProcessor::~MidiOutputProcessor() {
    if (midiOutput != nullptr)
        clear();
    midiOutput.reset();
}

void MidiOutputProcessor::send(const NoteOnEvent event) {
    const auto channel = channelForInstrument(event.instrument, event.note);
    if (! channel.has_value() || midiOutput == nullptr)
        return;
    midiOutput->sendMessageNow(
        juce::MidiMessage::noteOn(*channel, event.note, event.velocity));
}

void MidiOutputProcessor::send(const NoteOffEvent event) {
    const auto channel = channelForInstrument(event.instrument, event.note);
    if (! channel.has_value() || midiOutput == nullptr)
        return;
    midiOutput->sendMessageNow(juce::MidiMessage::noteOff(*channel, event.note));
}

/**
 * This should only be used to relay incoming messages!
 * @param midiMessage The MIDI message to relay
 */
void MidiOutputProcessor::relayMidi(const juce::MidiMessage &midiMessage) {
    juce::MidiMessage newMessage = midiMessage;
    newMessage.setChannel(16);
    midiOutput->sendMessageNow(newMessage);
}

void MidiOutputProcessor::clear() {
    OutputProcessor::clear();
    if (midiOutput == nullptr)
        return;
    // TODO Potentially use allNotesOff if supported by Omnisphere
    for (int32_t i = 0; i < 16; i++) {
        for (int32_t j = 0; j < 128; j++) {
            midiOutput->sendMessageNow(juce::MidiMessage::noteOff(i + 1, j));
        }
    }
}
