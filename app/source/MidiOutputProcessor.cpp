#include "VirtualOrch/MidiOutputProcessor.h"

#include <ranges>

MidiOutputProcessor::MidiOutputProcessor(MidiOutputType midiOutputType,
                                         const juce::String &midiOutputName,
                                         std::vector<int32_t> generatedInstruments) : outputType(midiOutputType),
    generatedInstruments(generatedInstruments) {
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
    midiOutput.reset();
}

void MidiOutputProcessor::send(const BarSeparatorEvent event) {
    // We send a Song Position Pointer message with the number of 16ths of a bar that have passed since the start, hence
    // we multiply the bar by 16.
    // TODO (Lancelot & Perry): Make this configurable
    // midiOutput->sendMessageNow(juce::MidiMessage::noteOn(15, 36, 1.0F));
}

void MidiOutputProcessor::send(const NoteOnEvent event) {
    // Find the index of the instrument in the generated instruments
    ptrdiff_t instrumentIndex = std::distance(generatedInstruments.begin(),
                                              std::ranges::find(generatedInstruments, event.instrument));
    // If the instrument is not found, use channel 16
    if (instrumentIndex >= generatedInstruments.size()) {
        instrumentIndex = 15;
    }
    // We add 1 to the instrument index because MIDI channels are 1-indexed
    midiOutput->sendMessageNow(juce::MidiMessage::noteOn(instrumentIndex + 1, event.note, event.velocity));
}

void MidiOutputProcessor::send(const NoteOffEvent event) {
    // Find the index of the instrument in the generated instruments
    ptrdiff_t instrumentIndex = std::distance(generatedInstruments.begin(),
                                              std::ranges::find(generatedInstruments, event.instrument));
    // If the instrument is not found, use channel 16
    if (instrumentIndex >= generatedInstruments.size()) {
        instrumentIndex = 15;
    }
    // We add 1 to the instrument index because MIDI channels are 1-indexed
    midiOutput->sendMessageNow(juce::MidiMessage::noteOff(instrumentIndex + 1, event.note));
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
    // TODO Potentially use allNotesOff if supported by Omnisphere
    for (int32_t i = 0; i < 16; i++) {
        for (int32_t j = 0; j < 128; j++) {
            midiOutput->sendMessageNow(juce::MidiMessage::noteOff(i + 1, j));
        }
    }
}
