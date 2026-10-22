#include "VirtualOrch/MidiOutputProcessor.h"

#include "VirtualOrch/InstrumentConstants.h"
#include "VirtualOrch/SessionLog.h"

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
    const auto message = juce::MidiMessage::noteOn(*channel, event.note, event.velocity);
    midiOutput->sendMessageNow(message);
    SessionLog::add("MIDI_OUT", "instr " + juce::String(event.instrument) + " | "
                                    + SessionLog::describeMidi(message));
}

void MidiOutputProcessor::send(const NoteOffEvent event) {
    const auto channel = channelForInstrument(event.instrument, event.note);
    if (! channel.has_value() || midiOutput == nullptr)
        return;
    const auto message = juce::MidiMessage::noteOff(*channel, event.note);
    midiOutput->sendMessageNow(message);
    SessionLog::add("MIDI_OUT", "instr " + juce::String(event.instrument) + " | "
                                    + SessionLog::describeMidi(message));
}

/**
 * This should only be used to relay incoming messages!
 * @param midiMessage The MIDI message to relay
 */
void MidiOutputProcessor::relayMidi(const juce::MidiMessage &midiMessage) {
    juce::MidiMessage newMessage = midiMessage;
    newMessage.setChannel(InstrumentConstants::kReductionMonitorMidiChannel);
    midiOutput->sendMessageNow(newMessage);
    SessionLog::add("MIDI_OUT", "thru | " + SessionLog::describeMidi(newMessage));
}

void MidiOutputProcessor::clear() {
    OutputProcessor::clear();
    if (midiOutput == nullptr)
        return;
    SessionLog::add("MIDI_OUT", "clear: note off on every channel and pitch");
    // TODO Potentially use allNotesOff if supported by Omnisphere
    for (int32_t i = 0; i < 16; i++) {
        for (int32_t j = 0; j < 128; j++) {
            midiOutput->sendMessageNow(juce::MidiMessage::noteOff(i + 1, j));
        }
    }
}
