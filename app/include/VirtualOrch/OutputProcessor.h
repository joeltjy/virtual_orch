#pragma once

#include "VirtualOrch/MusicTransformer.h"

struct NoteOnEvent {
    int32_t instrument;
    int32_t note;
    float velocity;
};

struct NoteOffEvent {
    int32_t instrument;
    int32_t note;
};

/**
 * Base class for processing output. Inherited by [MidiOutputProcessor] and [OSCOutputProcessor].
 */
class OutputProcessor {
public:
    OutputProcessor() = default;

    virtual ~OutputProcessor() = default;

    virtual void send(NoteOnEvent event) = 0;

    virtual void send(NoteOffEvent event) = 0;

    virtual void relayMidi(const juce::MidiMessage &midiMessage) {
    }

    virtual void clear() {
    }
};
