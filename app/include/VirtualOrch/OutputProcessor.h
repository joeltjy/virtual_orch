#pragma once

#include "VirtualOrch/InstrumentConstants.h"
#include "VirtualOrch/MusicTransformer.h"

#include <algorithm>
#include <optional>
#include <vector>

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
 * MIDI channel routing uses InstrumentConstants::kLocalInstrumentOutputChannels.
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
        outputInstruments.clear();
    }

    auto ensureInstrument(int32_t instrumentId) -> void {
        if (std::find(outputInstruments.begin(), outputInstruments.end(), instrumentId)
            == outputInstruments.end()) {
            outputInstruments.push_back(instrumentId);
        }
    }

    /** 1-based MIDI channel for local instrument id + pitch, if routed. */
    [[nodiscard]] auto channelForInstrument(int32_t localInstrumentId, int32_t pitch) const
        -> std::optional<int32_t> {
        return InstrumentConstants::outputChannelForLocalInstrumentId(localInstrumentId, pitch);
    }

    std::vector<int32_t> outputInstruments;
};
