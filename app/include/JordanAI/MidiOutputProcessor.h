#pragma once

#include "JordanAI/OutputProcessor.h"
#include "JordanAI/MusicTransformer.h"

enum MidiOutputType : uint8_t {
    VIRTUAL,
    HARDWARE
};

class MidiOutputProcessor : public OutputProcessor {
public:
    MidiOutputProcessor(MidiOutputType midiOutputType, const juce::String &midiOutputName,
                        std::vector<int32_t> generatedInstruments);

    ~MidiOutputProcessor() override;

    void send(BarSeparatorEvent event) override;

    void send(NoteOnEvent event) override;

    void send(NoteOffEvent event) override;

    void relayMidi(const juce::MidiMessage &midiMessage) override;

    void clear() override;

private:
    MidiOutputType outputType;
    std::unique_ptr<juce::MidiOutput> midiOutput;

    std::vector<int32_t> generatedInstruments;
};
