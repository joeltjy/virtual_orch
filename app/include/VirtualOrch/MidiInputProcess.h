#pragma once

#include <JuceHeader.h>

#include <deque>
#include <map>
#include <memory>
#include <optional>

#include "Clock.h"
#include "MusicTransformer.h"
#include "OutputProcessor.h"

/**
 * MIDI / MTC input → tokens into MusicTransformer.
 * Owns windowing, buffer/direct modes, and note collection state.
 */
class MidiInputProcess : public juce::MidiInputCallback, public juce::Timer {
public:
    MidiInputProcess(Clock &clock,
                     MusicTransformer &musicTransformer,
                     ModelConfig &modelConfig,
                     std::unique_ptr<OutputProcessor> &outputProcessor,
                     juce::String &selectedMidiInputIdentifier,
                     juce::String &selectedMidiInput2Identifier,
                     juce::String &selectedMtcClockIdentifier,
                     bool &mtcClockActive);

    void handleIncomingMidiMessage(juce::MidiInput *source, const juce::MidiMessage &message) override;

    void timerCallback() override;

    /** Clear pending tokens / bass hold when generation starts. */
    void resetForStart();

    void setInputThru(bool enabled) { inputThruEnabled = enabled; }

private:
    void sendTokensToMusicTransformer(std::optional<int32_t> atTime);

    void handleContinuousControl(int controllerNumber, int controllerValue);

    void handleNoteOn(int midiNoteNumber, float velocity);

    void handleNoteOff(int midiNoteNumber);

    Clock &clock;
    MusicTransformer &musicTransformer;
    ModelConfig &modelConfig;
    std::unique_ptr<OutputProcessor> &outputProcessor;

    juce::String &selectedMidiInputIdentifier;
    juce::String &selectedMidiInput2Identifier;
    juce::String &selectedMtcClockIdentifier;
    bool &mtcClockActive;

    bool inputThruEnabled = false;

    std::deque<Token> tokensToSend;
    uint32_t lastTokenAddedTime = 0;
    std::optional<int32_t> bassHeld = std::nullopt;
    std::map<int32_t, uint32_t> notesOnToSend;

    int32_t hours = 0, minutes = 0, seconds = 0, frames = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MidiInputProcess)
};
