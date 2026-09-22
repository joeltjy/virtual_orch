#pragma once

#include <JuceHeader.h>

#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>

#include "Clock.h"
#include "Fifo.h"
#include "OutputProcessor.h"
#include "VirtualOrch/InputFilter.h"
#include "VirtualOrch/InstrumentConstants.h"
#include "VirtualOrch/LaunchpadGrid.h"
#include "VirtualOrch/reduction/ReductionTransformer.h"
#include "VirtualOrch/ui/ModelConfigurationComponent.h"

/**
 * MIDI / MTC input → tokens into the active music backend via InputFilter.
 * Owns windowing, buffer/direct modes, and note collection state.
 */
class MidiInputProcess : public juce::MidiInputCallback, public juce::Timer {
public:
    MidiInputProcess(Clock &clock,
                     std::function<ReductionTransformer &()> activeReduction,
                     ModelConfig &modelConfig,
                     std::unique_ptr<OutputProcessor> &outputProcessor,
                     std::unique_ptr<InputFilter> &inputFilter,
                     juce::String &selectedMidiInputIdentifier,
                     juce::String &selectedLaunchpadMidiIdentifier,
                     juce::String &selectedMtcClockIdentifier,
                     bool &mtcClockActive);

    ~MidiInputProcess() override;

    void handleIncomingMidiMessage(juce::MidiInput *source, const juce::MidiMessage &message) override;

    void timerCallback() override;

    /** Clear pending tokens / bass hold when generation starts. */
    void resetForStart();

    void setInputThru(bool enabled) { inputThruEnabled = enabled; }

    /** When set, input thru only relays if this returns true (e.g. Jam mode). */
    std::function<bool()> allowInputThru;

    [[nodiscard]] auto getLaunchpadGrid() -> LaunchpadGrid & { return launchpadGrid; }

    /** Open/replace MIDI out for Launchpad LEDs. */
    auto setLaunchpadMidiOutput(std::unique_ptr<juce::MidiOutput> output) -> void;

    /** Resolve MIDI out for a Launchpad input (ids often differ on Linux), enter programmer mode. */
    auto setLaunchpadMidiOutputForInputDevice(const juce::MidiDeviceInfo &inputInfo) -> void;

private:
    void sendTokensToMusicTransformer(std::optional<int32_t> atTime);

    void logIncomingMidiMessage(juce::MidiInput *source, const juce::MidiMessage &message);

    void handleLaunchpadMessage(juce::MidiInput *source, const juce::MidiMessage &message);

    void handleNoteOn(int midiNoteNumber, float velocity);

    void handleNoteOff(int midiNoteNumber);

    [[nodiscard]] auto isMusicThreadRunning() const -> bool;

    [[nodiscard]] auto musicDirectInputBlock() -> juce::Atomic<bool> &;

    Clock &clock;
    std::function<ReductionTransformer &()> activeReduction;
    ModelConfig &modelConfig;
    std::unique_ptr<OutputProcessor> &outputProcessor;
    std::unique_ptr<InputFilter> &inputFilter;

    juce::String &selectedMidiInputIdentifier;
    juce::String &selectedLaunchpadMidiIdentifier;
    juce::String &selectedMtcClockIdentifier;
    bool &mtcClockActive;

    bool inputThruEnabled = true;

    LaunchpadGrid launchpadGrid;
    std::unique_ptr<juce::MidiOutput> launchpadMidiOutput;
    juce::CriticalSection launchpadMidiOutputLock;
    /** 0 unknown, 1 MK2, 2 Mini Mk3 / X — set when MIDI out opens. */
    int launchpadFamily = 0;

    struct HeldDirectNote {
        uint32_t onset = 0;
        std::optional<int32_t> flushedTime;
        int32_t velocity = 100;
    };

    std::deque<Token> tokensToSend;
    uint32_t lastTokenAddedTime = 0;
    std::optional<int32_t> bassHeld = std::nullopt;
    std::map<int32_t, HeldDirectNote> notesOnToSend;

    auto hasUnflushedDirectNotes() const -> bool;

    int32_t hours = 0, minutes = 0, seconds = 0, frames = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MidiInputProcess)
};
