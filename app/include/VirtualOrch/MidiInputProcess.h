#pragma once

#include <JuceHeader.h>

#include <algorithm>
#include <atomic>
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
namespace PerformanceActivity {
class ActivityLog;
}

class MidiInputProcess : public juce::MidiInputCallback, public juce::Timer {
public:
    MidiInputProcess(Clock &clock,
                     std::function<ReductionTransformer &()> activeReduction,
                     ModelConfig &modelConfig,
                     std::unique_ptr<OutputProcessor> &outputProcessor,
                     std::unique_ptr<InputFilter> &inputFilter,
                     juce::String &selectedMidiInputIdentifier,
                     juce::String &selectedPedalMidiInputIdentifier,
                     juce::String &selectedAdditionalMidiInputIdentifier,
                     juce::String &selectedLaunchpadMidiIdentifier,
                     juce::String &selectedMtcClockIdentifier,
                     bool &mtcClockActive);

    ~MidiInputProcess() override;

    void handleIncomingMidiMessage(juce::MidiInput *source, const juce::MidiMessage &message) override;

    /**
     * Handle a message from the input whose identifier is `sourceId` (roles are matched by
     * identifier, as for hardware). handleIncomingMidiMessage calls this for devices; the
     * scripted input (ScriptedInputPlayer, identifier kScriptedInputIdentifier) calls it
     * directly, so a MIDI file selected as "MIDI Input" plays exactly like a keyboard.
     */
    void handleMidiFrom(const juce::String &sourceId, const juce::String &sourceName,
                        const juce::MidiMessage &message);

    void timerCallback() override;
    /** Send every key not yet sent to the reduction (the direct-input flush). */
    void flushDirectNotes();

    /** Clear pending tokens / bass hold when generation starts. */
    void resetForStart();

    void setInputThru(bool enabled) { inputThruEnabled = enabled; }

    /** When set, input thru only relays if this returns true (e.g. Jam mode). */
    std::function<bool()> allowInputThru;

    /**
     * Called on the MIDI thread each time the pedal stops the orchestra (CC64 going from down
     * to up) with the app clock time in cs. AppSession points it at
     * OutputPlayback::pedalClearRequests so the playback queue is cleared without waiting for
     * the transformer threads.
     */
    std::function<void(int32_t pedalTimeCs)> onPedalSilence;

    /**
     * A pedal device was selected (also on startup, when the saved one is restored). A pedal at
     * rest sends nothing, so start silenced: the orchestra sounds once the pedal goes down.
     */
    void pedalDeviceSelected();

    /** The pedal device was cleared (the pianos' CC64 is used again): the orchestra sounds. */
    void pedalDeviceCleared();

    /**
     * Pedal state from any source: `pause` stops the orchestra (pause + clear the future), false
     * lets it play. CC64 (through the pedal polarity) and the Sympathetic Stop/Play gestures both
     * come here. Safe from the MIDI and message threads.
     */
    auto setPedal(bool pause, const juce::String &source) -> void;

    /**
     * Log replay of layout-1 sessions (before Nov 2026): the app's pedal had the old polarity,
     * pedal DOWN stopped the orchestra and pedal up let it play. Never set in the live app.
     */
    std::atomic<bool> layout1PedalPolarity{false};

    /** Record whether the pedal is silencing the orchestra (pedal up), at wall-clock `nowMs`. */
    auto notePedalState(bool silenced, int64_t nowMs) -> void {
        if (silenced) {
            int64_t expected = -1;
            pedalSilentSinceMs.compare_exchange_strong(expected, nowMs);
        } else {
            pedalSilentSinceMs.store(-1);
        }
    }

    /**
     * How long the pedal has kept the orchestra silent (pedal up), in ms; -1 while it lets it
     * sound (pedal down) or before any pedal message.
     */
    [[nodiscard]] auto getPedalSilentForMs(int64_t nowMs) const -> int64_t {
        const auto since = pedalSilentSinceMs.load();
        return since < 0 ? -1 : std::max<int64_t>(0, nowMs - since);
    }

    /** Play-keyboard notes are reported here for the seating plan; set by AppSession. */
    PerformanceActivity::ActivityLog *activityLog = nullptr;

    [[nodiscard]] auto getLaunchpadGrid() -> LaunchpadGrid & { return launchpadGrid; }

    /** Open/replace MIDI out for Launchpad LEDs. */
    auto setLaunchpadMidiOutput(std::unique_ptr<juce::MidiOutput> output) -> void;

    /** Resolve MIDI out for a Launchpad input (ids often differ on Linux), enter programmer mode. */
    auto setLaunchpadMidiOutputForInputDevice(const juce::MidiDeviceInfo &inputInfo) -> void;

private:
    void sendTokensToMusicTransformer(std::optional<int32_t> atTime);

    void logIncomingMidiMessage(const juce::String &sourceId, const juce::String &sourceName,
                                const juce::MidiMessage &message);

    void handleLaunchpadMessage(const juce::MidiMessage &message);

    /** CC64 → generationPause (up pauses and clears, down resumes); ignores everything else. */
    void handleSustainPedal(const juce::MidiMessage &message);

    /** `piano`: 0 = MIDI Input, 1 = Additional MIDI (held keys are tracked per piano). */
    void handleNoteOn(int midiNoteNumber, float velocity, int piano = 0);

    void handleNoteOff(int midiNoteNumber, int piano = 0);

    [[nodiscard]] auto isMusicThreadRunning() const -> bool;

    [[nodiscard]] auto musicDirectInputBlock() -> juce::Atomic<bool> &;

    Clock &clock;
    std::function<ReductionTransformer &()> activeReduction;
    /**
     * The pedal is up, so the orchestra is paused. False until the first CC64, or true from when
     * a pedal device is selected. Written on the MIDI and message threads.
     */
    std::atomic<bool> pedalSilenced{false};
    /** Wall-clock ms the pedal last silenced the orchestra, -1 while it sounds. */
    std::atomic<int64_t> pedalSilentSinceMs{-1};
    ModelConfig &modelConfig;
    std::unique_ptr<OutputProcessor> &outputProcessor;
    std::unique_ptr<InputFilter> &inputFilter;

    juce::String &selectedMidiInputIdentifier;
    juce::String &selectedPedalMidiInputIdentifier;
    juce::String &selectedAdditionalMidiInputIdentifier;
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
        /** Token::keyId of this key press (flushed token and its note-off update). */
        int32_t keyId = -1;
    };

    /** Next Token::keyId for a keyboard note-on. */
    int32_t nextKeyId = 0;

    std::deque<Token> tokensToSend;
    uint32_t lastTokenAddedTime = 0;
    std::optional<int32_t> bassHeld = std::nullopt;
    /** Held keys by heldKeyOf(piano, pitch), so both pianos can hold the same pitch. */
    std::map<int32_t, HeldDirectNote> notesOnToSend;
    [[nodiscard]] static auto heldKeyOf(int piano, int pitch) -> int32_t { return piano * 128 + pitch; }
    [[nodiscard]] static auto pitchOfHeldKey(int32_t key) -> int32_t { return key % 128; }

    auto hasUnflushedDirectNotes() const -> bool;

    int32_t hours = 0, minutes = 0, seconds = 0, frames = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MidiInputProcess)
};
