#pragma once

#include <JuceHeader.h>

#include <atomic>
#include <functional>
#include <vector>

/**
 * Scripted input: plays a predetermined piano sequence (a MIDI file) as if it were live input.
 *
 * Select "Scripted input (MIDI file)" as the MIDI Input on the Settings page: the app then
 * takes its play-keyboard input from this player instead of a device (no MIDI keyboard needed).
 * Every message goes through MidiInputProcess::handleMidiFrom with kScriptedInputIdentifier, so
 * it is routed, relayed (input thru), logged (MIDI_IN, role "piano") and turned into reduction
 * tokens exactly like a keyboard. Sustain (CC64) in the file acts as the pedal when no Pedal
 * MIDI device is selected, the same rule as for a keyboard.
 *
 * Sent: note on / off, and controllers (CC64 sustain, CC67 soft, …). Not sent: program changes,
 * pitch bend, sysex, meta events, and channel 10 (drums) notes.
 *
 * Timing runs on its own thread (1 ms resolution, wall clock). `startOnGeneration`: the app
 * starts the file from the top on Start Generation and stops it on Stop.
 */
class ScriptedInputPlayer : private juce::Thread {
public:
    /** The pseudo device identifier / name (also what the session log shows). */
    static constexpr const char *kScriptedInputIdentifier = "virtual-orch:scripted-input";
    static constexpr const char *kScriptedInputName = "Scripted input (MIDI file)";

    using Sink = std::function<void(const juce::String &id, const juce::String &name,
                                    const juce::MidiMessage &message)>;

    explicit ScriptedInputPlayer(Sink sink);
    ~ScriptedInputPlayer() override;

    struct TrackInfo {
        int index = 0;
        juce::String name;
        int noteCount = 0;
    };

    /** Load a .mid/.midi file. Returns an error message, or empty on success. Stops playback. */
    auto load(const juce::File &file) -> juce::String;

    /** Tracks of the loaded file that have notes (for the track chooser). */
    [[nodiscard]] auto getTracks() const -> std::vector<TrackInfo> { return tracks; }

    /** -1 = all tracks merged (default), else one track index from getTracks(). Stops playback. */
    auto setTrack(int trackIndex) -> void;
    [[nodiscard]] auto getTrack() const -> int { return selectedTrack; }

    auto play() -> void;  // from the top
    auto stop() -> void;  // releases every held note and the pedal

    [[nodiscard]] auto isPlaying() const -> bool { return isThreadRunning() && ! threadShouldExit(); }
    [[nodiscard]] auto isLoaded() const -> bool { return ! events.empty(); }
    [[nodiscard]] auto getFile() const -> juce::File { return file; }
    [[nodiscard]] auto getLengthSeconds() const -> double { return lengthSeconds; }
    /** Seconds since play() (0 when stopped). */
    [[nodiscard]] auto getPositionSeconds() const -> double;

    std::atomic<bool> loop{false};
    std::atomic<bool> startOnGeneration{true};

    /** Called on the message thread when playback ends on its own (end of file, no loop). */
    std::function<void()> onFinished;

    struct Event {
        double timeSeconds = 0.0;
        juce::MidiMessage message;
    };
    /** The messages that will be sent, in order (for tests). */
    [[nodiscard]] auto getEvents() const -> const std::vector<Event> & { return events; }

    /** Build the event list from a parsed file (tracks merged or one track); for tests. */
    static auto eventsFrom(const juce::MidiFile &midi, int trackIndex) -> std::vector<Event>;

private:
    auto run() -> void override;
    auto rebuild() -> void;

    Sink sink;
    juce::File file;
    juce::MidiFile midiFile;
    std::vector<TrackInfo> tracks;
    int selectedTrack = -1;
    std::vector<Event> events;
    double lengthSeconds = 0.0;
    std::atomic<double> startMs{0.0};
    juce::CriticalSection lock;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ScriptedInputPlayer)
};
