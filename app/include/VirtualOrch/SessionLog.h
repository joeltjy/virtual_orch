#pragma once

#include <JuceHeader.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

class Clock;

/**
 * One uncapped log file per app run, Documents/virtual-orch/Logs/sessions/session_<start>.log:
 * every MIDI message in and out, every Launchpad action, every reduction (RT) and orchestration
 * (OT) note. Each line is
 *
 *   <ms since start> | clock <cs> | <KIND> | <text>
 *
 * add() only appends to a buffer under a short lock (it is called from the MIDI, model and
 * playback threads); a writer thread flushes the buffer to disk every 100 ms. Until start() is
 * called (Main does it, tests do not) add() does nothing.
 */
class SessionLog {
public:
    /** Start writing to `file` (Main: logsDir() / session_<stamp>.log, ProjectPaths.h). */
    static auto start(const juce::File &file) -> void;
    /** Flush what is left and close the file. */
    static auto stop() -> void;
    [[nodiscard]] static auto isRunning() -> bool;
    [[nodiscard]] static auto currentFile() -> juce::File;

    /** The app clock stamped on every line (AppSession sets it; nullptr = none). */
    static auto setClock(const Clock *clock) -> void;
    [[nodiscard]] static auto clock() -> const Clock *;

    static auto add(const char *kind, const juce::String &text) -> void;

    /**
     * Also hand every line to `capture` (kind, text), with or without a log file; isRunning()
     * is true while one is set. The offline simulator records its lines this way. nullptr stops.
     */
    using Capture = std::function<void(const char *kind, const juce::String &text)>;
    static auto setCapture(Capture capture) -> void;

    /** "90 3C 64 (Note on C3 Velocity 100 Channel 1)"-style text of a MIDI message. */
    [[nodiscard]] static auto describeMidi(const juce::MidiMessage &message) -> juce::String;

    /** Timing clock and active sensing are not logged (dozens per second, no content). */
    [[nodiscard]] static auto isSkippedMidi(const juce::MidiMessage &message) -> bool;

private:
    class Writer;
    static auto writer() -> std::unique_ptr<Writer> &;
    static auto writerMutex() -> std::mutex &;
    static inline std::atomic<bool> running{false};
    static inline std::atomic<const Clock *> clockPtr{nullptr};
    static inline std::atomic<bool> capturing{false};
    static auto captureFn() -> Capture & {
        static Capture fn;
        return fn;
    }
};
