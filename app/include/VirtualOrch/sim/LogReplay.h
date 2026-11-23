#pragma once

#include <JuceHeader.h>

#include <functional>

#include <map>
#include <optional>
#include <vector>

/**
 * Replays a recorded session log (Documents/virtual-orch/Logs/sessions/session_*.log) through the
 * app's own pipeline, offline, to check that the simulator reproduces a live run.
 *
 * Inputs are the log's MIDI_IN lines (the pianos, the pedal, the Launchpad), delivered at their
 * logged times through MidiInputProcess, and every generation start / stop the log shows. The
 * random choices are replayed from the log ("logged samples"): each reduction note the RT drew,
 * each W2/W3 WAIT decision, and each orchestration (OT) output. Everything else is the real code
 * running on simulated time: the input window, the reduction's context (given notes, ties, TD and
 * WAIT tokens, cuts, rewinds, the hybrid model switches), the OT hold-back and Clear Queues,
 * playback, and MIDI out. No model files are needed: the RT runs on a zero-logit backend (every
 * draw is forced) and the OT's model is replaced by the logged outputs.
 *
 * The replay writes its own log lines (SessionLog capture, simulated time) and compares them with
 * the recorded ones, tag by tag (see Report).
 *
 * Supports sessions recorded with the Hybrid reduction (Reduction Transformer Hybrid).
 */
class AppSession;

namespace LogReplay {

struct Line {
    double ms = 0.0;      // ms since the session started
    int32_t clock = -1;   // app clock (cs), -1 when the line has none ("clock -")
    juce::String tag;
    juce::String text;
};

/** Read a session log (.log, or .log.gz). Lines that do not parse are skipped. */
auto readLog(const juce::File &file) -> std::vector<Line>;
auto parseLog(const juce::String &content) -> std::vector<Line>;

struct Options {
    /** Preset JSON applied to the model config (the app's preset format). Empty: app defaults. */
    juce::var preset;
    /** Time step of the simulation in ms. */
    double stepMs = 1.0;
    /** A replayed line matches a recorded one with the same text this close in time (ms). */
    double matchToleranceMs = 25.0;
    /** Only replay the log up to this time (ms); < 0 = all of it. */
    double untilMs = -1.0;
    /**
     * Run the voice-separation model (vocsep) on the reduction's notes, as the app does. Off by
     * default: it is most of the replay's run time, its voice ids only reproduce with the exact model
     * file the recorded run used, and nothing the replay checks depends on them (RT lines are
     * compared without the voice; "RT voice" compares them when this is on).
     */
    bool voiceSeparation = false;
    /** How many differences to keep per tag for the report. */
    int keepDifferences = 12;

    /**
     * Real models (step 2 / oplog fill-in). When set, the RT runs these ONNX files (v1 = piano
     * silent, W2/W3 = piano playing) instead of the zero-logit backend, and the OT runs the real
     * IodPretrained model; every choice is still forced to the logged one, so the replay reproduces
     * the session while the models supply their probabilities (the operation log's draws).
     */
    juce::File rtSilentModel, rtWaitModel, otModel;
    /** Write the replay's operation log (.jsonl.gz, format: performance_streams/oplogs/README.md). */
    juce::File oplog;
    /** Shown in the oplog's SESSION_START (`source`): the session log it was replayed from. */
    juce::String sourceName;
    /**
     * Replay with the layout-1 Launchpad grid and pedal polarity (sessions recorded before the
     * two-sided layout, Nov 2026): presses are translated to the layout-2 pad with the same
     * meaning, both sides respond in every mode, FLUSH works in every mode and switches your pads
     * off, pedal down stops the orchestra, and the replay's MODE / LAUNCHPAD / CUT lines are
     * compared in the old words. Unset: decided by the log's "launchpad layout" line (none = 1).
     */
    std::optional<bool> layout1;

    /**
     * Replay page (ReplayDriver): called on the replaying thread once the replay's AppSession is
     * set up, before the first event; the session lives until replay() returns.
     */
    std::function<void(AppSession &)> onSession;
    /**
     * Called on the replaying thread after every simulated step with the simulated ms (the session
     * log's ms), and once more when the log has been replayed to its end. Return false to stop
     * early. It may block: the session stays as it is meanwhile (ReplayDriver pauses here).
     */
    std::function<bool(double simMs)> onTick;
};

struct TagResult {
    int recorded = 0;
    int replayed = 0;
    int matched = 0;
    std::vector<juce::String> missing; // recorded, not replayed (first few)
    std::vector<juce::String> extra;   // replayed, not recorded (first few)
    [[nodiscard]] auto exact() const -> bool { return matched == recorded && matched == replayed; }
};

struct Report {
    /** Per tag: MIDI_OUT, RT_IN, RT, OT, CUT, DROP, HYBRID, WAIT, MODE, LAUNCHPAD, MIDI_IN. */
    std::map<juce::String, TagResult> tags;
    /** Problems the replay hit (a forced draw the log could not supply, an unknown device, ...). */
    std::vector<juce::String> problems;
    /** Every line the replay logged (simulated ms). */
    std::vector<Line> replayed;
    int generationRuns = 0;
    /** The app clock's tick phase recovered from the log (ms added to the log's ms). */
    double clockPhaseMs = 0.0;
    [[nodiscard]] auto summary() const -> juce::String;
};

/**
 * Replay `log`. Needs a JUCE GUI context (juce::ScopedJuceInitialiser_GUI) on the calling thread;
 * builds its own AppSession and changes process-wide state while it runs (simulated time, the
 * SessionLog capture, TemporalDensity / RegisterRange), restored when it returns.
 */
auto replay(const std::vector<Line> &log, const Options &options = {}) -> Report;

/** The preset the Hybrid runs of Oct–Nov 2026 used (as logged on 12 Nov: hybrid_v1_w3). */
auto hybridV1W3Preset() -> juce::var;

} // namespace LogReplay
