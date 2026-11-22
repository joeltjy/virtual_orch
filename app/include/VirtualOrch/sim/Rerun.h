#pragma once

#include <JuceHeader.h>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "VirtualOrch/sim/OpLogReader.h"

namespace StreamingKv {
class OnnxBackend;
}
class IodPretrained;

/**
 * "Run through the model again" (Replay page): rebuilds a logged draw's model input from the
 * operation log and runs the real model on it, giving the whole distribution of every field, not
 * just the logged top 10.
 *
 *  - RT_DRAW: the drawing model's token context at that record (OpLogReader::stateAt), then the
 *    drawn onset, duration and pitch fed one by one, as when it was drawn.
 *  - OT_DRAW: the window's encoder input (OT_INPUT), with the combos drawn before it in the window
 *    teacher-forced in, then this note's steps one by one.
 *
 * Probabilities are the model's own (softmax of its raw logits at temperature 1: the oplog's
 * p_model), so the rerun's p_model of each logged choice should equal the logged one when the
 * model file is the one the session used: a check that the input was rebuilt exactly.
 */
namespace Rerun {

struct Field {
    std::string name;             // "onset" / "dur" / "pitch" / "vel"; "groups" / a family
    int lo = 0, hi = 0;           // the field's token range [lo, hi)
    int chosen = -1;              // the logged choice (absolute token), -1 = none
    double loggedPModel = 0.0;
    double rerunPModel = 0.0;
    std::vector<std::pair<int, double>> top; // most probable tokens of the field (absolute, p_model)
};

struct Result {
    std::string error;
    std::string what;             // "RT_DRAW r12 (v1)"
    std::string model;            // "v1", "W", "T3", "OT"
    std::string modelFile;
    double t = 0.0;               // the record's time (ms)
    int64_t seq = 0;
    int contextTokens = 0;        // RT: tokens fed before the draw
    int loggedContextTokens = 0;  // RT: context_len in the log
    bool exactContext = true;     // false: the context could only be rebuilt approximately
    std::vector<Field> fields;
    [[nodiscard]] auto matchesLog() const -> bool;
};

/** The model file a tag defaults to (Documents/virtual_orch_models; the Hybrid v1 + W3 set). */
auto defaultModelFile(const std::string &tag) -> juce::File;

/** Where the reruns of a launch are kept: next to its oplog, <stem>.reruns.jsonl. */
auto rerunsFileFor(const juce::File &oplog) -> juce::File;

/** One JSON line for a result (the reruns file's format). */
auto toJson(const Result &r, const juce::String &source) -> std::string;

/** Append a result to the reruns file. */
auto append(const juce::File &file, const Result &r, const juce::String &source) -> bool;

class Engine {
public:
    Engine();
    ~Engine();

    /** Rerun the draw at `index` (RT_DRAW or OT_DRAW). Loads the model on first use (slow). */
    auto rerun(const OpLogReader::Log &log, size_t index) -> Result;

    /** Model files per tag ("v1", "W", "T3", "OT"); unset tags use defaultModelFile. */
    std::map<std::string, juce::File> modelFiles;
    /** How many of the most probable tokens each field keeps. */
    int topCount = 20;

private:
    auto rtBackend(const std::string &tag, Result &r) -> StreamingKv::OnnxBackend *;
    auto otModel(Result &r) -> IodPretrained *;
    auto rerunRt(const OpLogReader::Log &log, size_t index, Result &r) -> void;
    auto rerunOt(const OpLogReader::Log &log, size_t index, Result &r) -> void;

    std::map<std::string, std::unique_ptr<StreamingKv::OnnxBackend>> rt;
    std::map<std::string, std::string> rtFile;
    std::unique_ptr<IodPretrained> ot;
    std::string otFile;
};

} // namespace Rerun
