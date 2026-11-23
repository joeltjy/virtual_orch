#pragma once

#include <JuceHeader.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <variant>
#include <vector>

/**
 * Operation log: everything that happens in a performance, one JSON object per line, gzipped
 * (Documents/virtual-orch/Logs/oplogs/oplog_<start>.jsonl.gz). The format is documented in
 * orchestration/performance_streams/oplogs/README.md.
 *
 * The performance threads only build a small record (field names are string literals, values are
 * numbers, short strings or vectors moved in) and append it to a buffer under a short lock. A writer
 * thread formats the JSON and compresses it every 100 ms, so no thread that makes music formats or
 * writes anything. Until start() (or a capture) nothing is recorded and every call returns at once.
 *
 * Usage:
 *   if (OpLog::on())
 *       OpLog::emit(OpLog::Src::Rt, "RT_NOTE", {{"r", OpLog::id('r', noteId)}, {"onset", onset}, ...});
 */
namespace OpLog {

enum class Src : uint8_t { Midi, Input, Rt, Ot, Play, Ui, State };

struct Raw {
    std::string json; // an already formatted JSON value (objects, arrays of objects)
};
using Ints = std::vector<int32_t>;
using Floats = std::vector<float>;
using Strings = std::vector<std::string>;
using Value = std::variant<std::monostate, bool, int64_t, double, std::string, Ints, Floats, Strings, Raw>;

struct Field {
    const char *key;
    Value value;
    Field(const char *k, bool v) : key(k), value(v) {}
    Field(const char *k, int v) : key(k), value(static_cast<int64_t>(v)) {}
    Field(const char *k, int64_t v) : key(k), value(v) {}
    Field(const char *k, uint32_t v) : key(k), value(static_cast<int64_t>(v)) {}
    Field(const char *k, size_t v) : key(k), value(static_cast<int64_t>(v)) {}
    Field(const char *k, float v) : key(k), value(static_cast<double>(v)) {}
    Field(const char *k, double v) : key(k), value(v) {}
    Field(const char *k, const char *v) : key(k), value(std::string(v)) {}
    Field(const char *k, std::string v) : key(k), value(std::move(v)) {}
    Field(const char *k, const juce::String &v) : key(k), value(v.toStdString()) {}
    Field(const char *k, Ints v) : key(k), value(std::move(v)) {}
    Field(const char *k, Floats v) : key(k), value(std::move(v)) {}
    Field(const char *k, Strings v) : key(k), value(std::move(v)) {}
    Field(const char *k, Raw v) : key(k), value(std::move(v)) {}
    Field(const char *k, Value v) : key(k), value(std::move(v)) {} // monostate = absent
};

struct Record {
    double t = 0.0;
    int32_t clock = -1;
    Src src = Src::Ui;
    const char *op = "";
    std::vector<Field> fields;
};

/** True while records are kept (a file is open or a capture is set). Cheap. */
[[nodiscard]] auto on() -> bool;

/** Record an operation (time and clock are filled in from the time / clock sources). */
auto emit(Src src, const char *op, std::vector<Field> fields) -> void;

/** Open `file` (.jsonl.gz) and start the writer thread. `t` of every line counts from here. */
auto start(const juce::File &file) -> void;
/** Flush everything left, write SESSION_END and close the file. */
auto stop() -> void;
[[nodiscard]] auto currentFile() -> juce::File;

/** The app clock (cs) and the session time (ms) stamped on every line; the app / replay set them. */
auto setClock(std::function<int32_t()> clockCs) -> void;
auto setTime(std::function<double()> sessionMs) -> void;

/**
 * Also (or only) hand every record to `capture`, formatted as one JSON line, on the calling thread
 * (the offline replay writes its oplog this way, in order, with simulated time). nullptr stops it.
 */
auto setCapture(std::function<void(const std::string &line)> capture) -> void;

/** Format one record as a JSON line (no newline). */
[[nodiscard]] auto format(const Record &r, int64_t seq) -> std::string;

/** "k57", "r4244", "o9120", "p88", "run3"; -1 → "" */
[[nodiscard]] inline auto id(char prefix, int64_t n) -> std::string {
    return n < 0 ? std::string() : std::string(1, prefix) + std::to_string(n);
}
[[nodiscard]] inline auto runId(int64_t n) -> std::string { return "run" + std::to_string(n); }

/** Records kept since start() / setCapture() (SESSION_END `ops` counts them). */
[[nodiscard]] auto recordCount() -> int64_t;

/** Next press / output / run number (process-wide counters). */
[[nodiscard]] auto nextPadId() -> int64_t;
[[nodiscard]] auto nextOutputId() -> int64_t;
[[nodiscard]] auto nextRunId() -> int64_t;
/** Reset the counters (a new oplog file starts at k0, r0, o0, p0, run0 as far as these go). */
auto resetCounters() -> void;

/** Checkpoint interval (ms of session time) and whether one is due for `part` now. */
inline constexpr double kCheckpointEveryMs = 5000.0;
[[nodiscard]] auto checkpointDue(Src part, double nowMs) -> bool;
auto markCheckpoint(Src part, double nowMs) -> void;
/** Force the next checkpointDue for every part (RUN_START, RT_REBASE). */
auto requestCheckpoints() -> void;

/** The Launchpad press being handled on this thread (PAD_ACTION / MODE `by`), -1 = none. */
inline thread_local int64_t currentPad = -1;
/** Set while a computer key presses a Launchpad button (PAD_PRESS `source`: keyboard). */
inline thread_local const char *padKey = nullptr;

/** The JSON text of a string (quoted and escaped). */
[[nodiscard]] auto quote(const std::string &s) -> std::string;

/** A small JSON object builder for Raw values (draw fields, checkpoint parts). */
class Obj {
public:
    auto add(const char *key, const Value &value) -> Obj &;
    [[nodiscard]] auto str() const -> std::string { return "{" + body + "}"; }
    [[nodiscard]] auto raw() const -> Raw { return Raw{str()}; }

private:
    std::string body;
};
[[nodiscard]] auto valueJson(const Value &value) -> std::string;

} // namespace OpLog
