#pragma once

#include <JuceHeader.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

/**
 * Reads operation logs (docs/OPLOGS.txt) for the Replay page.
 *
 *  - Launches: the log pairs in Documents/virtual-orch/Logs (oplog_<stamp>.jsonl.gz with
 *    session_<stamp>.log), and the older ones in Logs/oplogs and Logs/sessions (paired by the
 *    nearest stamp within a few seconds).
 *  - A loaded oplog keeps the decompressed text and the header of every record (seq, t, clock,
 *    src, op); a record's fields are parsed when asked for. A file cut short (a crash) is read
 *    up to its last whole line.
 *  - Sessions: one per RUN_START (to its RUN_STOP, or the next RUN_START, or the end).
 *  - State at a record: each RT model's token context (the last checkpoint, then every RT_ADD /
 *    RT_REMOVE / RT_REWRITE after it), the OT's last encoder input (OT_INPUT, plus the combos
 *    drawn since), vocsep's last window (VOCSEP), and what is waiting in each queue (Q_PUSH /
 *    Q_POP / Q_DROP since the session started).
 */
namespace OpLogReader {

struct Launch {
    juce::String stamp;      // "2026-11-21_15-47-27"
    juce::File oplog;        // may not exist (old launches without an oplog)
    juce::File sessionLog;   // may not exist
    juce::Time time;
};

/** Every launch with an oplog or a session log under `logsDir`, newest first. */
auto listLaunches(const juce::File &logsDir) -> std::vector<Launch>;

struct Record {
    int64_t seq = 0;
    double t = 0.0;        // ms since the file was opened
    int32_t clock = -1;    // app clock, cs
    std::string src;
    std::string op;
    size_t offset = 0;     // the line in Log::text
    size_t length = 0;
};

struct Session {
    std::string run;       // "run3"
    size_t first = 0;      // index of RUN_START
    size_t last = 0;       // index of the last record in the session (RUN_STOP if any)
    double startMs = 0.0;
    double endMs = 0.0;
    std::string stopReason; // "stop" | "restart" | "" (no RUN_STOP)
};

class Log {
public:
    /** Read an oplog (.jsonl.gz or .jsonl). Empty with `error` set if nothing could be read. */
    static auto load(const juce::File &file) -> std::unique_ptr<Log>;
    /** From text already in memory (tests, captures): one JSON object per line. */
    static auto fromText(std::string text) -> std::unique_ptr<Log>;

    [[nodiscard]] auto records() const -> const std::vector<Record> & { return recs; }
    [[nodiscard]] auto sessions() const -> const std::vector<Session> & { return sess; }
    [[nodiscard]] auto line(size_t index) const -> std::string;
    /** The record's fields (parsed now). */
    [[nodiscard]] auto fields(size_t index) const -> juce::var;
    /** The first record at or after `ms` (records().size() if none). */
    [[nodiscard]] auto indexAt(double ms) const -> size_t;
    /** Lines that did not parse (cut-off last line, stray text). */
    [[nodiscard]] auto skippedLines() const -> int { return skipped; }

    juce::String error;

private:
    auto index() -> void;
    std::string text;
    std::vector<Record> recs;
    std::vector<Session> sess;
    int skipped = 0;
};

/** One item of an RT model's context (a note, TD, WAIT, rest or control token group). */
struct ContextItem {
    std::string kind; // "given" | "gen" | "td" | "wait" | "rest" | "control"
    std::string id;   // "k12" / "r40", "" for TD / WAIT
    std::vector<int> tokens;
};

struct RtContext {
    std::string model;               // "W" | "T3" | "v1"
    int base = 0;
    std::vector<int> prefix;         // tokens before the first item
    std::vector<ContextItem> items;
    bool exact = true;               // false between a rebase and the next checkpoint
    size_t fromCheckpoint = SIZE_MAX; // record index of the checkpoint it started from
    [[nodiscard]] auto tokens() const -> std::vector<int>;
};

struct QueueItem {
    std::string item;   // "k12", "r40", "o7", "CLEAR c1234"
    size_t pushedAt = 0; // record index of its Q_PUSH
};

/** What the models had as input, and what was waiting between the stages, at a record. */
struct State {
    std::map<std::string, RtContext> rt;                 // per model
    std::optional<size_t> otInput;                        // record index of the last OT_INPUT
    std::vector<size_t> otDrawsSinceInput;                // OT_DRAWs after it (their combos)
    std::optional<size_t> vocsep;                         // record index of the last VOCSEP
    std::map<std::string, std::vector<QueueItem>> queues; // per queue name, oldest first
    std::optional<size_t> rtCheckpointState;              // the last rt CHECKPOINT (its "state")
};

/**
 * The state just after record `index` of `log` (within its session: queues count from the
 * session's start; contexts from the last checkpoint before `index`).
 */
auto stateAt(const Log &log, size_t index) -> State;

/** The session a record belongs to, if any. */
auto sessionOf(const Log &log, size_t index) -> std::optional<size_t>;

} // namespace OpLogReader
