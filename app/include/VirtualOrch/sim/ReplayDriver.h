#pragma once

#include <JuceHeader.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "VirtualOrch/sim/LogReplay.h"

class AppSession;

/**
 * Runs a recorded session (its session log) through the app's pipeline on a thread of its own,
 * up to a chosen moment, and holds it there so the normal widgets can show that moment from the
 * replay's AppSession (the Replay page).
 *
 *  - seek(ms) moves the target. Later than where the replay is: it carries on from there.
 *    Earlier: the run is dropped and started again from the beginning (LogReplay cannot restore
 *    the pipeline's state mid-run), which takes about 0.5% of the session's length.
 *  - The replay's AppSession is handed out through onSession (on the replay thread) when a run
 *    starts and nullptr before it goes away (onSession(nullptr), also on the replay thread: the
 *    caller must stop using it before returning).
 *  - Models: none are loaded (zero-logit RT, logged OT outputs), every choice is the logged one;
 *    supports sessions recorded with the Hybrid reduction, as LogReplay.
 *  - While a replay runs the process-wide simulated time, the session-log capture and an oplog
 *    capture (which drops the replay's records, so the live oplog is not written) are set:
 *    use it only while generation is stopped.
 */
class ReplayDriver {
public:
    explicit ReplayDriver(std::vector<LogReplay::Line> sessionLog, LogReplay::Options options = {});
    ~ReplayDriver();

    /** Replay to `ms` (session-log ms) and hold there. */
    auto seek(double ms) -> void;
    /** Where the replay is (simulated ms), and where it is going. */
    [[nodiscard]] auto position() const -> double { return current.load(); }
    [[nodiscard]] auto target() const -> double { return wanted.load(); }
    /** True while the replay is catching up with the target. */
    [[nodiscard]] auto busy() const -> bool { return current.load() < wanted.load() - 1.0 && ! finished.load(); }
    /** The log has been replayed to its end (the last state is held). */
    [[nodiscard]] auto atEnd() const -> bool { return finished.load(); }
    [[nodiscard]] auto firstMs() const -> double { return log.empty() ? 0.0 : log.front().ms; }
    [[nodiscard]] auto lastMs() const -> double { return log.empty() ? 0.0 : log.back().ms; }
    /** How many runs were started (a backward seek starts one). */
    [[nodiscard]] auto runs() const -> int { return runCount.load(); }

    /** The replay's session, or nullptr (called on the replay thread; see above). */
    std::function<void(AppSession *)> onSession;

private:
    auto threadMain() -> void;
    auto onTick(double simMs) -> bool;

    std::vector<LogReplay::Line> log;
    LogReplay::Options baseOptions;
    std::thread thread;
    mutable std::mutex mutex;
    std::condition_variable wake;
    std::atomic<double> current{0.0};
    std::atomic<double> wanted{0.0};
    std::atomic<bool> finished{false};
    std::atomic<int> runCount{0};
    bool restart = false;
    bool quit = false;
};
