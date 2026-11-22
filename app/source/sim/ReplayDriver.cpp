#include "VirtualOrch/sim/ReplayDriver.h"

#include "VirtualOrch/OpLog.h"

ReplayDriver::ReplayDriver(std::vector<LogReplay::Line> sessionLog, LogReplay::Options options)
    : log(std::move(sessionLog)), baseOptions(std::move(options)) {
    if (baseOptions.preset.isVoid())
        baseOptions.preset = LogReplay::hybridV1W3Preset();
    current = firstMs();
    wanted = firstMs();
    thread = std::thread([this] { threadMain(); });
}

ReplayDriver::~ReplayDriver() {
    {
        const std::lock_guard<std::mutex> lock(mutex);
        quit = true;
    }
    wake.notify_all();
    if (thread.joinable())
        thread.join();
}

auto ReplayDriver::seek(double ms) -> void {
    {
        const std::lock_guard<std::mutex> lock(mutex);
        if (ms < current.load() - 0.5)
            restart = true; // the pipeline cannot go back: start again
        wanted = ms;
    }
    wake.notify_all();
}

auto ReplayDriver::onTick(double simMs) -> bool {
    current = simMs;
    std::unique_lock<std::mutex> lock(mutex);
    // Hold here (the session stays as it is) until the target moves past us, or a restart / quit.
    // At the end of the log only a restart / quit releases it, so the session is never dropped
    // while still handed out.
    const bool atLogEnd = simMs >= lastMs() - 1e-6;
    wake.wait(lock, [&] { return restart || quit || (! atLogEnd && simMs < wanted.load()); });
    return ! restart && ! quit;
}

auto ReplayDriver::threadMain() -> void {
    if (log.empty())
        return;
    // The replay's own operation-log records are dropped, not written into the live app's oplog.
    OpLog::setCapture([](const std::string &) {});
    while (true) {
        {
            const std::lock_guard<std::mutex> lock(mutex);
            if (quit)
                break;
            restart = false;
        }
        finished = false;
        current = firstMs();
        ++runCount;
        auto options = baseOptions;
        bool handedOut = false;
        options.onSession = [this, &handedOut](AppSession &s) {
            handedOut = true;
            if (onSession)
                onSession(&s);
        };
        options.onTick = [this, &handedOut](double simMs) {
            if (simMs >= lastMs() - 1e-6)
                finished = true;
            const bool go = onTick(simMs);
            // Stopping (restart / quit): take the session back while it still exists. Every way
            // out of replay() after the first step comes through here (the end blocks until then).
            if (! go && handedOut) {
                handedOut = false;
                if (onSession)
                    onSession(nullptr);
            }
            return go;
        };
        (void) LogReplay::replay(log, options);
        if (handedOut && onSession)
            onSession(nullptr); // replay() gave up before its first step
    }
    OpLog::setCapture(nullptr);
}
