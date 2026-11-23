#include <gtest/gtest.h>

#include <VirtualOrch/Clock.h>
#include <VirtualOrch/Metrics.h>
#include <VirtualOrch/SessionLog.h>

#include <mutex>
#include <thread>
#include <vector>

namespace virtual_orch_test {

namespace {
/** Collects the CLOCK lines SessionLog receives while alive. */
struct ClockLineCapture {
    std::mutex mutex;
    std::vector<juce::String> lines;

    ClockLineCapture() {
        SessionLog::setCapture([this](const char *kind, const juce::String &text) {
            if (juce::String(kind) != "CLOCK")
                return;
            const std::lock_guard lock(mutex);
            lines.push_back(text);
        });
    }
    ~ClockLineCapture() { SessionLog::setCapture(nullptr); }

    auto snapshot() -> std::vector<juce::String> {
        const std::lock_guard lock(mutex);
        return lines;
    }
};
} // namespace

TEST(ClockMtcWatchdog, logsWhenMissingMtcStopsTheClock) {
    ClockLineCapture capture;
    Metrics metrics;
    Clock clock(metrics);

    clock.setMtcTime(0); // what Start does with MTC Clock ticked
    EXPECT_TRUE(clock.isRunning());
    std::this_thread::sleep_for(std::chrono::milliseconds(300)); // no further MTC arrives

    EXPECT_FALSE(clock.isRunning());
    const auto lines = capture.snapshot();
    ASSERT_EQ(lines.size(), 1U);
    EXPECT_TRUE(lines[0].contains("MTC watchdog")) << lines[0];
    EXPECT_TRUE(lines[0].contains("untick MTC Clock")) << lines[0];
}

TEST(ClockMtcWatchdog, staysQuietWhileMtcKeepsArriving) {
    ClockLineCapture capture;
    Metrics metrics;
    Clock clock(metrics);

    // A full MTC frame every ~33 ms, as a DAW sends at 30 fps.
    for (uint32_t t = 0; t < 10; ++t) {
        clock.setMtcTime(t * 3);
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }

    EXPECT_TRUE(clock.isRunning());
    EXPECT_TRUE(capture.snapshot().empty());
    clock.startFreeRunning(); // cancel the pending watchdog before the clock is destroyed
}

} // namespace virtual_orch_test
