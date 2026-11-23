#include <gtest/gtest.h>

#include <VirtualOrch/AppSession.h>
#include <VirtualOrch/sim/ReplayDriver.h>

#include <chrono>
#include <thread>

namespace virtual_orch_test {

namespace {

auto fixture(const char *name) -> juce::File {
    return juce::File(VIRTUAL_ORCH_TEST_RESOURCES).getChildFile("log_replay").getChildFile(name);
}

/** Wait until `done()` or ~30 s. */
template <typename F>
auto waitFor(F done) -> bool {
    for (int i = 0; i < 3000; ++i) {
        if (done())
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

} // namespace

TEST(ReplayDriver, holdsAtAMomentGoesOnAndStartsAgainForAnEarlierOne) {
    const juce::ScopedJuceInitialiser_GUI gui;
    const auto log = LogReplay::readLog(fixture("session_2026-10-22_14-51-28.log.gz"));
    ASSERT_FALSE(log.empty());
    std::atomic<AppSession *> session{nullptr};
    std::atomic<int> handOuts{0}, takeBacks{0};
    {
        ReplayDriver driver(log);
        driver.onSession = [&](AppSession *s) {
            (s != nullptr ? handOuts : takeBacks)++;
            session = s;
        };
        const double t1 = driver.firstMs() + 60000.0;
        driver.seek(t1);
        ASSERT_TRUE(waitFor([&] { return driver.position() >= t1 && ! driver.busy(); }));
        ASSERT_NE(session.load(), nullptr);
        const auto clockAtT1 = session.load()->clock.getTime();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        EXPECT_EQ(session.load()->clock.getTime(), clockAtT1) << "held: nothing moves";
        EXPECT_LT(driver.position(), t1 + 50.0);

        const double t2 = t1 + 30000.0; // later: carries on, same run
        driver.seek(t2);
        ASSERT_TRUE(waitFor([&] { return driver.position() >= t2 && ! driver.busy(); }));
        EXPECT_EQ(driver.runs(), 1);

        const double t3 = driver.firstMs() + 20000.0; // earlier: a new run
        driver.seek(t3);
        ASSERT_TRUE(waitFor([&] { return driver.runs() == 2 && driver.position() >= t3 && ! driver.busy(); }));
        EXPECT_LT(driver.position(), t3 + 50.0);
        EXPECT_EQ(takeBacks.load(), 1) << "the first run's session was taken back before it went";
        EXPECT_NE(session.load(), nullptr);

        driver.seek(driver.lastMs() + 1000.0); // past the end: holds the last state
        ASSERT_TRUE(waitFor([&] { return driver.atEnd(); }));
        EXPECT_NE(session.load(), nullptr) << "kept at the end";
    }
    EXPECT_EQ(session.load(), nullptr) << "taken back when the driver goes";
    EXPECT_EQ(handOuts.load(), takeBacks.load());
}

} // namespace virtual_orch_test
