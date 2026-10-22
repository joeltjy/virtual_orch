#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "VirtualOrch/Clock.h"
#include "VirtualOrch/Metrics.h"
#include "VirtualOrch/reduction/ReductionTransformer.h"
#include "VirtualOrch/ui/ModelConfigurationComponent.h"

namespace {

/** Minimal ReductionTransformer so ahead-throttle helpers are reachable in tests. */
class TestReduction final : public ReductionTransformer {
public:
    explicit TestReduction(ModelConfig &config)
        : ReductionTransformer("Test Reduction", config) {
    }

    [[nodiscard]] auto isModelLoaded() const -> bool override { return true; }

    void run() override {}

    using ReductionTransformer::finishAheadThrottleIfDue;
    using ReductionTransformer::startAheadThrottle;
    auto setGeneratedUpTo(int32_t cs) -> void { currentTime = cs; }
};

} // namespace

TEST(ReductionAheadThrottle, ResetClearsOverflowPauseWithDeadline) {
    ModelConfig config;
    config.outputAheadThrottleSeconds = 2;
    TestReduction reduction(config);

    reduction.startAheadThrottle();
    ASSERT_TRUE(reduction.overflowPause.get());
    ASSERT_TRUE(reduction.isGenerationStopped());

    // Mimic the old restart bug: zero only the deadline.
    // (Cannot touch the private field; resetAheadThrottle is the correct path.)
    reduction.resetAheadThrottle();

    EXPECT_FALSE(reduction.overflowPause.get());
    EXPECT_FALSE(reduction.isGenerationStopped());
}

TEST(ReductionAheadThrottle, RestartMustNotLeaveGenerationStopped) {
    ModelConfig config;
    config.outputAheadThrottleSeconds = 5;
    TestReduction reduction(config);

    reduction.startAheadThrottle();
    ASSERT_TRUE(reduction.isGenerationStopped());

    // stopGeneration / threadRun entry both call this; without it RT emits nothing.
    reduction.resetAheadThrottle();
    EXPECT_FALSE(reduction.isGenerationStopped());
}

TEST(ReductionAheadThrottle, keepsWaitingWhileTheModelIsStillTooFarAhead) {
    ModelConfig config;
    config.outputAheadThrottleSeconds = 1;
    config.outputMaxAheadSeconds = 5;
    TestReduction reduction(config);
    Metrics metrics;
    Clock clock(metrics);
    reduction.clock = &clock; // not running: getTime() stays put
    const auto now = static_cast<int32_t>(clock.getTime());

    reduction.setGeneratedUpTo(now + 600); // 6 s ahead (long gaps)
    reduction.startAheadThrottle();
    std::this_thread::sleep_for(std::chrono::milliseconds(1100)); // the 1 s pause is over
    reduction.finishAheadThrottleIfDue();
    EXPECT_TRUE(reduction.isGenerationStopped()) << "no extra note while still > 5 s ahead";

    reduction.setGeneratedUpTo(now + 450); // the clock caught up (or a cut moved it back)
    reduction.finishAheadThrottleIfDue();
    EXPECT_FALSE(reduction.isGenerationStopped());
}
