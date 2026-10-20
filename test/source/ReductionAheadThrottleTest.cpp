#include <gtest/gtest.h>

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

    using ReductionTransformer::startAheadThrottle;
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
