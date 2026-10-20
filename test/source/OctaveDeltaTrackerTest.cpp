#include <gtest/gtest.h>

#include <VirtualOrch/OctaveDeltaTracker.h>

namespace virtual_orch_test {

TEST(OctaveDeltaTracker, CountsByDelta) {
    OctaveDeltaTracker tracker;
    tracker.record(-1);
    tracker.record(0);
    tracker.record(0);
    tracker.record(1);
    tracker.record(2); // ignored
    tracker.record(-2); // ignored

    const auto snap = tracker.snapshot();
    EXPECT_EQ(snap.counts[0], 1u);
    EXPECT_EQ(snap.counts[1], 2u);
    EXPECT_EQ(snap.counts[2], 1u);
    EXPECT_EQ(snap.total, 4u);
    EXPECT_GT(snap.sequence, 0u);
}

TEST(OctaveDeltaTracker, ResetClearsCounts) {
    OctaveDeltaTracker tracker;
    tracker.record(0);
    tracker.record(1);
    const auto before = tracker.snapshot().sequence;
    tracker.reset();
    const auto snap = tracker.snapshot();
    EXPECT_EQ(snap.total, 0u);
    EXPECT_EQ(snap.counts[0], 0u);
    EXPECT_EQ(snap.counts[1], 0u);
    EXPECT_EQ(snap.counts[2], 0u);
    EXPECT_GT(snap.sequence, before);
}

} // namespace virtual_orch_test
