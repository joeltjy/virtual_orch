#include <gtest/gtest.h>

#include <JuceHeader.h>

#include <VirtualOrch/reduction/WaitTracker.h>
#include <VirtualOrch/widgets/WaitVisualizerWidget.h>

/** The WAIT Visualizer's state (WaitTracker) and how a result reads. */
namespace {

using Event = InformativeWaitEngine::Event;

auto ev(const char *what, int32_t at, int32_t a = 0, int32_t b = 0, std::string text = {}) -> Event {
    return Event{what, at, a, b, std::move(text)};
}

} // namespace

TEST(WaitTracker, aWaitIsPendingFromNowToRefPlusD) {
    WaitTracker t;
    t.onEvent(ev("wait", 100, 120, 3), 110); // ref 100, d 120, COUNT 3, drawn at clock 110
    const auto s = t.snapshot();
    ASSERT_TRUE(s.pending.has_value());
    EXPECT_EQ(s.pending->drawnAtCs, 110);
    EXPECT_EQ(s.pending->predictedCs, 220);
    EXPECT_EQ(s.pending->k, 3);
    EXPECT_TRUE(s.recent.empty());
}

TEST(WaitTracker, aFireRecordsTheRankAndThePlacementTime) {
    WaitTracker t;
    t.onEvent(ev("wait", 100, 120, 3), 110);
    t.onEvent(ev("anticipate", 221, 64, 2), 221);
    t.onEvent(ev("fire", 221, 3, 3), 221);
    t.onEvent(ev("placed", 221, 3, 800, "precomputed|key"), 221);
    const auto s = t.snapshot();
    EXPECT_FALSE(s.pending.has_value());
    ASSERT_EQ(s.recent.size(), 1u);
    const auto &r = s.recent.back();
    EXPECT_EQ(r.kind, WaitTracker::Result::Kind::Hit);
    EXPECT_EQ(r.rank, 2);
    EXPECT_EQ(r.placeUs, 800);
    EXPECT_TRUE(r.precomputed);
    EXPECT_TRUE(r.fromKey);
    EXPECT_EQ(WaitVisualizerView::resultText(r), "#2");
}

TEST(WaitTracker, expiredOutsideTheTopFiveAndNoAnticipation) {
    WaitTracker t;
    t.onEvent(ev("wait", 0, 30, 2), 0);
    t.onEvent(ev("expired", 36, 30, 2), 37);
    t.onEvent(ev("wait", 40, 30, 1), 40);
    t.onEvent(ev("anticipate", 70, 50, 9), 70);
    t.onEvent(ev("fire", 70, 1, 0), 70);
    t.onEvent(ev("wait", 80, 30, 1), 80);
    t.onEvent(ev("fire", 110, 1, 0), 110); // no anticipation was ready
    const auto s = t.snapshot();
    ASSERT_EQ(s.recent.size(), 3u);
    EXPECT_EQ(WaitVisualizerView::resultText(s.recent[0]), "EXPIRED");
    EXPECT_EQ(s.recent[0].k, 2);
    EXPECT_EQ(WaitVisualizerView::resultText(s.recent[1]), ">5");
    EXPECT_EQ(WaitVisualizerView::resultText(s.recent[2]), "HIT");
    EXPECT_EQ(s.results, 3u);
}

TEST(WaitTracker, aHybridSwitchCancelsThePendingWaitAndKeepsTheResults) {
    WaitTracker t;
    t.onEvent(ev("wait", 0, 30, 2), 0);
    t.onEvent(ev("expired", 36, 30, 2), 37);
    t.onEvent(ev("wait", 40, 30, 1), 40);
    t.cancel();
    const auto s = t.snapshot();
    EXPECT_FALSE(s.pending.has_value());
    EXPECT_EQ(s.recent.size(), 1u);
}
