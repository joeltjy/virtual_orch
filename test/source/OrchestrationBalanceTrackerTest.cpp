#include <gtest/gtest.h>

#include <cmath>
#include <set>
#include <vector>

#include "VirtualOrch/OrchestrationBalanceTracker.h"

TEST(OrchestrationBalanceTracker, TauStartsAtTwoAndDecaysByTenthPerSecond) {
    OrchestrationBalanceTracker tracker;
    tracker.onInstrumentAdded(0, 1000);
    EXPECT_FLOAT_EQ(tracker.tauFor(0, 1000), 2.0f);
    EXPECT_FLOAT_EQ(tracker.tauFor(0, 1000 + 1000), 1.9f);
    EXPECT_FLOAT_EQ(tracker.tauFor(0, 1000 + 5000), 1.5f);
}

TEST(OrchestrationBalanceTracker, TauFloorsAtZero) {
    OrchestrationBalanceTracker tracker;
    tracker.onInstrumentAdded(1, 0);
    EXPECT_FLOAT_EQ(tracker.tauFor(1, 30'000), 0.0f);
    EXPECT_FLOAT_EQ(tracker.tauFor(1, 100'000), 0.0f);
}

TEST(OrchestrationBalanceTracker, ReaddResetsTauToTwo) {
    OrchestrationBalanceTracker tracker;
    tracker.onInstrumentAdded(2, 0);
    tracker.onInstrumentRemoved(2);
    tracker.onInstrumentAdded(2, 10'000);
    EXPECT_FLOAT_EQ(tracker.tauFor(2, 10'000), 2.0f);
    EXPECT_FLOAT_EQ(tracker.tauFor(2, 11'000), 1.9f);
}

TEST(OrchestrationBalanceTracker, RemovedInstrumentHasZeroTau) {
    OrchestrationBalanceTracker tracker;
    tracker.onInstrumentAdded(3, 0);
    tracker.onInstrumentRemoved(3);
    EXPECT_FLOAT_EQ(tracker.tauFor(3, 500), 0.0f);
}

TEST(OrchestrationBalanceTracker, MultiInstrumentNoteIncrementsBothNumeratorsOnce) {
    OrchestrationBalanceTracker tracker;
    tracker.recordNote({0, 9}); // violin + clarinet
    tracker.recordNote({0});

    EXPECT_EQ(tracker.getTotalNotes(), 2u);
    EXPECT_FLOAT_EQ(tracker.proportion(0), 1.0f);   // both notes
    EXPECT_FLOAT_EQ(tracker.proportion(9), 0.5f);   // first note only
    EXPECT_FLOAT_EQ(tracker.proportion(1), 0.0f);
}

TEST(OrchestrationBalanceTracker, DuplicateIdsInOneNoteCountOnce) {
    OrchestrationBalanceTracker tracker;
    tracker.recordNote({0, 0, 0});
    EXPECT_EQ(tracker.getTotalNotes(), 1u);
    EXPECT_FLOAT_EQ(tracker.proportion(0), 1.0f);
}

TEST(OrchestrationBalanceTracker, ComboPenaltyMatchesSumTauLogP) {
    OrchestrationBalanceTracker tracker;
    tracker.onInstrumentAdded(0, 0);
    tracker.onInstrumentAdded(9, 0);
    tracker.recordNote({0, 9});
    tracker.recordNote({0});

    const auto view = tracker.makeBiasView(true, 0);
    const float expected =
        2.0f * std::log(1.0f) + 2.0f * std::log(0.5f); // p_0=1, p_9=0.5, tau=2
    EXPECT_NEAR(view.comboPenalty({0, 9}), expected, 1.0e-5f);
}

TEST(OrchestrationBalanceTracker, ComboPenaltyZeroWhenBiasOffOrNoHistory) {
    OrchestrationBalanceTracker tracker;
    tracker.onInstrumentAdded(0, 0);
    EXPECT_FLOAT_EQ(tracker.makeBiasView(true, 0).comboPenalty({0}), 0.0f);

    tracker.recordNote({0});
    EXPECT_FLOAT_EQ(tracker.makeBiasView(false, 0).comboPenalty({0}), 0.0f);
}

TEST(OrchestrationBalanceTracker, ComboPenaltyUsesLogFloorForZeroProportion) {
    OrchestrationBalanceTracker tracker;
    tracker.onInstrumentAdded(0, 0);
    tracker.onInstrumentAdded(1, 0);
    tracker.recordNote({0}); // p_1 == 0

    const auto view = tracker.makeBiasView(true, 0);
    const float expected = 2.0f * std::log(OrchestrationBalanceTracker::logFloor);
    EXPECT_NEAR(view.comboPenalty({1}), expected, 1.0e-5f);
}

TEST(OrchestrationBalanceTracker, SyncActiveSeedsTau) {
    OrchestrationBalanceTracker tracker;
    tracker.syncActive(std::set<int32_t>{0, 7}, 5000);
    EXPECT_FLOAT_EQ(tracker.tauFor(0, 5000), 2.0f);
    EXPECT_FLOAT_EQ(tracker.tauFor(7, 6000), 1.9f);
}

TEST(OrchestrationBalanceTracker, JamBiasViewTakesTauFromOwningStream) {
    OrchestrationBalanceTracker user;
    OrchestrationBalanceTracker model;
    user.onInstrumentAdded(0, 0);   // violin on user, aged
    model.onInstrumentAdded(9, 0);  // clarinet on model, fresh relative to later now
    model.recordNote({0, 9});

    const auto view = OrchestrationBalanceTracker::makeJamBiasView(
        user, model, model, {0}, {9}, true, 10'000);

    EXPECT_FLOAT_EQ(view.tau[0], 1.0f); // 2 - 0.1*10
    EXPECT_FLOAT_EQ(view.tau[9], 1.0f);
    EXPECT_FLOAT_EQ(view.p[0], 1.0f);
    EXPECT_FLOAT_EQ(view.p[9], 1.0f);
}
