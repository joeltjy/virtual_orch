#include <gtest/gtest.h>

#include <VirtualOrch/reduction/MelodyStream.h>

namespace {

using Piece = MelodyBassState::Piece;

auto piece(int64_t key, int32_t onset, int32_t dur, int pitch, int line = 0, bool prov = false)
    -> Piece {
    Piece p;
    p.key = key; p.onset = onset; p.durationCs = dur; p.pitch = pitch; p.velocity = 64;
    p.line = line; p.provisional = prov;
    return p;
}

auto noteIds(const std::vector<int32_t> &tokens) -> std::vector<int32_t> {
    std::vector<int32_t> out;
    for (size_t i = 1; i + 3 < tokens.size() + 0; i += 4)
        out.push_back(tokens[i + 2]);
    return out;
}

} // namespace

namespace virtual_orch_test {

TEST(MelodyStream, startsWithAnticipateAndWritesControlQuadsInTheAmtRanges) {
    MelodyStream s;
    s.reset(0);
    s.syncControls({piece(1, 50, 100, 60)});
    EXPECT_EQ(s.insertDue(10), 0) << "not due yet";
    EXPECT_EQ(s.insertDue(50), 1);
    const auto t = s.tokens();
    ASSERT_EQ(t.size(), 5u);
    EXPECT_EQ(t[0], 55027);
    EXPECT_EQ(t[1], 27513 + 50);
    EXPECT_EQ(t[2], 37513 + 100);
    EXPECT_EQ(t[3], 38513 + 60);
    EXPECT_EQ(t[4], 55156 + 64);
}

TEST(MelodyStream, eventsAndVisibleRestsUseTheEventRanges) {
    MelodyStream s;
    s.reset(0);
    s.appendEvent({30, 50, 70, 80});
    s.appendRest(130);
    const auto t = s.tokens();
    ASSERT_EQ(t.size(), 9u);
    EXPECT_EQ(t[1], 30);
    EXPECT_EQ(t[2], 10050);
    EXPECT_EQ(t[3], 11000 + 128 + 70);
    EXPECT_EQ(t[4], 55028 + 80);
    EXPECT_EQ(t[5], 130);
    EXPECT_EQ(t[6], 10000);
    EXPECT_EQ(t[7], 27512);
    EXPECT_EQ(t[8], 55028);
    EXPECT_EQ(s.clock(), 130);
}

TEST(MelodyStream, aControlIsOfferedOnlyOnce) {
    MelodyStream s;
    s.reset(0);
    const std::vector<Piece> r{piece(1, 10, 50, 60)};
    s.syncControls(r);
    EXPECT_EQ(s.insertDue(100), 1);
    s.syncControls(r);
    EXPECT_EQ(s.pendingCount(), 0u);
    EXPECT_EQ(s.insertDue(200), 0);
}

TEST(MelodyStream, aLateControlLandsAfterLaterTokensOutOfOrder) {
    MelodyStream s;
    s.reset(0);
    s.appendEvent({400, 50, 70, 80}); // clock 400
    s.syncControls({piece(1, 100, 50, 60)});
    EXPECT_EQ(s.insertDue(s.clock()), 1);
    const auto t = s.tokens();
    ASSERT_EQ(t.size(), 9u);
    EXPECT_EQ(t[5], 27513 + 100) << "past-due control after the event with the later onset";
}

TEST(MelodyStream, monophonyCheckShortensAPredecessorInTheBatchAndSkipsAgainstWrittenOnes) {
    MelodyStream s;
    s.reset(0);
    s.syncControls({piece(1, 0, 100, 60, 0), piece(2, 60, 100, 62, 0), piece(3, 0, 80, 40, 1)});
    EXPECT_EQ(s.insertDue(100), 3);
    auto t = s.tokens();
    // melody note 1 is cut at 60 (a 60 cs note), the bass note is untouched
    bool sawCut = false;
    for (size_t i = 1; i + 3 < t.size(); i += 4)
        if (t[i + 2] == 38513 + 60 && t[i] == 27513 + 0) {
            sawCut = true;
            EXPECT_EQ(t[i + 1], 37513 + 60);
        }
    EXPECT_TRUE(sawCut);

    // a newcomer overlapping an already written melody control is skipped
    s.syncControls({piece(1, 0, 100, 60, 0), piece(2, 60, 100, 62, 0), piece(3, 0, 80, 40, 1),
                    piece(4, 120, 100, 65, 0)});
    EXPECT_EQ(s.insertDue(300), 0) << "120 < end of note 2 (160): skipped, and never offered again";
    EXPECT_EQ(s.pendingCount(), 0u);
    s.syncControls({piece(1, 0, 100, 60, 0), piece(2, 60, 100, 62, 0), piece(3, 0, 80, 40, 1),
                    piece(4, 120, 100, 65, 0), piece(5, 200, 100, 67, 0)});
    EXPECT_EQ(s.insertDue(300), 1) << "200 >= 160: fits";
}

TEST(MelodyStream, heldControlsAreProvisionalUntilTheNoteOffAndKeepTheirWrittenDuration) {
    MelodyStream s;
    s.reset(0);
    s.appendEvent({0, 50, 70, 80});
    const size_t before = s.committedLen();
    s.syncControls({piece(1, 10, 40, 60, 0, /*prov*/ true)});
    s.insertDue(100);
    EXPECT_EQ(s.committedLen(), before) << "stops at the held control";
    // R grows the held piece, still provisional: the stream keeps 40
    s.syncControls({piece(1, 10, 90, 60, 0, true)});
    EXPECT_EQ(s.tokens()[before], 27513 + 10);
    EXPECT_EQ(s.tokens()[before + 1], 37513 + 40);
    // note-off: final duration, committed
    s.syncControls({piece(1, 10, 120, 60, 0, false)});
    EXPECT_EQ(s.tokens()[before + 1], 37513 + 120);
    EXPECT_EQ(s.committedLen(), s.tokens().size());
}

TEST(MelodyStream, longHeldControlsAreCommittedAtTheHoldLimit) {
    MelodyStream s;
    s.reset(0);
    s.syncControls({piece(1, 0, 40, 60, 0, true)});
    s.insertDue(50);
    EXPECT_LT(s.committedLen(), s.tokens().size());
    s.commitLongHolds(MelodyStream::kHoldCommitCs + 1);
    EXPECT_EQ(s.committedLen(), s.tokens().size());
}

TEST(MelodyStream, trimsToHalfWhenItPassesTwoHundredFiftySixTokens) {
    MelodyStream s;
    s.reset(0);
    for (int i = 0; i < 70; ++i)
        s.appendEvent({i * 10, 50, 60 + i % 12, 64});
    EXPECT_GT(s.tokens().size(), 256u);
    EXPECT_TRUE(s.trimIfNeeded());
    EXPECT_LE(s.tokens().size(), 129u);
    EXPECT_GE(s.tokens().size(), 125u);
    EXPECT_EQ(s.tokens()[0], 55027);
    EXPECT_FALSE(s.trimIfNeeded());
}

TEST(MelodyStream, trimmingADroppedHeldControlIsCountedAsAForcedCommit) {
    MelodyStream s;
    s.reset(0);
    s.syncControls({piece(1, 0, 40, 60, 0, true)});
    s.insertDue(10);
    for (int i = 0; i < 70; ++i)
        s.appendEvent({20 + i * 10, 50, 60, 64});
    EXPECT_TRUE(s.trimIfNeeded());
    EXPECT_EQ(s.forcedCommitCount(), 1);
}

TEST(MelodyStream, rebaseKeepsRecentItemsAndAllIdsInRange) {
    MelodyStream s;
    s.reset(0);
    for (int i = 0; i <= 9100; i += 500)
        s.appendEvent({i, 50, 60, 64});
    EXPECT_TRUE(s.needsRebase(s.clock()));
    s.rebase();
    EXPECT_EQ(s.base(), 6000);
    for (const auto id: s.tokens())
        EXPECT_GE(id, 0);
    EXPECT_FALSE(s.needsRebase(s.clock()));
}

} // namespace virtual_orch_test
