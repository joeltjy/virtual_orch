#include <gtest/gtest.h>

#include <VirtualOrch/LaunchpadGrid.h>
#include <VirtualOrch/LaunchpadProgrammerMap.h>

namespace virtual_orch_test {

TEST(LaunchpadProgrammerMap, PadCorners) {
    const auto bottomLeft = LaunchpadProgrammerMap::padFromNote(11);
    ASSERT_TRUE(bottomLeft.has_value());
    EXPECT_EQ(bottomLeft->row, 7);
    EXPECT_EQ(bottomLeft->col, 0);

    const auto bottomRight = LaunchpadProgrammerMap::padFromNote(18);
    ASSERT_TRUE(bottomRight.has_value());
    EXPECT_EQ(bottomRight->row, 7);
    EXPECT_EQ(bottomRight->col, 7);

    const auto topLeft = LaunchpadProgrammerMap::padFromNote(81);
    ASSERT_TRUE(topLeft.has_value());
    EXPECT_EQ(topLeft->row, 0);
    EXPECT_EQ(topLeft->col, 0);

    const auto topRight = LaunchpadProgrammerMap::padFromNote(88);
    ASSERT_TRUE(topRight.has_value());
    EXPECT_EQ(topRight->row, 0);
    EXPECT_EQ(topRight->col, 7);
}

TEST(LaunchpadProgrammerMap, SideNotesColNine) {
    const auto topSide = LaunchpadProgrammerMap::padFromNote(89);
    ASSERT_TRUE(topSide.has_value());
    EXPECT_EQ(topSide->row, 0);
    EXPECT_EQ(topSide->col, LaunchpadGrid::kSideCol);

    const auto bottomSide = LaunchpadProgrammerMap::padFromNote(19);
    ASSERT_TRUE(bottomSide.has_value());
    EXPECT_EQ(bottomSide->row, 7);
    EXPECT_EQ(bottomSide->col, LaunchpadGrid::kSideCol);
}

TEST(LaunchpadProgrammerMap, RejectsOutOfRange) {
    EXPECT_FALSE(LaunchpadProgrammerMap::padFromNote(10).has_value());
    EXPECT_FALSE(LaunchpadProgrammerMap::padFromNote(91).has_value());
    EXPECT_FALSE(LaunchpadProgrammerMap::padFromNote(20).has_value());
}

TEST(LaunchpadProgrammerMap, TopCcToSceneIndex) {
    EXPECT_EQ(LaunchpadProgrammerMap::sceneIndexFromTopCc(104), std::optional<int>{0});
    EXPECT_EQ(LaunchpadProgrammerMap::sceneIndexFromTopCc(111), std::optional<int>{7});
    EXPECT_FALSE(LaunchpadProgrammerMap::sceneIndexFromTopCc(103).has_value());
    EXPECT_FALSE(LaunchpadProgrammerMap::sceneIndexFromTopCc(112).has_value());
}

TEST(LaunchpadProgrammerMap, MiniMk3TopCcToSceneIndex) {
    EXPECT_EQ(LaunchpadProgrammerMap::sceneIndexFromTopCc(91), std::optional<int>{0});
    EXPECT_EQ(LaunchpadProgrammerMap::sceneIndexFromTopCc(92), std::optional<int>{1});
    EXPECT_EQ(LaunchpadProgrammerMap::sceneIndexFromTopCc(98), std::optional<int>{7});
    EXPECT_FALSE(LaunchpadProgrammerMap::sceneIndexFromTopCc(90).has_value());
    EXPECT_FALSE(LaunchpadProgrammerMap::sceneIndexFromTopCc(99).has_value()); // logo
}

TEST(LaunchpadProgrammerMap, MiniMk3SideCcToRow) {
    EXPECT_EQ(LaunchpadProgrammerMap::sideRowFromCc(89), std::optional<int>{0});
    EXPECT_EQ(LaunchpadProgrammerMap::sideRowFromCc(79), std::optional<int>{1}); // AUTO_TD
    EXPECT_EQ(LaunchpadProgrammerMap::sideRowFromCc(69), std::optional<int>{2}); // flush
    EXPECT_EQ(LaunchpadProgrammerMap::sideRowFromCc(19), std::optional<int>{7});
    EXPECT_FALSE(LaunchpadProgrammerMap::sideRowFromCc(9).has_value());
    EXPECT_FALSE(LaunchpadProgrammerMap::sideRowFromCc(99).has_value());
    EXPECT_FALSE(LaunchpadProgrammerMap::sideRowFromCc(64).has_value());
    EXPECT_FALSE(LaunchpadProgrammerMap::sideRowFromCc(91).has_value());
    for (int cc = 0; cc < 128; ++cc)
        if (LaunchpadProgrammerMap::sideRowFromCc(cc).has_value())
            EXPECT_FALSE(LaunchpadProgrammerMap::sceneIndexFromTopCc(cc).has_value()) << cc;
}

TEST(LaunchpadProgrammerMap, NoteFromPadRoundTrip) {
    for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 8; ++col) {
            const auto note = LaunchpadProgrammerMap::noteFromPad(row, col);
            ASSERT_TRUE(note.has_value()) << "row=" << row << " col=" << col;
            const auto pad = LaunchpadProgrammerMap::padFromNote(*note);
            ASSERT_TRUE(pad.has_value());
            EXPECT_EQ(pad->row, row);
            EXPECT_EQ(pad->col, col);
        }
        const auto sideNote = LaunchpadProgrammerMap::noteFromPad(row, LaunchpadGrid::kSideCol);
        ASSERT_TRUE(sideNote.has_value());
        const auto sidePad = LaunchpadProgrammerMap::padFromNote(*sideNote);
        ASSERT_TRUE(sidePad.has_value());
        EXPECT_EQ(sidePad->row, row);
        EXPECT_EQ(sidePad->col, LaunchpadGrid::kSideCol);
    }
}

TEST(LaunchpadProgrammerMap, FullGridUnique) {
    bool seen[8][8] = {};
    bool sideSeen[8] = {};
    for (int note = 0; note < 128; ++note) {
        if (const auto pad = LaunchpadProgrammerMap::padFromNote(note)) {
            if (pad->col == LaunchpadGrid::kSideCol) {
                EXPECT_FALSE(sideSeen[pad->row]) << "duplicate side for note " << note;
                sideSeen[pad->row] = true;
            } else {
                ASSERT_GE(pad->col, 0);
                ASSERT_LT(pad->col, 8);
                EXPECT_FALSE(seen[pad->row][pad->col]) << "duplicate for note " << note;
                seen[pad->row][pad->col] = true;
            }
        }
    }
    for (int row = 0; row < 8; ++row) {
        EXPECT_TRUE(sideSeen[row]) << "missing side row=" << row;
        for (int col = 0; col < 8; ++col)
            EXPECT_TRUE(seen[row][col]) << "missing pad row=" << row << " col=" << col;
    }
}

} // namespace virtual_orch_test
