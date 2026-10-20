#include <gtest/gtest.h>

#include <VirtualOrch/ChordNaming.h>

namespace virtual_orch_test {

using ChordNaming::nameFor;

TEST(ChordNaming, triadsSeventhsAndInversions) {
    EXPECT_EQ(nameFor({60, 64, 67}), "C");
    EXPECT_EQ(nameFor({64, 67, 72}), "C/E") << "lowest note is the bass";
    EXPECT_EQ(nameFor({55, 59, 62, 65}), "G7");
    EXPECT_EQ(nameFor({60, 64, 67, 74}), "Cadd9");
    EXPECT_EQ(nameFor({62, 65, 69, 72}), "Dm7");
    EXPECT_EQ(nameFor({57, 60, 63, 66}), "Adim7");
    EXPECT_EQ(nameFor({65, 69, 72, 76}), "Fmaj7");
}

TEST(ChordNaming, theTwoFixedCases) {
    EXPECT_EQ(nameFor({59, 62, 65, 69}), "Bm7b5") << "was Dm/B (half-diminished ranking)";
    EXPECT_EQ(nameFor({60, 63, 67, 70}), "Cm7") << "was C57 (Eb parsed as Bb)";
}

TEST(ChordNaming, flatRootsAreSpelledFlat) {
    EXPECT_EQ(nameFor({58, 62, 65}), "Bb");
    EXPECT_EQ(nameFor({63, 67, 70, 74}), "Ebmaj7");
    EXPECT_EQ(nameFor({61, 65, 68}), "Db");
    EXPECT_EQ(nameFor({66, 70, 73}), "F#") << "F# stays sharp";
    EXPECT_TRUE(ChordNaming::rootIsFlat("Ebmaj7"));
    EXPECT_FALSE(ChordNaming::rootIsFlat("F#m"));
    EXPECT_FALSE(ChordNaming::rootIsFlat("Bm7b5"));
}

TEST(ChordNaming, doublingsDoNotMatterAndSmallSetsHaveNoName) {
    EXPECT_EQ(nameFor({48, 60, 64, 67, 72}), "C");
    EXPECT_EQ(nameFor({60, 67}), "") << "two pitch classes";
    EXPECT_EQ(nameFor({60, 72, 84}), "") << "one pitch class";
    EXPECT_EQ(nameFor({}), "");
}

} // namespace virtual_orch_test
