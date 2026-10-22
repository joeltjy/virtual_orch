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

TEST(ChordNaming, slashBassesAreSpelledFromTheRoot) {
    EXPECT_EQ(nameFor({63, 66, 71}), "B/D#") << "not B/Eb";
    EXPECT_EQ(nameFor({68, 71, 76}), "E/G#") << "not E/Ab";
    EXPECT_EQ(nameFor({61, 64, 69}), "A/C#") << "not A/Db";
    EXPECT_EQ(nameFor({63, 68, 72}), "Ab/Eb") << "a flat root keeps a flat fifth";
    EXPECT_EQ(nameFor({58, 60, 64, 67}), "C/Bb") << "the seventh is a B, flattened";
}

TEST(ChordNaming, minorRootsFollowTheMinorKeys) {
    EXPECT_EQ(nameFor({61, 64, 68}), "C#m") << "not Dbm";
    EXPECT_EQ(nameFor({68, 71, 75}), "G#m") << "not Abm";
    EXPECT_EQ(nameFor({63, 66, 70}), "Ebm");
    EXPECT_EQ(nameFor({58, 61, 65}), "Bbm");
    EXPECT_EQ(nameFor({61, 65, 68}), "Db") << "major stays flat";
    EXPECT_EQ(nameFor({68, 72, 75}), "Ab");
    EXPECT_FALSE(ChordNaming::rootIsFlat("C#m"));
}

TEST(ChordNaming, theScoreDropsExtensions) {
    using ChordNaming::basicChord;
    EXPECT_EQ(basicChord("Cmaj7"), "C");
    EXPECT_EQ(basicChord("G7"), "G");
    EXPECT_EQ(basicChord("Cadd9"), "C");
    EXPECT_EQ(basicChord("Dm7"), "Dm");
    EXPECT_EQ(basicChord("C#m9"), "C#m");
    EXPECT_EQ(basicChord("Bm7b5"), "Bdim");
    EXPECT_EQ(basicChord("Adim7"), "Adim");
    EXPECT_EQ(basicChord("Ebmaj7/G"), "Eb");
    EXPECT_EQ(basicChord("G7sus4"), "Gsus4");
    EXPECT_EQ(basicChord("Csus2"), "Csus2");
    EXPECT_EQ(basicChord("C5"), "C5");
    EXPECT_EQ(basicChord("Caug"), "Caug");
}

TEST(ChordNaming, theScoreShowsTheChordWithoutItsBass) {
    EXPECT_EQ(ChordNaming::withoutBass("C/E"), "C");
    EXPECT_EQ(ChordNaming::withoutBass("Am/C"), "Am");
    EXPECT_EQ(ChordNaming::withoutBass("G7"), "G7");
}

TEST(ChordNaming, doublingsDoNotMatterAndEveryTwoNotesHaveAName) {
    EXPECT_EQ(nameFor({48, 60, 64, 67, 72}), "C");
    EXPECT_EQ(nameFor({60, 72, 84}), "C") << "octaves: the note name";
    EXPECT_EQ(nameFor({60}), "") << "a single note";
    EXPECT_EQ(nameFor({}), "");
}

TEST(ChordNaming, twoPitchClassesAreNamedFromTheirInterval) {
    EXPECT_EQ(nameFor({60, 67}), "C5");
    EXPECT_EQ(nameFor({60, 63}), "Cm");
    EXPECT_EQ(nameFor({60, 64}), "C");
    EXPECT_EQ(nameFor({60, 65}), "F/C");
    EXPECT_EQ(nameFor({60, 68}), "Ab/C");
    EXPECT_EQ(nameFor({60, 69}), "Am/C");
    EXPECT_EQ(nameFor({60, 70}), "C7");
    EXPECT_EQ(nameFor({60, 71}), "Cmaj7");
    EXPECT_EQ(nameFor({59, 72}), "Cmaj7/B");
    EXPECT_EQ(nameFor({62, 66}), "D");
    EXPECT_EQ(nameFor({70, 77}), "Bb5") << "flat root";
}

TEST(ChordNaming, aClusterTheNamerCannotNameStillGetsOne) {
    // C Db D Eb: no chord; leaving notes out ends at a name.
    EXPECT_FALSE(nameFor({60, 61, 62, 63}).empty());
    EXPECT_FALSE(nameFor({60, 61, 66, 71}).empty());
}

} // namespace virtual_orch_test
