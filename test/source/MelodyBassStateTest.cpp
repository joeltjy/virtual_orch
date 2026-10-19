#include <gtest/gtest.h>

#include <VirtualOrch/reduction/MelodyBassState.h>

namespace virtual_orch_test {

using State = MelodyBassState;

TEST(MelodyBassState, aHighLineAndALowLineBecomeMelodyAndBass) {
    State s;
    s.addNote(1000, 72, 80, 100, false);
    s.addNote(1000, 40, 80, 100, false);
    EXPECT_TRUE(s.runWindow(1100));
    ASSERT_EQ(s.getPieces().size(), 2u);
    int melodyPitch = 0, bassPitch = 0;
    for (const auto &p: s.getPieces())
        (p.line == State::kMelodyLine ? melodyPitch : bassPitch) = p.pitch;
    EXPECT_EQ(melodyPitch, 72);
    EXPECT_EQ(bassPitch, 40);
    EXPECT_FALSE(s.runWindow(1100)) << "same input, same R";
}

TEST(MelodyBassState, notesAreQ5V8AndNeverVelocityZero) {
    State s;
    s.addNote(1000, 130, 3, 53, false); // pitch clamps to C8, velocity 3 -> 8, duration 55
    s.runWindow(1100);
    ASSERT_EQ(s.getNotes().size(), 1u);
    EXPECT_EQ(s.getNotes()[0].pitch, 108);
    EXPECT_EQ(s.getNotes()[0].velocity, 8);
    EXPECT_EQ(s.getNotes()[0].durationCs, 55);
}

TEST(MelodyBassState, aHeldNoteCountsUntilNowAndIsProvisional) {
    State s;
    s.addNote(1000, 72, 80, 0, /*held*/ true);
    s.runWindow(1050);
    ASSERT_EQ(s.getPieces().size(), 1u);
    EXPECT_EQ(s.getPieces()[0].durationCs, 50);
    EXPECT_TRUE(s.getPieces()[0].provisional);
    EXPECT_TRUE(s.hasHeldNotes());

    // time passes without a pull or note-off: R keeps its length until A runs again
    s.runWindow(1090);
    EXPECT_EQ(s.getPieces()[0].durationCs, 90);

    EXPECT_TRUE(s.noteOff(1000, 72, 120));
    s.runWindow(1130);
    ASSERT_EQ(s.getPieces().size(), 1u);
    EXPECT_EQ(s.getPieces()[0].durationCs, 120);
    EXPECT_FALSE(s.getPieces()[0].provisional);
    EXPECT_FALSE(s.hasHeldNotes());
    EXPECT_FALSE(s.noteOff(1000, 72, 10));
}

TEST(MelodyBassState, aLongHeldNoteBeforeTheWindowStaysAtItsRealOnset) {
    State s;
    s.addNote(0, 48, 80, 0, true); // held bass note since t = 0
    s.addNote(900, 80, 80, 50, false);
    s.runWindow(1000);
    bool bass = false;
    for (const auto &p : s.getPieces())
        if (p.pitch == 48) {
            bass = true;
            EXPECT_EQ(p.onset, 0) << "not clipped on the left";
            EXPECT_EQ(p.durationCs, 1000);
        }
    EXPECT_TRUE(bass);
}

TEST(MelodyBassState, aRunReplacesExactlyTheNotesItWasGiven) {
    State s;
    s.addNote(1000, 60, 80, 100, false);
    s.runWindow(1100);
    ASSERT_EQ(s.getPieces().size(), 1u);
    // a much higher note starting inside the first one covers it as melody (cover test)
    s.addNote(1020, 84, 80, 100, false);
    s.runWindow(1120);
    bool low60AsMelody = false;
    for (const auto &p: s.getPieces())
        if (p.pitch == 60 && p.line == State::kMelodyLine)
            low60AsMelody = true;
    EXPECT_FALSE(low60AsMelody) << "A's new result replaced the old entry of that note";
}

TEST(MelodyBassState, notesOutsideTheWindowKeepTheirOldEntries) {
    State s;
    s.addNote(0, 60, 80, 50, false);
    s.runWindow(60);
    ASSERT_EQ(s.getPieces().size(), 1u);
    s.addNote(1000, 70, 80, 50, false);
    s.runWindow(1060); // the first note ended long before the window
    EXPECT_EQ(s.getPieces().size(), 2u);
}

TEST(MelodyBassState, silenceGivesAnEmptyWindowButKeepsHistoryUntilPruned) {
    State s;
    s.addNote(0, 60, 80, 50, false);
    s.runWindow(60);
    s.runWindow(3000);
    EXPECT_EQ(s.getPieces().size(), 1u);
    s.runWindow(9000);
    EXPECT_TRUE(s.getPieces().empty());
}

} // namespace virtual_orch_test
