#include <gtest/gtest.h>

#include <VirtualOrch/reduction/AnticipatedStream.h>

namespace {

using Note = AnticipatedStream::Note;

auto note(int32_t onset, int32_t dur, int32_t pitch, int32_t vel = 64) -> Note {
    return Note{onset, dur, pitch, vel};
}

constexpr int32_t kTimeOf(int32_t t) { return t; }
constexpr int32_t kPitch1(int32_t p) { return 11000 + 128 + p; }
constexpr int32_t kPitch3(int32_t p) { return 11000 + 128 * 3 + p; }

struct Fixture {
    AnticipatedStream stream;
    int32_t tdId = 11770;
    Fixture() { stream.tdTokenProvider = [this] { return tdId; }; }
};

} // namespace

namespace virtual_orch_test {

TEST(AnticipatedStream, startsAtABoundaryAndWritesTdAfterTheFirstQuad) {
    Fixture f;
    f.stream.reset(48730);
    EXPECT_EQ(f.stream.base(), 48600);
    f.stream.appendEvent(note(48650, 50, 60));
    const auto t = f.stream.tokens();
    ASSERT_EQ(t.size(), 5u);
    EXPECT_EQ(t[0], 50);
    EXPECT_EQ(t[1], 10050);
    EXPECT_EQ(t[2], kPitch1(60));
    EXPECT_EQ(t[3], 11640 + 64);
    EXPECT_EQ(t[4], 11770) << "TD(0) follows the first quad";
}

TEST(AnticipatedStream, aQuadCrossingSeveralBoundariesIsFollowedByAllOfThem) {
    Fixture f;
    f.stream.reset(0);
    f.stream.appendEvent(note(0, 50, 60));   // boundary 0
    f.stream.appendEvent(note(150, 50, 61)); // none
    f.tdId = 11772;
    f.stream.appendEvent(note(650, 50, 62)); // boundaries 200, 400, 600
    const auto t = f.stream.tokens();
    ASSERT_EQ(t.size(), 5u + 4u + 4u + 3u);
    EXPECT_EQ(t[4], 11770);
    EXPECT_EQ(t[13], 11772);
    EXPECT_EQ(t[14], 11772);
    EXPECT_EQ(t[15], 11772);
}

TEST(AnticipatedStream, controlsAreInsertedOnlyOncePastDueAndInOnsetOrder) {
    Fixture f;
    f.stream.reset(0);
    f.stream.addControl(note(400, 100, 70), false);
    f.stream.addControl(note(100, 100, 65), false);
    f.stream.addControl(note(900, 100, 72), false);
    f.stream.appendEvent(note(50, 40, 60)); // clock 50
    EXPECT_EQ(f.stream.insertDue(f.stream.clock(), 0), 0);
    EXPECT_EQ(f.stream.pendingCount(), 3u);

    f.stream.appendEvent(note(450, 40, 61)); // clock 450: controls 100 and 400 are past due
    EXPECT_EQ(f.stream.insertDue(f.stream.clock(), 0), 2);
    EXPECT_EQ(f.stream.pendingCount(), 1u);
    const auto t = f.stream.tokens();
    // event, TD0 | event, TD200, TD400 | control 100 | control 400
    std::vector<int32_t> notes;
    for (size_t i = 0; i + 3 < t.size();) {
        if (t[i] >= 11768) { ++i; continue; }
        notes.push_back(t[i + 2]);
        i += 4;
    }
    ASSERT_EQ(notes.size(), 4u);
    EXPECT_EQ(notes[2], kPitch3(65));
    EXPECT_EQ(notes[3], kPitch3(70));
}

TEST(AnticipatedStream, heldControlIsFrozenAndHoldsBackTheCommittedBoundary) {
    Fixture f;
    f.stream.reset(0);
    f.stream.appendEvent(note(10, 50, 60));
    f.stream.addControl(note(20, 0, 64), true); // key down, duration unknown
    f.stream.appendEvent(note(300, 50, 61));    // makes it past due
    const size_t before = f.stream.tokens().size(); // both events and TD(0), TD(200)
    f.stream.insertDue(300, 320);               // held for 300 cs so far
    auto t = f.stream.tokens();
    EXPECT_EQ(f.stream.committedLen(), before + 0) << "commit stops at the held control's first token";
    // find the control quad (instrument 3 band) and check its frozen duration
    size_t at = 0;
    for (size_t i = 0; i + 3 < t.size(); ++i)
        if (t[i + 2] == kPitch3(64)) { at = i; break; }
    ASSERT_GT(at, 0u);
    EXPECT_EQ(t[at + 1], 10000 + 300);

    // a later insert of other tokens does not change the frozen value
    f.stream.appendEvent(note(400, 50, 62));
    f.stream.insertDue(400, 900);
    t = f.stream.tokens();
    EXPECT_EQ(t[at + 1], 10000 + 300) << "frozen while held";

    // note-off: final duration, everything commits
    EXPECT_TRUE(f.stream.confirmControl(20, 64, 700));
    t = f.stream.tokens();
    EXPECT_EQ(t[at + 1], 10000 + 700);
    EXPECT_EQ(f.stream.committedLen(), t.size());
}

TEST(AnticipatedStream, aHeldNoteIsCommittedAfterTheHoldLimit) {
    Fixture f;
    f.stream.reset(0);
    f.stream.addControl(note(0, 0, 64), true);
    f.stream.appendEvent(note(50, 50, 60));
    f.stream.insertDue(50, 60);
    EXPECT_LT(f.stream.committedLen(), f.stream.tokens().size());
    f.stream.commitLongHolds(60 + AnticipatedStream::kHoldCommitCs);
    EXPECT_EQ(f.stream.committedLen(), f.stream.tokens().size());
}

TEST(AnticipatedStream, confirmingAPendingControlBeforeItIsInsertedUsesTheFinalDuration) {
    Fixture f;
    f.stream.reset(0);
    f.stream.addControl(note(100, 0, 64), true);
    EXPECT_TRUE(f.stream.confirmControl(100, 64, 250));
    f.stream.appendEvent(note(500, 50, 60));
    f.stream.insertDue(500, 600);
    const auto t = f.stream.tokens();
    bool found = false;
    for (size_t i = 0; i + 3 < t.size(); ++i)
        if (t[i + 2] == kPitch3(64)) { found = true; EXPECT_EQ(t[i + 1], 10000 + 250); }
    EXPECT_TRUE(found);
    EXPECT_FALSE(f.stream.confirmControl(999, 1, 10));
}

TEST(AnticipatedStream, rebaseShiftsByAMultipleOfTheBoundaryAndDropsOldItems) {
    Fixture f;
    f.stream.reset(0);
    for (int32_t t = 0; t <= 9100; t += 700)
        f.stream.appendEvent(note(t, 50, 60 + (t / 700) % 12));
    EXPECT_TRUE(f.stream.needsRebase(f.stream.clock()));
    const size_t before = f.stream.itemCount();
    f.stream.rebase();
    EXPECT_EQ(f.stream.base(), AnticipatedStream::kRebaseShiftCs);
    EXPECT_EQ(f.stream.base() % TdTokens::kBoundaryCs, 0);
    EXPECT_LT(f.stream.itemCount(), before);
    EXPECT_FALSE(f.stream.needsRebase(f.stream.clock()));
    for (const auto id: f.stream.tokens())
        EXPECT_GE(id, 0);
    // times are written relative to the new base
    const auto t = f.stream.tokens();
    EXPECT_LT(t[0], 4000);
}

} // namespace virtual_orch_test
