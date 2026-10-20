#include <gtest/gtest.h>

#include <VirtualOrch/reduction/AnticipatedStream.h>

namespace {

using Note = AnticipatedStream::Note;

auto note(int32_t onset, int32_t dur, int32_t pitch, int32_t vel = 64) -> Note {
    return Note{onset, dur, pitch, vel};
}

constexpr int32_t kTimeOf(int32_t t) { return t; }
constexpr int32_t kPitch1(int32_t p) { return 11000 + 128 + p; }
constexpr int32_t kPitch3(int32_t p) { return 11000 + 128 * 0 + p; } // the given instrument (0)

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
    // find the given quad (instrument 0 band) and check its frozen duration
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

namespace virtual_orch_test {

TEST(AnticipatedStream, doubleAnticipationControlsUseActualTimeAndPriority) {
    Fixture f;
    f.stream.reset(0);
    // M (melody, instrument 2): delta 300 cs, priority 0; X (piano, instrument 0): priority 1.
    f.stream.addControl(note(700, 100, 60), false, /*instrument*/ 2, /*delta*/ 300, /*priority*/ 0);
    f.stream.addControl(note(400, 100, 72), false, 0, 0, 1);
    f.stream.addControl(note(400, 100, 50), false, 2, 0, 0); // M exactly tied with X at 400? (actual 400)
    EXPECT_FALSE(f.stream.hasPendingDueBy(399));
    EXPECT_TRUE(f.stream.hasPendingDueBy(400));
    f.stream.appendEvent(note(10, 40, 61));  // clock 10
    EXPECT_EQ(f.stream.insertDue(400, 0), 3) << "M(700) is due at 700 - 300 = 400, tied with X";
    const auto t = f.stream.tokens();
    std::vector<int32_t> instruments;
    for (size_t i = 0; i + 3 < t.size();) {
        if (t[i] >= 11768 && i + 3 >= t.size()) break;
        if (t[i] >= 11768) { ++i; continue; }
        instruments.push_back((t[i + 2] - 11000) / 128);
        i += 4;
    }
    ASSERT_EQ(instruments.size(), 4u);
    EXPECT_EQ(instruments[0], 1) << "the event";
    EXPECT_EQ(instruments[1], 2) << "melody first on the actual-time tie (priority 0)";
    EXPECT_EQ(instruments[2], 2);
    EXPECT_EQ(instruments[3], 0) << "piano last";
}

TEST(AnticipatedStream, tdBoundariesFollowTheActualTimeOfAControlQuad) {
    Fixture f;
    f.stream.reset(0);
    f.stream.addControl(note(1000, 100, 60), false, 2, 300, 0); // actual 700
    f.stream.appendEvent(note(0, 40, 61));                      // TD(0)
    f.stream.insertDue(800, 0);
    int td = 0;
    for (const auto id: f.stream.tokens())
        if (id >= 11768)
            ++td;
    EXPECT_EQ(td, 1 + 3) << "TD(0), then TD 200, 400, 600 after the melody quad (actual 700)";
}

} // namespace virtual_orch_test

namespace virtual_orch_test {

TEST(AnticipatedStream, insertNextTakesOnePendingNoteAtATime) {
    Fixture f;
    f.stream.reset(0);
    f.stream.addControl(note(100, 50, 60), false);
    f.stream.addControl(note(300, 50, 62), false);
    EXPECT_TRUE(f.stream.insertNext(0));
    EXPECT_EQ(f.stream.pendingCount(), 1u);
    EXPECT_EQ(f.stream.nextPendingActual(), 300);
    EXPECT_TRUE(f.stream.insertNext(0));
    EXPECT_FALSE(f.stream.insertNext(0));
}

TEST(AnticipatedStream, theT3TieRuleWritesTheFirstGeneratedNoteBeforeTheGivenOnesAtThatOnset) {
    Fixture f;
    f.stream.reset(0);
    f.stream.addControl(note(500, 50, 60), false);
    f.stream.appendEvent(note(500, 50, 70));
    EXPECT_TRUE(f.stream.eventEmittedAt(500));
    EXPECT_FALSE(f.stream.eventEmittedAt(400));
    // the given note at 500 is released only inclusively; exclusive leaves it for after the event
    f.stream.addControl(note(700, 50, 61), false);
    EXPECT_EQ(f.stream.insertDue(500, 0, /*inclusive*/ true), 1);
    const auto t = f.stream.tokens();
    std::vector<int32_t> instruments;
    for (size_t i = 0; i + 3 < t.size();) {
        if (t[i] >= 11768) { ++i; continue; }
        instruments.push_back((t[i + 2] - 11000) / 128);
        i += 4;
    }
    ASSERT_EQ(instruments.size(), 2u);
    EXPECT_EQ(instruments[0], 1) << "generated first";
    EXPECT_EQ(instruments[1], 0) << "then the given note at the same onset";

    Fixture g;
    g.stream.reset(0);
    g.stream.addControl(note(500, 50, 60), false);
    EXPECT_EQ(g.stream.insertDue(500, 0, /*inclusive*/ false), 0) << "exclusive: not yet at 500";
    EXPECT_EQ(g.stream.insertDue(501, 0, false), 1);
}

TEST(AnticipatedStream, amtVocabHasTheFlagControlQuadsRestsAndEventVelocities) {
    AnticipatedStream stream(AnticipatedStream::Vocab::Amt);
    int32_t td = 55289;
    stream.tdTokenProvider = [&] { return td; };
    stream.reset(0);
    AnticipatedStream::ControlSpec melody;
    melody.instrument = 0;
    melody.deltaCs = 300;
    melody.priority = 0;
    melody.controlToken = true;
    stream.addControl(note(700, 100, 64, 80), false, melody); // M, due at 400
    stream.addControl(note(450, 60, 72, 96), false);          // piano event (given)
    stream.appendEvent(note(10, 40, 61, 64));
    stream.appendRest(110);
    stream.insertDue(400, 0);
    auto t = stream.tokens();
    EXPECT_EQ(t[0], 55027) << "ANTICIPATE";
    EXPECT_EQ(stream.prefixTokens(), 1u);
    // event quad: time, dur, note (instr 1), velocity 55028 + v ; TD(0) ; rest quad ; M control
    EXPECT_EQ(t[1], 10);
    EXPECT_EQ(t[2], 10040);
    EXPECT_EQ(t[3], 11000 + 128 + 61);
    EXPECT_EQ(t[4], 55028 + 64);
    EXPECT_EQ(t[5], 55289) << "TD id of the AMT block";
    EXPECT_EQ(t[6], 110);
    EXPECT_EQ(t[7], 10000);
    EXPECT_EQ(t[8], 27512);
    EXPECT_EQ(t[9], 55028);
    // M control quad written with the control ids (time 700, dur 100, instr 0 pitch 64, vel 80)
    bool sawControl = false;
    for (size_t i = 1; i + 3 < t.size(); ++i)
        if (t[i] == 27513 + 700) {
            sawControl = true;
            EXPECT_EQ(t[i + 1], 37513 + 100);
            EXPECT_EQ(t[i + 2], 38513 + 64);
            EXPECT_EQ(t[i + 3], 55156 + 80);
        }
    EXPECT_TRUE(sawControl);

    // the piano event (due at 450) goes in as an ordinary instrument-0 event quad
    stream.insertDue(450, 0);
    t = stream.tokens();
    bool sawPiano = false;
    for (size_t i = 1; i + 3 < t.size(); ++i)
        if (t[i] == 450) {
            sawPiano = true;
            EXPECT_EQ(t[i + 2], 11000 + 72);
            EXPECT_EQ(t[i + 3], 55028 + 96);
        }
    EXPECT_TRUE(sawPiano);
    EXPECT_EQ(stream.committedLen(), t.size());
}

TEST(AnticipatedStream, amtEvictionRebaseKeepsTheAnticipateFlagAsFirstToken) {
    AnticipatedStream stream(AnticipatedStream::Vocab::Amt);
    stream.tdTokenProvider = [] { return 55289; };
    stream.reset(0);
    for (int32_t onset = 0; onset <= 7000; onset += 400)
        stream.appendEvent(note(onset, 50, 60 + (onset / 400) % 12));
    const auto before = stream.tokens();
    // pretend the cache evicted 60 tokens after the 4 sink rows
    const int32_t k = stream.rebaseForEviction(60, 4);
    ASSERT_GT(k, 0);
    EXPECT_EQ(k % 3000, 0);
    const auto after = stream.tokens();
    EXPECT_EQ(after[0], 55027);
    EXPECT_LT(after.size(), before.size());
    for (const auto id: after)
        EXPECT_GE(id, 0);
}

TEST(AnticipatedStream, perLineMonophonyShortensAPredecessorInTheBatchAndSkipsAgainstWrittenOnes) {
    Fixture f;
    f.stream.reset(0);
    AnticipatedStream::ControlSpec mel;
    mel.line = 0;
    f.stream.addControl(note(0, 100, 60), false, mel);
    f.stream.addControl(note(60, 100, 62), false, mel);
    EXPECT_EQ(f.stream.insertDue(100, 0), 2);
    bool cut = false;
    const auto t = f.stream.tokens();
    for (size_t i = 0; i + 3 < t.size();) {
        if (t[i] >= 11768) { ++i; continue; }
        if (t[i + 2] == 11000 + 60 && t[i] == 0) { cut = true; EXPECT_EQ(t[i + 1], 10060); }
        i += 4;
    }
    EXPECT_TRUE(cut) << "the first note was cut at the onset of the second";
    f.stream.addControl(note(120, 100, 65), false, mel);
    EXPECT_EQ(f.stream.insertDue(300, 0), 0) << "starts before the written note ends: skipped";
}

} // namespace virtual_orch_test
