#include <gtest/gtest.h>

#include <VirtualOrch/OrchestraTdMeter.h>

#include <vector>

namespace virtual_orch_test {

using Note = OrchestraTdMeter::Note;

TEST(OrchestraTdMeter, plainOnsetsCountOnceEvenAsChords) {
    // 6 onsets in the 3 s ending at the newest (300), 3 notes each: 6 x 200 / 300 = 4 (TD 4)
    std::vector<Note> notes;
    for (int i = 0; i < 6; ++i)
        for (int k = 0; k < 3; ++k)
            notes.push_back({50 * (i + 1), 40, false});
    const auto r = OrchestraTdMeter::measureNotes(notes);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->events, 6.0);
    EXPECT_DOUBLE_EQ(r->td, 4.0);
    EXPECT_DOUBLE_EQ(r->tremoloEvents, 0.0);
    EXPECT_EQ(r->endCs, 300);
}

TEST(OrchestraTdMeter, aTremoloNoteHeldXSecondsCountsEightX) {
    // one tremolo note held through the whole 3 s window: 8 x 3 = 24 events = TD 16
    const auto r = OrchestraTdMeter::measureNotes({{0, 400, true}}, 300);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->endCs, 300);
    EXPECT_DOUBLE_EQ(r->tremoloEvents, 24.0);
    EXPECT_DOUBLE_EQ(r->td, 16.0);
    // only the part inside the window counts: 0.5 s -> 4 events
    const auto half = OrchestraTdMeter::measureNotes({{100, 50, true}}, 300);
    ASSERT_TRUE(half.has_value());
    EXPECT_DOUBLE_EQ(half->tremoloEvents, 4.0);
}

TEST(OrchestraTdMeter, everyTremoloNoteCountsOnItsOwn) {
    // a 3-note tremolo chord held 1 s = 24 events; 4 tremolo notes held the whole window = 96 events = TD 64
    const auto chord = OrchestraTdMeter::measureNotes({{0, 100, true}, {0, 100, true}, {0, 100, true}}, 300);
    ASSERT_TRUE(chord.has_value());
    EXPECT_DOUBLE_EQ(chord->tremoloEvents, 24.0);
    std::vector<Note> four(4, Note{0, 400, true});
    const auto r = OrchestraTdMeter::measureNotes(four, 300);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->td, 64.0);
}

TEST(OrchestraTdMeter, plainAndTremoloAdd) {
    // 1 s of tremolo (8) + 4 plain onsets = 12 events = TD 8
    const auto r = OrchestraTdMeter::measureNotes(
        {{0, 100, true}, {50, 10, false}, {100, 10, false}, {150, 10, false}, {300, 10, false}});
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->events, 12.0);
    EXPECT_DOUBLE_EQ(r->td, 8.0);
}

TEST(OrchestraTdMeter, aHeldTremoloStillCountsWithNothingNewer) {
    // the window ends at now when that is later than the newest onset
    const auto atOnset = OrchestraTdMeter::measureNotes({{1000, 500, true}});
    const auto later = OrchestraTdMeter::measureNotes({{1000, 500, true}}, 1300);
    ASSERT_TRUE(atOnset.has_value() && later.has_value());
    EXPECT_DOUBLE_EQ(atOnset->events, 0.0);
    EXPECT_DOUBLE_EQ(later->td, 16.0);
}

TEST(OrchestraTdMeter, aClearRemovesTheNotesFromItsTime) {
    OrchestraTdMeter m;
    for (int t = 0; t <= 600; t += 25)
        m.add(t, 20, false);
    ASSERT_DOUBLE_EQ(m.read()->td, 12 * 200.0 / 300.0); // (300, 600]: 325..600 = 12 onsets
    m.clearFrom(400);
    const auto r = m.read();
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->endCs, 375);
    m.clear();
    EXPECT_FALSE(m.read().has_value());
}

TEST(OrchestraTdMeter, aNotesRealLengthReplacesItsProvisionalOne) {
    // Hybrid v1.3: a tremolo note goes out with 5 s until the model's OFF; the OFF's length then counts
    OrchestraTdMeter meter;
    meter.add(0, 500, true, 7);
    auto r = meter.read(300);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(r->td, 16.0); // 8 x 3 s = 24 events
    meter.setDuration(-1, 10);     // no source: nothing changes
    EXPECT_DOUBLE_EQ(meter.read(300)->td, 16.0);
    meter.setDuration(7, 50);
    r = meter.read(300);
    EXPECT_DOUBLE_EQ(r->events, 4.0); // 8 x 0.5 s
    EXPECT_DOUBLE_EQ(r->td, 8.0 / 3.0);
}

TEST(TremoloBiasController, growsBelowTheTargetAndDecaysAtOrAboveIt) {
    TremoloBiasController c;
    EXPECT_DOUBLE_EQ(c.tick(16, 4.0), TremoloBiasController::kGain * 2.0); // e = 2
    for (int i = 0; i < 20; ++i)
        c.tick(16, 4.0);
    EXPECT_DOUBLE_EQ(c.bonus(), TremoloBiasController::kMax);
    c.tick(16, 20.0);
    EXPECT_DOUBLE_EQ(c.bonus(), TremoloBiasController::kMax * TremoloBiasController::kDecay);
}

TEST(TremoloBiasController, offAtAutoLowTdOrWithoutAReading) {
    TremoloBiasController c;
    c.tick(32, 4.0);
    ASSERT_GT(c.bonus(), 0.0);
    const double b = c.bonus();
    c.tick(TemporalDensity::kAuto, 4.0);
    EXPECT_DOUBLE_EQ(c.bonus(), b * TremoloBiasController::kDecay);
    TremoloBiasController low;
    EXPECT_DOUBLE_EQ(low.tick(4, 1.0), 0.0) << "TD <= 4 never adds tremolo";
    EXPECT_DOUBLE_EQ(low.tick(16, std::nullopt), 0.0) << "no reading (paused, stale): no growth";
}

} // namespace virtual_orch_test
