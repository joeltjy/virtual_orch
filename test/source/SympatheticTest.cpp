#include <gtest/gtest.h>

#include <VirtualOrch/Sympathetic.h>

using namespace Sympathetic;

namespace {

auto partOf(std::string_view name) -> int32_t {
    const auto index = partIndex(name);
    EXPECT_TRUE(index.has_value()) << name;
    return index.value_or(-1);
}

} // namespace

// ---- Warp ---------------------------------------------------------------------------------

TEST(SympatheticWarp, IdentityAtWarpOne) {
    const auto [onset, duration] = warpNote(1200, 40, 1000, 1.0f, 0, 999);
    EXPECT_EQ(onset, 1200);
    EXPECT_EQ(duration, 40);
}

TEST(SympatheticWarp, StretchesAboutNow) {
    // onset' = t + w (o - t) = 1000 + 2 * 200; duration' = 2 * 40
    const auto [onset, duration] = warpNote(1200, 40, 1000, 2.0f, 0, 999);
    EXPECT_EQ(onset, 1400);
    EXPECT_EQ(duration, 80);
}

TEST(SympatheticWarp, CompressesAboutNow) {
    const auto [onset, duration] = warpNote(1200, 40, 1000, 0.5f, 0, 999);
    EXPECT_EQ(onset, 1100);
    EXPECT_EQ(duration, 20);
}

TEST(SympatheticWarp, PastOnsetKeepsItsTime) {
    const auto [onset, duration] = warpNote(900, 40, 1000, 2.0f, 0, 999);
    EXPECT_EQ(onset, 900);
    EXPECT_EQ(duration, 80);
}

TEST(SympatheticWarp, NeverBeforePreviousOnset) {
    // Previous (warped) note at 1300; warp dropped to 0.5: 1000 + 0.5 * 400 = 1200 < 1300.
    const auto [onset, duration] = warpNote(1400, 40, 1000, 0.5f, 1300, 999);
    EXPECT_EQ(onset, 1300);
    EXPECT_EQ(duration, 20);
}

TEST(SympatheticWarp, FloorIsCappedByTheModelsOwnOnset) {
    // The model's own onset (1250) is before the floor (1300): the note may not be pushed past
    // where the model put it, so it lands there, not at the warped 1125 or the floor.
    const auto [onset, duration] = warpNote(1250, 40, 1000, 0.5f, 1300, 999);
    EXPECT_EQ(onset, 1250);
    (void) duration;
}

TEST(SympatheticWarp, DurationClamped) {
    EXPECT_EQ(warpNote(1200, 800, 1000, 2.0f, 0, 999).second, 999);
    EXPECT_EQ(warpNote(1200, 1, 1000, 0.5f, 0, 999).second, 1);
}

// ---- Beat tracker -------------------------------------------------------------------------

TEST(SympatheticBeats, ReferenceFromFirstBeatsThenRatio) {
    BeatTracker beats;
    EXPECT_FALSE(beats.onBeat(0).has_value());
    EXPECT_FALSE(beats.onBeat(500).has_value());
    EXPECT_FALSE(beats.onBeat(1000).has_value());
    const auto first = beats.onBeat(1500); // 3 intervals of 500 ms: reference set
    ASSERT_TRUE(first.has_value());
    EXPECT_FLOAT_EQ(*first, 1.0f);
    EXPECT_DOUBLE_EQ(beats.referenceIntervalMs(), 500.0);

    // Slower beats (750 ms) move warp above 1 gradually (smoothing).
    float warp = 1.0f;
    double t = 1500;
    for (int i = 0; i < 20; ++i) {
        t += 750;
        const auto w = beats.onBeat(t);
        ASSERT_TRUE(w.has_value());
        EXPECT_GE(*w, warp - 1e-6f);
        warp = *w;
    }
    EXPECT_NEAR(warp, 1.5f, 0.01f);
}

TEST(SympatheticBeats, FasterConductingCompresses) {
    BeatTracker beats;
    for (double t: {0.0, 600.0, 1200.0, 1800.0})
        beats.onBeat(t);
    float warp = 1.0f;
    double t = 1800;
    for (int i = 0; i < 20; ++i) {
        t += 300;
        warp = beats.onBeat(t).value_or(warp);
    }
    EXPECT_NEAR(warp, 0.5f, 0.01f);
}

TEST(SympatheticBeats, ClampedAndIgnoresDoubleDetections) {
    BeatTracker beats;
    for (double t: {0.0, 300.0, 600.0, 900.0})
        beats.onBeat(t);
    EXPECT_FALSE(beats.onBeat(950).has_value()); // 50 ms: a double detection
    float warp = 1.0f;
    double t = 900;
    for (int i = 0; i < 30; ++i) {
        t += 1900;
        warp = beats.onBeat(t).value_or(warp);
    }
    EXPECT_FLOAT_EQ(warp, kMaxWarp);
}

TEST(SympatheticBeats, LongGapRestartsWithoutChangingWarp) {
    BeatTracker beats;
    for (double t: {0.0, 500.0, 1000.0, 1500.0})
        beats.onBeat(t);
    EXPECT_FALSE(beats.onBeat(10000).has_value()); // conductor paused
    const auto w = beats.onBeat(10500);
    ASSERT_TRUE(w.has_value());
    EXPECT_NEAR(*w, 1.0f, 1e-6f);
}

// ---- Cueing / Select ----------------------------------------------------------------------

TEST(SympatheticCue, PointCuesThenSelects) {
    State state;
    const auto oboe = partOf("Oboe");
    EXPECT_TRUE(state.point(oboe));
    EXPECT_EQ(state.selectedPart.load(), -1);
    EXPECT_FALSE(state.point(oboe));
    EXPECT_EQ(state.selectedPart.load(), oboe);
}

TEST(SympatheticCue, OnlyCuedInstrumentsOfferedOnceEnabled) {
    State state;
    const std::vector<int32_t> pads{0, 1, 4, 6, 7, 8, 15};
    EXPECT_EQ(state.filterAllowed(pads), pads); // off: unchanged

    state.enabled.store(true);
    EXPECT_TRUE(state.filterAllowed(pads).empty()); // nothing cued yet

    state.point(partOf("Flute"));
    EXPECT_EQ(state.filterAllowed(pads), (std::vector<int32_t>{7}));

    state.point(partOf("Viola")); // a solo string brings the articulations (4, 6)
    EXPECT_EQ(state.filterAllowed(pads), (std::vector<int32_t>{1, 4, 6, 7}));
}

TEST(SympatheticCue, ViolinOneAndTwoShareTheViolin) {
    State state;
    EXPECT_TRUE(state.point(partOf("Violin 1")));
    // Violin 2 maps to the same instrument, already cued: this is a select.
    EXPECT_FALSE(state.point(partOf("Violin 2")));
}

TEST(SympatheticCue, ResetUncuesEverything) {
    State state;
    state.enabled.store(true);
    state.point(partOf("Tuba"));
    state.warp.store(1.7f);
    state.reset();
    EXPECT_EQ(state.cuedMask.load(), 0u);
    EXPECT_FLOAT_EQ(state.warp.load(), 1.0f);
    EXPECT_TRUE(state.enabled.load()); // reset keeps the layer on
}

TEST(SympatheticCue, StringEnsembleSoundsOnlyOnCuedSoloStrings) {
    State state;
    state.enabled.store(true);
    state.point(partOf("Cello"));
    // A high ensemble note routes to the violin (uncued): silent. Cello range note: sounds.
    EXPECT_FALSE(state.maySound(InstrumentConstants::kStringEnsembleLocalId, 84));
    EXPECT_TRUE(state.maySound(2, 48));
    EXPECT_FALSE(state.maySound(7, 72)); // flute not cued
}

// ---- Velocity -----------------------------------------------------------------------------

TEST(SympatheticVelocity, SelectBoostsSelectedAndCutsTheRest) {
    const auto horns = partOf("French Horns");
    EXPECT_EQ(velocityFor(80, 12, horns, 1.0f, 20), 100);
    EXPECT_EQ(velocityFor(80, 7, horns, 1.0f, 20), 60);
    EXPECT_EQ(velocityFor(80, 7, -1, 1.0f, 20), 80); // nothing selected
}

TEST(SympatheticVelocity, ExpressionScalesAndClamps) {
    EXPECT_FLOAT_EQ(expressionScaleForSpread(0.5f), 1.0f);
    EXPECT_EQ(velocityFor(100, 7, -1, expressionScaleForSpread(1.0f), 20), 127);
    EXPECT_EQ(velocityFor(10, 7, partOf("Oboe"), expressionScaleForSpread(0.0f), 20), 1);
}

TEST(SympatheticVelocity, StringArticulationsFollowASelectedStringPart) {
    const auto cello = partOf("Cello");
    EXPECT_EQ(velocityFor(80, 4, cello, 1.0f, 20), 100); // tremolo strings
    EXPECT_EQ(velocityFor(80, 4, partOf("Flute"), 1.0f, 20), 60);
}

TEST(SympatheticVelocity, DisabledLeavesVelocityAlone) {
    State state;
    state.selectedPart.store(partOf("Flute"));
    EXPECT_EQ(state.playbackVelocity(80, 8, 70), 80);
    EXPECT_TRUE(state.maySound(8, 70));
}
