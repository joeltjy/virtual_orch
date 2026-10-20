#include <gtest/gtest.h>

#include <VirtualOrch/PerformanceActivity.h>

namespace {
using namespace PerformanceActivity;

auto has(const std::vector<Sector> &v, Sector s) -> bool {
    return std::find(v.begin(), v.end(), s) != v.end();
}
} // namespace

namespace virtual_orch_test {

TEST(SeatingPlan, soloInstrumentsMapToTheirSectors) {
    EXPECT_EQ(sectorsFor(7, 72), std::vector<Sector>{Sector::Flutes});
    EXPECT_EQ(sectorsFor(12, 55), std::vector<Sector>{Sector::Horns});
    EXPECT_EQ(sectorsFor(14, 40), std::vector<Sector>{Sector::Tuba});
    EXPECT_EQ(sectorsFor(15, 60), std::vector<Sector>{Sector::Piano});
    const auto violin = sectorsFor(0, 80);
    EXPECT_TRUE(has(violin, Sector::FirstViolins));
    EXPECT_TRUE(has(violin, Sector::SecondViolins));
    EXPECT_EQ(violin.size(), 2u) << "violin lights both violin sections";
}

TEST(SeatingPlan, stringEnsembleTremoloAndPizzicatoFollowThePitchRange) {
    for (const int32_t local: {4, 5, 6}) {
        const auto high = sectorsFor(local, 90);
        EXPECT_TRUE(has(high, Sector::FirstViolins)) << local;
        const auto low = sectorsFor(local, 30);
        ASSERT_EQ(low.size(), 1u) << local;
        EXPECT_EQ(low[0], Sector::DoubleBasses) << local;
    }
}

TEST(SeatingPlan, ignoredInstrumentsHaveNoSector) {
    for (const int32_t local: {16, 17, 18, 19, 99, -1})
        EXPECT_TRUE(sectorsFor(local, 60).empty()) << local;
}

TEST(ActivityLog, sectorTurnsOnAtOnceAndOffTwoSecondsAfterTheNoteEnds) {
    ActivityLog log;
    const auto flutes = static_cast<size_t>(Sector::Flutes);
    EXPECT_FALSE(log.activeSectors(0)[flutes]);
    log.noteOn(1000, 7, 72, 500); // sounds 1000..1500
    EXPECT_TRUE(log.activeSectors(1000)[flutes]);
    EXPECT_TRUE(log.activeSectors(1499)[flutes]);
    EXPECT_TRUE(log.activeSectors(3499)[flutes]) << "held for 2 s after the end";
    EXPECT_FALSE(log.activeSectors(3500)[flutes]);
    EXPECT_FALSE(log.activeSectors(1000)[static_cast<size_t>(Sector::Oboes)]);
}

TEST(ActivityLog, aLaterNoteExtendsTheHoldWithoutStrobing) {
    ActivityLog log;
    const auto idx = static_cast<size_t>(Sector::Tuba);
    log.noteOn(0, 14, 40, 100);
    log.noteOn(1500, 14, 41, 100); // within the hold of the first note
    EXPECT_TRUE(log.activeSectors(3000)[idx]);
    EXPECT_TRUE(log.activeSectors(3599)[idx]);
    EXPECT_FALSE(log.activeSectors(3600)[idx]);
}

TEST(ActivityLog, pianoLightsWhileAKeyIsDownAndHoldsAfterRelease) {
    ActivityLog log;
    const auto idx = static_cast<size_t>(Sector::Piano);
    log.keyboardNoteOn(0, 60);
    EXPECT_TRUE(log.activeSectors(10000)[idx]) << "key still down";
    log.keyboardNoteOff(10000, 60);
    EXPECT_TRUE(log.activeSectors(11999)[idx]);
    EXPECT_FALSE(log.activeSectors(12000)[idx]);
}

TEST(ActivityLog, groupProportionsUseSoundingTimeOverFiveSeconds) {
    ActivityLog log;
    log.noteOn(10000, 0, 80, 3000);  // strings 3 s
    log.noteOn(10000, 7, 72, 1000);  // woodwind 1 s
    log.noteOn(11000, 11, 70, 1000); // brass 1 s
    log.noteOn(10000, 16, 40, 5000); // timpani: ignored
    const auto p = log.groupProportions(14000);
    EXPECT_NEAR(p[0], 0.6f, 1e-4);
    EXPECT_NEAR(p[1], 0.2f, 1e-4);
    EXPECT_NEAR(p[2], 0.2f, 1e-4);
    EXPECT_NEAR(p[0] + p[1] + p[2], 1.0f, 1e-5);
}

TEST(ActivityLog, groupWindowClipsOldAndFutureTime) {
    ActivityLog log;
    log.noteOn(0, 0, 80, 4000); // strings 0..4000
    // at 7000 the window is 2000..7000: only 2000 ms of it
    EXPECT_NEAR(log.groupSoundingMs(7000)[0], 2000.0, 1e-6);
    // a note that has only just started counts only what already sounded
    log.noteOn(9000, 7, 72, 6000);
    EXPECT_NEAR(log.groupSoundingMs(10000)[1], 1000.0, 1e-6);
    EXPECT_EQ(log.groupProportions(100000), (std::array<float, 3>{0, 0, 0}));
}

TEST(TdBar, sevenLogStepsAndAuto) {
    EXPECT_FALSE(tdBarFraction(TemporalDensity::kAuto).has_value());
    EXPECT_FALSE(tdBarFraction(3).has_value());
    EXPECT_NEAR(*tdBarFraction(1), 1.0f / 7.0f, 1e-6);
    EXPECT_NEAR(*tdBarFraction(8), 4.0f / 7.0f, 1e-6);
    EXPECT_NEAR(*tdBarFraction(64), 1.0f, 1e-6);
}

TEST(Console, describesActionsInWordsAndKeepsTheNewestLines) {
    using K = LaunchpadAction::Kind;
    const auto name = [](int32_t) { return juce::String("Flute"); };
    EXPECT_EQ(describe({K::InstrumentPad, 7, 0, true, false}, name), "Input: Flute on");
    EXPECT_EQ(describe({K::InstrumentPad, 7, 0, false, true}, name), "Model: Flute off");
    EXPECT_EQ(describe({K::TdLevel, 0, 16, false, false}, name), "Temporal density set to 16");
    EXPECT_EQ(describe({K::TdAuto, 0, 0, true, false}, name), "Temporal density AUTO on");
    EXPECT_EQ(describe({K::Mode, 0, 1, false, false}, name), "Mode: Jam");
    EXPECT_EQ(describe({K::Mode, 0, 0, false, false}, name), "Mode: Edit");
    EXPECT_EQ(describe({K::Mode, 0, 2, false, false}, name), "Mode: Accompaniment");
    EXPECT_EQ(describe({K::Polyphony, 0, 3, false, false}, name), "Max polyphony 3");
    EXPECT_EQ(describe({K::Pause, 0, 2, true, false}, name), "Orchestration paused");

    ConsoleLog log;
    for (int i = 0; i < 80; ++i)
        log.add("line " + juce::String(i));
    const auto last = log.latest(5);
    ASSERT_EQ(last.size(), 5u);
    EXPECT_EQ(last.front(), "line 75");
    EXPECT_EQ(last.back(), "line 79");
    EXPECT_EQ(ConsoleLog().latest(5).size(), 0u);
}

} // namespace virtual_orch_test
