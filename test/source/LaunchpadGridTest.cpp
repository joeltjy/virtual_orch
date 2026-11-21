#include <gtest/gtest.h>

#include <VirtualOrch/LaunchpadGrid.h>
#include <VirtualOrch/LaunchpadLighting.h>
#include <VirtualOrch/LaunchpadProgrammerMap.h>
#include <VirtualOrch/MoodPresets.h>
#include <VirtualOrch/RegisterRange.h>
#include <VirtualOrch/TemporalDensity.h>

#include <map>
#include <set>
#include <vector>

namespace virtual_orch_test {

namespace {

constexpr int kStrings = LaunchpadGrid::kInstrumentRowFirst;
constexpr int kWinds = LaunchpadGrid::kInstrumentRowFirst + 1;
constexpr int kBrass = LaunchpadGrid::kInstrumentRowFirst + 2;
constexpr int kPerc = LaunchpadGrid::kInstrumentRowFirst + 3;
/** Column of pad `c` (0-3) on your side / the model's side. */
constexpr auto user(int c) -> int { return LaunchpadGrid::kUserColFirst + c; }
constexpr auto model(int c) -> int { return LaunchpadGrid::kModelColFirst + c; }

/** Map the app's instruments on both sides (as AppSession does). */
auto mapAppInstruments(LaunchpadGrid &grid) -> void {
    struct Pad { int row, col; int32_t id; };
    const Pad pads[] = {{kStrings, 0, 6}, {kStrings, 1, 5}, {kStrings, 2, 4}, {kWinds, 0, 7},
                        {kWinds, 1, 8},   {kWinds, 2, 9},   {kWinds, 3, 10},  {kBrass, 0, 12},
                        {kBrass, 1, 11},  {kBrass, 2, 13},  {kBrass, 3, 14},  {kPerc, 0, 16}};
    for (const auto &pad: pads) {
        grid.padInstruments[static_cast<size_t>(pad.row)][static_cast<size_t>(user(pad.col))] = pad.id;
        grid.padInstruments[static_cast<size_t>(pad.row)][static_cast<size_t>(model(pad.col))] = pad.id;
    }
}

auto drain(CircularFifo<InstrumentUpdate> &updates) -> std::vector<InstrumentUpdate> {
    std::vector<InstrumentUpdate> out;
    InstrumentUpdate u{};
    while (updates.pull(u))
        out.push_back(u);
    return out;
}

auto ledFor(std::map<int, uint8_t> &leds, int row, int col) -> uint8_t {
    const auto note = LaunchpadProgrammerMap::noteFromPad(row, col);
    EXPECT_TRUE(note.has_value());
    return note.has_value() ? leds[*note] : LaunchpadLighting::kOff;
}

/** TemporalDensity is process-wide; put it back to AUTO so tests don't leak into each other. */
auto restoreAutoTd(LaunchpadGrid &grid, uint32_t nowMs) -> void {
    if (! TemporalDensity::isAuto())
        grid.handleInput(LaunchpadGrid::kTdRow, LaunchpadGrid::kSideCol, true, nowMs);
    ASSERT_TRUE(TemporalDensity::isAuto());
}

} // namespace

TEST(LaunchpadGrid, StartsAllOff) {
    LaunchpadGrid grid;
    for (int row = 0; row < LaunchpadGrid::kRows; ++row) {
        for (int col = 0; col < LaunchpadGrid::kCols; ++col) {
            EXPECT_FALSE(grid.getGridState(row, col));
            EXPECT_EQ(grid.getEffectiveState(row, col), 0);
        }
    }
}

TEST(LaunchpadGrid, InstrumentPadPressTogglesAndEmits) {
    LaunchpadGrid grid;
    grid.setSelectedMode(OrchestrationMode::Edit);
    CircularFifo<InstrumentUpdate> updates;
    grid.instrumentUpdates = &updates;
    grid.padInstruments[kWinds][user(1)] = 7;

    grid.handleInput(kWinds, user(1), true, 1000);
    InstrumentUpdate u{};
    ASSERT_TRUE(updates.pull(u));
    EXPECT_EQ(u.localInstrumentId, 7);
    EXPECT_EQ(u.target, InstrumentUpdateTarget::User);
    EXPECT_EQ(u.state, 1);
    EXPECT_FALSE(updates.pull(u));

    grid.handleInput(kWinds, user(1), false, 1000); // release ignored
    EXPECT_FALSE(updates.pull(u));

    grid.handleInput(kWinds, user(1), true, 1000 + LaunchpadGrid::kBurstMs);
    ASSERT_TRUE(updates.pull(u));
    EXPECT_EQ(u.state, 0);
}

TEST(LaunchpadGrid, LeftHalfIsUserRightHalfIsModel) {
    EXPECT_FALSE(LaunchpadGrid::targetForPad(LaunchpadGrid::kModeRow, 0).has_value());
    EXPECT_FALSE(LaunchpadGrid::targetForPad(LaunchpadGrid::kTdRow, 0).has_value());
    for (int row = kStrings; row <= kPerc; ++row) {
        EXPECT_EQ(LaunchpadGrid::targetForPad(row, user(0)), InstrumentUpdateTarget::User);
        EXPECT_EQ(LaunchpadGrid::targetForPad(row, user(3)), InstrumentUpdateTarget::User);
        EXPECT_EQ(LaunchpadGrid::targetForPad(row, model(0)), InstrumentUpdateTarget::Model);
        EXPECT_EQ(LaunchpadGrid::targetForPad(row, model(3)), InstrumentUpdateTarget::Model);
    }
    EXPECT_FALSE(LaunchpadGrid::targetForPad(LaunchpadGrid::kRegisterRow, 0).has_value());
    EXPECT_FALSE(LaunchpadGrid::targetForPad(LaunchpadGrid::kPresetRow, 0).has_value());
}

TEST(LaunchpadGrid, BothSidesRespondInStepInOnlyTheModelsInJamAndAccompaniment) {
    LaunchpadGrid grid;
    CircularFifo<InstrumentUpdate> updates;
    grid.instrumentUpdates = &updates;
    mapAppInstruments(grid);

    grid.setSelectedMode(OrchestrationMode::Edit);
    EXPECT_EQ(grid.controlledTarget(), InstrumentUpdateTarget::User);
    grid.handleInput(kBrass, user(0), true, 1000);
    grid.handleInput(kBrass, model(0), true, 1000); // Step-in: the model's side responds too
    EXPECT_TRUE(grid.getGridState(kBrass, user(0)));
    EXPECT_TRUE(grid.getGridState(kBrass, model(0)));

    for (const auto mode: {OrchestrationMode::Jam, OrchestrationMode::Accompaniment}) {
        grid.setSelectedMode(mode);
        EXPECT_EQ(grid.controlledTarget(), InstrumentUpdateTarget::Model);
    }
    grid.handleInput(kBrass, user(0), true, 2000); // ignored in Jam / Accompaniment
    grid.handleInput(kBrass, model(0), true, 2000); // toggles off
    EXPECT_TRUE(grid.getGridState(kBrass, user(0)));
    EXPECT_FALSE(grid.getGridState(kBrass, model(0)));

    const auto emitted = drain(updates);
    ASSERT_EQ(emitted.size(), 3u);
    EXPECT_EQ(emitted[0].target, InstrumentUpdateTarget::User);
    EXPECT_EQ(emitted[1].target, InstrumentUpdateTarget::Model);
    EXPECT_EQ(emitted[2].target, InstrumentUpdateTarget::Model);
}

TEST(LaunchpadGrid, AModelPadPressInStepInKeepsYourPresetActive) {
    LaunchpadGrid grid;
    mapAppInstruments(grid);
    grid.setSelectedMode(OrchestrationMode::Edit);
    grid.handleInput(LaunchpadGrid::kPresetRow, 0, true, 1000);
    ASSERT_EQ(grid.getActivePreset(), 0);
    grid.handleInput(kBrass, model(0), true, 2000);
    EXPECT_FALSE(grid.isActivePresetModified()) << "presets set your side in Step-in";
    grid.handleInput(kBrass, user(3), true, 3000);
    EXPECT_TRUE(grid.isActivePresetModified());
}

TEST(LaunchpadGrid, UnmappedPadIgnored) {
    LaunchpadGrid grid;
    grid.setSelectedMode(OrchestrationMode::Edit);
    CircularFifo<InstrumentUpdate> updates;
    grid.instrumentUpdates = &updates;
    mapAppInstruments(grid);
    grid.handleInput(kStrings, user(3), true, 1000); // gap after tremolo
    grid.handleInput(kPerc, user(1), true, 1000);    // only timpani in the percussion row
    EXPECT_FALSE(grid.getGridState(kStrings, user(3)));
    EXPECT_FALSE(grid.getGridState(kPerc, user(1)));
    EXPECT_TRUE(drain(updates).empty());
}

TEST(LaunchpadGrid, GridBurstIgnoresRapidRepress) {
    LaunchpadGrid grid;
    grid.setSelectedMode(OrchestrationMode::Edit);
    CircularFifo<InstrumentUpdate> updates;
    grid.instrumentUpdates = &updates;
    grid.padInstruments[kStrings][user(0)] = 6;

    grid.handleInput(kStrings, user(0), true, 5000);
    EXPECT_TRUE(grid.getGridState(kStrings, user(0)));

    grid.handleInput(kStrings, user(0), true, 5000 + LaunchpadGrid::kBurstMs - 1);
    EXPECT_TRUE(grid.getGridState(kStrings, user(0))); // ignored
    ASSERT_EQ(drain(updates).size(), 1u);

    grid.handleInput(kStrings, user(0), true, 5000 + LaunchpadGrid::kBurstMs);
    EXPECT_FALSE(grid.getGridState(kStrings, user(0)));
    const auto emitted = drain(updates);
    ASSERT_EQ(emitted.size(), 1u);
    EXPECT_EQ(emitted[0].state, 0);
}

TEST(LaunchpadGrid, ModePadsRequestModeAndLightSelection) {
    LaunchpadGrid grid;
    std::map<int, uint8_t> leds;
    grid.set_led = [&](int note, uint8_t colour) { leds[note] = colour; };
    std::vector<OrchestrationMode> requested;
    grid.mode_callback = [&](OrchestrationMode mode) { requested.push_back(mode); };

    grid.handleInput(LaunchpadGrid::kModeRow, LaunchpadGrid::kEditCol, true, 1000);
    grid.handleInput(LaunchpadGrid::kModeRow, LaunchpadGrid::kJamCol, true, 1000);
    grid.handleInput(LaunchpadGrid::kModeRow, LaunchpadGrid::kAccompanimentCol, true, 1000);
    grid.handleInput(LaunchpadGrid::kModeRow, 3, true, 1000); // not a mode pad
    ASSERT_EQ(requested.size(), 3u);
    EXPECT_EQ(requested[0], OrchestrationMode::Edit);
    EXPECT_EQ(requested[1], OrchestrationMode::Jam);
    EXPECT_EQ(requested[2], OrchestrationMode::Accompaniment);

    grid.setSelectedMode(OrchestrationMode::Edit);
    EXPECT_EQ(ledFor(leds, LaunchpadGrid::kModeRow, LaunchpadGrid::kEditCol),
              LaunchpadLighting::kWhite);
    EXPECT_EQ(ledFor(leds, LaunchpadGrid::kModeRow, LaunchpadGrid::kJamCol),
              LaunchpadLighting::kOff);

    grid.setSelectedMode(OrchestrationMode::Jam);
    EXPECT_EQ(ledFor(leds, LaunchpadGrid::kModeRow, LaunchpadGrid::kEditCol),
              LaunchpadLighting::kOff);
    EXPECT_EQ(ledFor(leds, LaunchpadGrid::kModeRow, LaunchpadGrid::kJamCol),
              LaunchpadLighting::kWhite);
}

TEST(LaunchpadGrid, TdLevelPressTurnsAutoOffAndSelectsLevel) {
    LaunchpadGrid grid;
    std::map<int, uint8_t> leds;
    grid.set_led = [&](int note, uint8_t colour) { leds[note] = colour; };
    restoreAutoTd(grid, 100);
    constexpr int row = LaunchpadGrid::kTdRow;

    grid.refreshAllLeds();
    EXPECT_EQ(ledFor(leds, row, LaunchpadGrid::kSideCol), LaunchpadLighting::kGreen);
    for (int col = 0; col < LaunchpadGrid::kTdLevelCount; ++col)
        EXPECT_EQ(ledFor(leds, row, col), LaunchpadLighting::kOff);

    grid.handleInput(row, 3, true, 1000);
    EXPECT_FALSE(TemporalDensity::isAuto());
    EXPECT_EQ(TemporalDensity::get(), 8);
    EXPECT_EQ(ledFor(leds, row, LaunchpadGrid::kSideCol), LaunchpadLighting::kRed);
    for (int col = 0; col < LaunchpadGrid::kTdLevelCount; ++col)
        EXPECT_EQ(ledFor(leds, row, col),
                  col == 3 ? LaunchpadLighting::kTdGradient[3] : LaunchpadLighting::kOff);

    grid.handleInput(row, 5, true, 1000);
    EXPECT_EQ(TemporalDensity::get(), 32);
    EXPECT_EQ(ledFor(leds, row, 3), LaunchpadLighting::kOff);
    EXPECT_EQ(ledFor(leds, row, 5), LaunchpadLighting::kTdGradient[5]);

    restoreAutoTd(grid, 2000);
}

TEST(LaunchpadGrid, theAppStartsAtTd4) {
    EXPECT_EQ(TemporalDensity::kDefault, 4);
    LaunchpadGrid grid;
    EXPECT_EQ(TemporalDensity::kLevels[static_cast<size_t>(grid.getManualTdLevelIndex())], 4);
}

TEST(LaunchpadGrid, AutoTdToggleRestoresLastManualLevel) {
    LaunchpadGrid grid;
    std::map<int, uint8_t> leds;
    grid.set_led = [&](int note, uint8_t colour) { leds[note] = colour; };
    restoreAutoTd(grid, 100);
    constexpr int row = LaunchpadGrid::kTdRow;

    // AUTO off with no level chosen yet → the default, TD 4.
    grid.handleInput(row, LaunchpadGrid::kSideCol, true, 1000);
    EXPECT_EQ(TemporalDensity::get(), TemporalDensity::kDefault);
    EXPECT_EQ(ledFor(leds, row, 2), LaunchpadLighting::kTdGradient[2]);

    grid.handleInput(row, 3, true, 1000);
    EXPECT_EQ(TemporalDensity::get(), 8);

    grid.handleInput(row, LaunchpadGrid::kSideCol, true, 2000); // AUTO on
    EXPECT_TRUE(TemporalDensity::isAuto());
    EXPECT_EQ(ledFor(leds, row, 3), LaunchpadLighting::kOff);
    EXPECT_EQ(ledFor(leds, row, LaunchpadGrid::kSideCol), LaunchpadLighting::kGreen);

    grid.handleInput(row, LaunchpadGrid::kSideCol, true, 2000 + 10); // burst: ignored
    EXPECT_TRUE(TemporalDensity::isAuto());

    grid.handleInput(row, LaunchpadGrid::kSideCol, true, 3000); // AUTO off → back to 8
    EXPECT_EQ(TemporalDensity::get(), 8);
    EXPECT_EQ(ledFor(leds, row, 3), LaunchpadLighting::kTdGradient[3]);

    restoreAutoTd(grid, 4000);
}

TEST(LaunchpadGrid, NonLevelTdPadsDoNothing) {
    LaunchpadGrid grid;
    restoreAutoTd(grid, 100);
    grid.handleInput(LaunchpadGrid::kTdRow, LaunchpadGrid::kTdLevelCount, true, 1000);
    grid.handleInput(LaunchpadGrid::kTdRow, LaunchpadGrid::kCols - 1, true, 1000);
    EXPECT_TRUE(TemporalDensity::isAuto());
}

TEST(LaunchpadGrid, FlushAddsYourInstrumentsToTheModelAndKeepsYours) {
    LaunchpadGrid grid;
    CircularFifo<InstrumentUpdate> updates;
    grid.instrumentUpdates = &updates;
    mapAppInstruments(grid);

    grid.setSelectedMode(OrchestrationMode::Jam);
    grid.handleInput(kStrings, model(1), true, 1000); // model already has pizzicato
    grid.setSelectedMode(OrchestrationMode::Edit);
    grid.handleInput(kStrings, user(0), true, 1000); // yours: string ensemble + flute
    grid.handleInput(kWinds, user(0), true, 1000);
    drain(updates);

    grid.handleInput(LaunchpadGrid::kFlushRow, LaunchpadGrid::kSideCol, true, 2000);

    EXPECT_TRUE(grid.getGridState(kStrings, model(0)));
    EXPECT_TRUE(grid.getGridState(kStrings, model(1))); // kept: flush adds, never replaces
    EXPECT_TRUE(grid.getGridState(kWinds, model(0)));
    EXPECT_TRUE(grid.getGridState(kStrings, user(0))); // yours stay on
    EXPECT_TRUE(grid.getGridState(kWinds, user(0)));

    const auto emitted = drain(updates);
    ASSERT_EQ(emitted.size(), 2u);
    for (const auto &u: emitted) {
        EXPECT_EQ(u.target, InstrumentUpdateTarget::Model);
        EXPECT_EQ(u.state, 1);
    }
}

TEST(LaunchpadGrid, FlushWithNothingOnDoesNothing) {
    LaunchpadGrid grid;
    grid.setSelectedMode(OrchestrationMode::Edit);
    CircularFifo<InstrumentUpdate> updates;
    grid.instrumentUpdates = &updates;
    mapAppInstruments(grid);

    grid.handleInput(LaunchpadGrid::kFlushRow, LaunchpadGrid::kSideCol, true, 1000);
    EXPECT_FALSE(grid.getGridState(kStrings, model(0)));
    EXPECT_TRUE(drain(updates).empty());
}

TEST(LaunchpadGrid, ResetClearsTheModelInstrumentsOnly) {
    LaunchpadGrid grid;
    CircularFifo<InstrumentUpdate> updates;
    grid.instrumentUpdates = &updates;
    mapAppInstruments(grid);
    grid.setSelectedMode(OrchestrationMode::Jam);
    grid.handleInput(kStrings, model(0), true, 1000);
    grid.handleInput(kBrass, model(2), true, 1000);
    grid.setSelectedMode(OrchestrationMode::Edit);
    grid.handleInput(kWinds, user(0), true, 1000);
    drain(updates);

    grid.handleInput(LaunchpadGrid::kResetRow, LaunchpadGrid::kSideCol, true, 2000);
    EXPECT_FALSE(grid.getGridState(kStrings, model(0)));
    EXPECT_FALSE(grid.getGridState(kBrass, model(2)));
    EXPECT_TRUE(grid.getGridState(kWinds, user(0)));
    const auto emitted = drain(updates);
    ASSERT_EQ(emitted.size(), 2u);
    for (const auto &u: emitted) {
        EXPECT_EQ(u.target, InstrumentUpdateTarget::Model);
        EXPECT_EQ(u.state, 0);
    }
}

TEST(LaunchpadGrid, FlushAndResetOnlyWorkAndLightInEdit) {
    LaunchpadGrid grid;
    std::map<int, uint8_t> leds;
    grid.set_led = [&](int note, uint8_t colour) { leds[note] = colour; };
    mapAppInstruments(grid);
    grid.setSelectedMode(OrchestrationMode::Edit);
    grid.handleInput(kStrings, user(0), true, 1000);
    grid.setSelectedMode(OrchestrationMode::Jam);
    grid.handleInput(kBrass, model(0), true, 1000);
    grid.refreshAllLeds();

    EXPECT_EQ(ledFor(leds, LaunchpadGrid::kFlushRow, LaunchpadGrid::kSideCol), LaunchpadLighting::kOff);
    EXPECT_EQ(ledFor(leds, LaunchpadGrid::kResetRow, LaunchpadGrid::kSideCol), LaunchpadLighting::kOff);
    grid.handleInput(LaunchpadGrid::kFlushRow, LaunchpadGrid::kSideCol, true, 2000);
    grid.handleInput(LaunchpadGrid::kResetRow, LaunchpadGrid::kSideCol, true, 2000);
    EXPECT_FALSE(grid.getGridState(kStrings, model(0))) << "no flush in Jam";
    EXPECT_TRUE(grid.getGridState(kBrass, model(0))) << "no reset in Jam";

    grid.setSelectedMode(OrchestrationMode::Edit);
    EXPECT_EQ(ledFor(leds, LaunchpadGrid::kFlushRow, LaunchpadGrid::kSideCol), LaunchpadLighting::kFlush);
    EXPECT_EQ(ledFor(leds, LaunchpadGrid::kResetRow, LaunchpadGrid::kSideCol), LaunchpadLighting::kReset);
}

TEST(LaunchpadGrid, LedsBrightForTheControlledSideGreyOrDimForTheOther) {
    LaunchpadGrid grid;
    std::map<int, uint8_t> leds;
    grid.set_led = [&](int note, uint8_t colour) { leds[note] = colour; };
    mapAppInstruments(grid);

    grid.setSelectedMode(OrchestrationMode::Edit);
    grid.handleInput(kStrings, user(0), true, 1000);
    grid.handleInput(kWinds, user(0), true, 1000);
    grid.handleInput(kBrass, user(0), true, 1000);
    grid.handleInput(kPerc, user(0), true, 1000);
    grid.handleInput(LaunchpadGrid::kFlushRow, LaunchpadGrid::kSideCol, true, 1000);
    grid.refreshAllLeds();

    // Edit: yours bright family colours, the model's dim.
    EXPECT_EQ(ledFor(leds, kStrings, user(0)), LaunchpadLighting::kYellow);
    EXPECT_EQ(ledFor(leds, kWinds, user(0)), LaunchpadLighting::kLime);
    EXPECT_EQ(ledFor(leds, kBrass, user(0)), LaunchpadLighting::kOrange);
    EXPECT_EQ(ledFor(leds, kPerc, user(0)), LaunchpadLighting::kMint);
    EXPECT_EQ(ledFor(leds, kStrings, model(0)),
              LaunchpadLighting::dimmed(LaunchpadLighting::kYellow, LaunchpadLighting::kOtherSideDimLevel));
    EXPECT_EQ(ledFor(leds, kStrings, user(1)), LaunchpadLighting::kOff); // mapped, off
    EXPECT_EQ(ledFor(leds, kStrings, user(3)), LaunchpadLighting::kOff); // unmapped

    // Jam: yours grey, the model's bright.
    grid.setSelectedMode(OrchestrationMode::Jam);
    EXPECT_EQ(ledFor(leds, kStrings, user(0)), LaunchpadLighting::kGrey);
    EXPECT_EQ(ledFor(leds, kBrass, user(0)), LaunchpadLighting::kGrey);
    EXPECT_EQ(ledFor(leds, kStrings, model(0)), LaunchpadLighting::kYellow);
    EXPECT_EQ(ledFor(leds, kBrass, model(0)), LaunchpadLighting::kOrange);
    EXPECT_EQ(ledFor(leds, kStrings, user(1)), LaunchpadLighting::kOff);

    EXPECT_EQ(ledFor(leds, 0, LaunchpadGrid::kSideCol), LaunchpadLighting::kOff) << "unused side pad";
    EXPECT_EQ(ledFor(leds, LaunchpadGrid::kRegisterRow, LaunchpadGrid::kSideCol),
              LaunchpadLighting::colourForAutoTd(RegisterRange::isAuto()))
        << "row 6 side = register AUTO";
}

TEST(LaunchpadGrid, PresetSetsTheControlledSideTdAndRegister) {
    LaunchpadGrid grid;
    CircularFifo<InstrumentUpdate> updates;
    grid.instrumentUpdates = &updates;
    mapAppInstruments(grid);
    std::vector<int> applied;
    grid.preset_callback = [&](int idx) { applied.push_back(idx); };
    const auto stormy = 6;
    const auto &preset = MoodPresets::all()[stormy];
    ASSERT_STREQ(preset.name, "Stormy");

    // Jam: the model's side becomes exactly the preset; yours is untouched.
    grid.setSelectedMode(OrchestrationMode::Edit);
    grid.handleInput(kWinds, user(0), true, 500);
    grid.setSelectedMode(OrchestrationMode::Jam);
    grid.handleInput(kWinds, model(0), true, 500); // flute: not in Stormy, switched off
    grid.handleInput(LaunchpadGrid::kPresetRow, stormy, true, 1000);

    ASSERT_EQ(applied.size(), 1u);
    EXPECT_EQ(applied[0], stormy);
    for (int row = kStrings; row <= kPerc; ++row)
        for (int c = 0; c < LaunchpadGrid::kColsPerSide; ++c) {
            const auto &id = grid.padInstruments[static_cast<size_t>(row)][static_cast<size_t>(model(c))];
            if (id.has_value())
                EXPECT_EQ(grid.getGridState(row, model(c)), MoodPresets::contains(preset, *id))
                    << "model pad " << row << "," << c;
        }
    EXPECT_TRUE(grid.getGridState(kWinds, user(0))) << "your side is not the controlled one";
    EXPECT_EQ(TemporalDensity::get(), TemporalDensity::kLevels[static_cast<size_t>(preset.tdLevelIndex)]);
    EXPECT_EQ(RegisterRange::get(), preset.registerValue);
    EXPECT_EQ(grid.getActivePreset(), stormy);
    EXPECT_FALSE(grid.isActivePresetModified());

    // Edit: the same preset sets your side.
    grid.setSelectedMode(OrchestrationMode::Edit);
    grid.handleInput(LaunchpadGrid::kPresetRow, stormy, true, 2000);
    EXPECT_FALSE(grid.getGridState(kWinds, user(0)));
    EXPECT_TRUE(grid.getGridState(kStrings, user(2))); // tremolo

    // Leave the process-wide TD / register as the other tests expect.
    grid.handleInput(LaunchpadGrid::kRegisterRow, LaunchpadGrid::kSideCol, true, 3000);
    restoreAutoTd(grid, 4000);
    ASSERT_TRUE(RegisterRange::isAuto());
}

TEST(LaunchpadGrid, PresetLedDimsWhenChangedAndAModeChangeClearsIt) {
    LaunchpadGrid grid;
    std::map<int, uint8_t> leds;
    grid.set_led = [&](int note, uint8_t colour) { leds[note] = colour; };
    mapAppInstruments(grid);
    grid.setSelectedMode(OrchestrationMode::Jam);
    const int tender = 2;
    const auto colour = MoodPresets::all()[tender].colour;

    grid.refreshAllLeds();
    EXPECT_EQ(ledFor(leds, LaunchpadGrid::kPresetRow, tender), LaunchpadLighting::dimmed(colour, 2));

    grid.handleInput(LaunchpadGrid::kPresetRow, tender, true, 1000);
    EXPECT_EQ(ledFor(leds, LaunchpadGrid::kPresetRow, tender), colour);

    grid.handleInput(kBrass, model(1), true, 1500); // trumpet by hand
    EXPECT_TRUE(grid.isActivePresetModified());
    EXPECT_EQ(ledFor(leds, LaunchpadGrid::kPresetRow, tender), LaunchpadLighting::dimmed(colour, 1));

    grid.handleInput(LaunchpadGrid::kPresetRow, tender, true, 2000); // re-applies it
    EXPECT_FALSE(grid.isActivePresetModified());
    EXPECT_FALSE(grid.getGridState(kBrass, model(1)));
    EXPECT_EQ(ledFor(leds, LaunchpadGrid::kPresetRow, tender), colour);

    grid.setSelectedMode(OrchestrationMode::Edit);
    EXPECT_EQ(grid.getActivePreset(), -1);
    EXPECT_EQ(ledFor(leds, LaunchpadGrid::kPresetRow, tender), LaunchpadLighting::dimmed(colour, 2));

    restoreAutoTd(grid, 3000);
}

TEST(LaunchpadGrid, PolyphonyPadsAreTheRightHalfOfRow6) {
    EXPECT_EQ(LaunchpadGrid::kAccompRow, LaunchpadGrid::kRegisterRow);
    EXPECT_FALSE(LaunchpadGrid::polyphonyForPad(LaunchpadGrid::kAccompRow, 0).has_value());
    EXPECT_FALSE(LaunchpadGrid::polyphonyForPad(LaunchpadGrid::kAccompRow, 3).has_value());
    for (int i = 0; i < LaunchpadGrid::kPolyphonyPadCount; ++i)
        EXPECT_EQ(LaunchpadGrid::polyphonyForPad(LaunchpadGrid::kAccompRow, LaunchpadGrid::kPolyphonyColFirst + i),
                  i + 1);
    EXPECT_TRUE(LaunchpadGrid::registerForPad(LaunchpadGrid::kRegisterRow, 2).has_value());
}

TEST(MoodPresets, InstrumentsAreMappedAndValuesValid) {
    const std::set<int32_t> mapped{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16};
    for (const auto &preset: MoodPresets::all()) {
        EXPECT_FALSE(preset.instruments.empty()) << preset.name;
        for (const auto id: preset.instruments)
            EXPECT_EQ(mapped.count(id), 1u) << preset.name << " uses unmapped instrument " << id;
        EXPECT_GE(preset.tdLevelIndex, -1) << preset.name;
        EXPECT_LT(preset.tdLevelIndex, LaunchpadGrid::kTdLevelCount) << preset.name;
        EXPECT_TRUE(RegisterRange::isValid(preset.registerValue)) << preset.name;
    }
}

TEST(LaunchpadLighting, DimmedShades) {
    using LaunchpadLighting::dimmed;
    EXPECT_EQ(dimmed(LaunchpadLighting::kYellow, 1), LaunchpadLighting::kYellow + 1);
    EXPECT_EQ(dimmed(LaunchpadLighting::kYellow, 2), LaunchpadLighting::kYellow + 2);
    EXPECT_EQ(dimmed(LaunchpadLighting::kRed, 2), LaunchpadLighting::kRed + 2);
    EXPECT_EQ(dimmed(LaunchpadLighting::kWhite, 1), 2);
    EXPECT_EQ(dimmed(LaunchpadLighting::kWhite, 2), 1);
    EXPECT_EQ(dimmed(LaunchpadLighting::kOff, 2), LaunchpadLighting::kOff);
}

TEST(LaunchpadGrid, SceneCallbackByIndex) {
    LaunchpadGrid grid;
    std::vector<int> idxs;
    grid.scene_callback = [&](int idx) { idxs.push_back(idx); };

    grid.handleSceneInput(3);
    grid.handleSceneInput(-1);
    grid.handleSceneInput(8);
    ASSERT_EQ(idxs.size(), 1u);
    EXPECT_EQ(idxs[0], 3);
}

TEST(LaunchpadLighting, FamilyColoursByTaxonomyRange) {
    using LaunchpadLighting::colourForInstrumentPad;
    EXPECT_EQ(colourForInstrumentPad(0, true), LaunchpadLighting::kYellow);
    EXPECT_EQ(colourForInstrumentPad(6, true), LaunchpadLighting::kYellow);
    EXPECT_EQ(colourForInstrumentPad(7, true), LaunchpadLighting::kLime);
    EXPECT_EQ(colourForInstrumentPad(10, true), LaunchpadLighting::kLime);
    EXPECT_EQ(colourForInstrumentPad(11, true), LaunchpadLighting::kOrange);
    EXPECT_EQ(colourForInstrumentPad(14, true), LaunchpadLighting::kOrange);
    EXPECT_EQ(colourForInstrumentPad(16, true), LaunchpadLighting::kMint);
    EXPECT_EQ(colourForInstrumentPad(17, true), LaunchpadLighting::kMint);
    EXPECT_EQ(colourForInstrumentPad(6, false), LaunchpadLighting::kOff);
    EXPECT_EQ(colourForInstrumentPad(42, true), LaunchpadLighting::kOff);
}

TEST(LaunchpadLighting, TdAndAutoColours) {
    EXPECT_EQ(LaunchpadLighting::colourForTdLevel(0, true), LaunchpadLighting::kCyan);
    EXPECT_EQ(LaunchpadLighting::colourForTdLevel(5, true), LaunchpadLighting::kPink);
    EXPECT_EQ(LaunchpadLighting::colourForTdLevel(2, false), LaunchpadLighting::kOff);
    EXPECT_EQ(LaunchpadLighting::colourForTdLevel(6, true), LaunchpadLighting::kRed); // TD 64
    EXPECT_EQ(LaunchpadLighting::colourForTdLevel(7, true), LaunchpadLighting::kOff);
    EXPECT_EQ(LaunchpadLighting::colourForAutoTd(true), LaunchpadLighting::kGreen);
    EXPECT_EQ(LaunchpadLighting::colourForAutoTd(false), LaunchpadLighting::kRed);
}

TEST(TemporalDensity, StartsAutoAndKnowsValidLevels) {
    EXPECT_TRUE(TemporalDensity::isValid(TemporalDensity::kAuto));
    for (const auto level: TemporalDensity::kLevels)
        EXPECT_TRUE(TemporalDensity::isValid(level));
    EXPECT_FALSE(TemporalDensity::isValid(3));
    EXPECT_TRUE(TemporalDensity::isValid(64));
    EXPECT_FALSE(TemporalDensity::isValid(128));
}

} // namespace virtual_orch_test
