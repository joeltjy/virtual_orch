#include <gtest/gtest.h>

#include <VirtualOrch/LaunchpadGrid.h>
#include <VirtualOrch/LaunchpadLighting.h>
#include <VirtualOrch/LaunchpadProgrammerMap.h>
#include <VirtualOrch/RegisterRange.h>
#include <VirtualOrch/TemporalDensity.h>

#include <map>
#include <vector>

namespace virtual_orch_test {

namespace {

constexpr int kUser = LaunchpadGrid::kUserRowFirst;
constexpr int kModel = LaunchpadGrid::kModelRowFirst;

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
    CircularFifo<InstrumentUpdate> updates;
    grid.instrumentUpdates = &updates;
    grid.padInstruments[kUser][5] = 7;

    grid.handleInput(kUser, 5, true, 1000);
    InstrumentUpdate u{};
    ASSERT_TRUE(updates.pull(u));
    EXPECT_EQ(u.localInstrumentId, 7);
    EXPECT_EQ(u.target, InstrumentUpdateTarget::User);
    EXPECT_EQ(u.state, 1);
    EXPECT_FALSE(updates.pull(u));

    grid.handleInput(kUser, 5, false, 1000); // release ignored
    EXPECT_FALSE(updates.pull(u));

    grid.handleInput(kUser, 5, true, 1000 + LaunchpadGrid::kBurstMs);
    ASSERT_TRUE(updates.pull(u));
    EXPECT_EQ(u.state, 0);
}

TEST(LaunchpadGrid, InputRowsAreUserModelRowsAreModel) {
    EXPECT_FALSE(LaunchpadGrid::targetForRow(LaunchpadGrid::kModeRow).has_value());
    EXPECT_FALSE(LaunchpadGrid::targetForRow(LaunchpadGrid::kTdRow).has_value());
    EXPECT_EQ(LaunchpadGrid::targetForRow(kUser), InstrumentUpdateTarget::User);
    EXPECT_EQ(LaunchpadGrid::targetForRow(kUser + 1), InstrumentUpdateTarget::User);
    EXPECT_EQ(LaunchpadGrid::targetForRow(kModel), InstrumentUpdateTarget::Model);
    EXPECT_EQ(LaunchpadGrid::targetForRow(kModel + 1), InstrumentUpdateTarget::Model);
    EXPECT_FALSE(LaunchpadGrid::targetForRow(6).has_value());
    EXPECT_FALSE(LaunchpadGrid::targetForRow(7).has_value());

    LaunchpadGrid grid;
    CircularFifo<InstrumentUpdate> updates;
    grid.instrumentUpdates = &updates;
    grid.padInstruments[kUser + 1][0] = 12;
    grid.padInstruments[kModel + 1][0] = 12;

    grid.handleInput(kUser + 1, 0, true, 1000);
    grid.handleInput(kModel + 1, 0, true, 1000);
    const auto emitted = drain(updates);
    ASSERT_EQ(emitted.size(), 2u);
    EXPECT_EQ(emitted[0].target, InstrumentUpdateTarget::User);
    EXPECT_EQ(emitted[1].target, InstrumentUpdateTarget::Model);
}

TEST(LaunchpadGrid, UnmappedPadIgnored) {
    LaunchpadGrid grid;
    CircularFifo<InstrumentUpdate> updates;
    grid.instrumentUpdates = &updates;
    grid.handleInput(6, 0, true, 1000);
    grid.handleInput(kUser, 3, true, 1000); // gap in the input layout
    EXPECT_FALSE(grid.getGridState(6, 0));
    EXPECT_FALSE(grid.getGridState(kUser, 3));
    EXPECT_TRUE(drain(updates).empty());
}

TEST(LaunchpadGrid, GridBurstIgnoresRapidRepress) {
    LaunchpadGrid grid;
    CircularFifo<InstrumentUpdate> updates;
    grid.instrumentUpdates = &updates;
    grid.padInstruments[kUser][0] = 6;

    grid.handleInput(kUser, 0, true, 5000);
    EXPECT_TRUE(grid.getGridState(kUser, 0));

    grid.handleInput(kUser, 0, true, 5000 + LaunchpadGrid::kBurstMs - 1);
    EXPECT_TRUE(grid.getGridState(kUser, 0)); // ignored
    ASSERT_EQ(drain(updates).size(), 1u);

    grid.handleInput(kUser, 0, true, 5000 + LaunchpadGrid::kBurstMs);
    EXPECT_FALSE(grid.getGridState(kUser, 0));
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

TEST(LaunchpadGrid, FlushAddsInputToModelThenClearsInput) {
    LaunchpadGrid grid;
    CircularFifo<InstrumentUpdate> updates;
    grid.instrumentUpdates = &updates;
    for (const int base: {kUser, kModel}) {
        grid.padInstruments[base][0] = 6;
        grid.padInstruments[base][1] = 5;
        grid.padInstruments[base][4] = 7;
    }

    grid.handleInput(kModel, 1, true, 1000); // model already has pizzicato
    grid.handleInput(kUser, 0, true, 1000);  // input: string ensemble + flute
    grid.handleInput(kUser, 4, true, 1000);
    drain(updates);

    grid.handleInput(LaunchpadGrid::kFlushRow, LaunchpadGrid::kSideCol, true, 2000);

    EXPECT_TRUE(grid.getGridState(kModel, 0));
    EXPECT_TRUE(grid.getGridState(kModel, 1)); // kept: flush adds, never replaces
    EXPECT_TRUE(grid.getGridState(kModel, 4));
    EXPECT_FALSE(grid.getGridState(kUser, 0));
    EXPECT_FALSE(grid.getGridState(kUser, 1));
    EXPECT_FALSE(grid.getGridState(kUser, 4));

    const auto emitted = drain(updates);
    ASSERT_EQ(emitted.size(), 4u);
    // Model on first, then input off.
    EXPECT_EQ(emitted[0].target, InstrumentUpdateTarget::Model);
    EXPECT_EQ(emitted[0].state, 1);
    EXPECT_EQ(emitted[1].target, InstrumentUpdateTarget::Model);
    EXPECT_EQ(emitted[1].state, 1);
    EXPECT_EQ(emitted[2].target, InstrumentUpdateTarget::User);
    EXPECT_EQ(emitted[2].state, 0);
    EXPECT_EQ(emitted[3].target, InstrumentUpdateTarget::User);
    EXPECT_EQ(emitted[3].state, 0);
}

TEST(LaunchpadGrid, FlushWithEmptyInputDoesNothing) {
    LaunchpadGrid grid;
    CircularFifo<InstrumentUpdate> updates;
    grid.instrumentUpdates = &updates;
    grid.padInstruments[kUser][0] = 6;
    grid.padInstruments[kModel][0] = 6;

    grid.handleInput(LaunchpadGrid::kFlushRow, LaunchpadGrid::kSideCol, true, 1000);
    EXPECT_FALSE(grid.getGridState(kModel, 0));
    EXPECT_TRUE(drain(updates).empty());
}

TEST(LaunchpadGrid, LedsFollowFamilyColoursAndSideButtons) {
    LaunchpadGrid grid;
    std::map<int, uint8_t> leds;
    grid.set_led = [&](int note, uint8_t colour) { leds[note] = colour; };
    grid.padInstruments[kUser][0] = 6;  // strings
    grid.padInstruments[kUser][4] = 7;  // woodwinds
    grid.padInstruments[kUser + 1][0] = 12; // brass
    grid.padInstruments[kUser + 1][4] = 16; // other

    grid.refreshAllLeds();
    EXPECT_EQ(ledFor(leds, kUser, 0), LaunchpadLighting::kOff);
    EXPECT_EQ(ledFor(leds, LaunchpadGrid::kFlushRow, LaunchpadGrid::kSideCol),
              LaunchpadLighting::kFlush);
    EXPECT_EQ(ledFor(leds, 0, LaunchpadGrid::kSideCol), LaunchpadLighting::kOff) << "unused side pad";
    EXPECT_EQ(ledFor(leds, LaunchpadGrid::kRegisterRow, LaunchpadGrid::kSideCol),
              LaunchpadLighting::colourForAutoTd(RegisterRange::isAuto()))
        << "row 6 side = register AUTO";

    grid.handleInput(kUser, 0, true, 1000);
    grid.handleInput(kUser, 4, true, 1000);
    grid.handleInput(kUser + 1, 0, true, 1000);
    grid.handleInput(kUser + 1, 4, true, 1000);
    EXPECT_EQ(ledFor(leds, kUser, 0), LaunchpadLighting::kYellow);
    EXPECT_EQ(ledFor(leds, kUser, 4), LaunchpadLighting::kLime);
    EXPECT_EQ(ledFor(leds, kUser + 1, 0), LaunchpadLighting::kOrange);
    EXPECT_EQ(ledFor(leds, kUser + 1, 4), LaunchpadLighting::kMint);
    EXPECT_EQ(ledFor(leds, kUser, 3), LaunchpadLighting::kOff); // unmapped
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
