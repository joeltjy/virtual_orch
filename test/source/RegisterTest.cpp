#include <gtest/gtest.h>

#include <cmath>
#include <map>

#include <VirtualOrch/LaunchpadGrid.h>
#include <VirtualOrch/PerformanceActivity.h>
#include <VirtualOrch/RegisterRange.h>
#include <VirtualOrch/reduction/AnticipationSampler.h>
#include <VirtualOrch/reduction/DenseTypes.h>

namespace {

struct Pads {
    LaunchpadGrid grid;
    std::map<int, uint8_t> leds;
    uint32_t now = 1000;
    Pads() {
        grid.set_led = [this](int note, uint8_t colour) { leds[note] = colour; };
        grid.setSelectedMode(OrchestrationMode::Jam);
        grid.refreshAllLeds();
    }
    ~Pads() {
        // the register is global: leave it as the default
        if (! RegisterRange::isAuto())
            press(LaunchpadGrid::kRegisterRow, LaunchpadGrid::kSideCol);
    }
    auto press(int row, int col) -> void {
        now += 500;
        grid.handleInput(row, col, true, now);
    }
    auto colourOf(int row, int col) -> uint8_t {
        return leds.at(*LaunchpadProgrammerMap::noteFromPad(row, col));
    }
};

} // namespace

namespace virtual_orch_test {

TEST(RegisterRange, rangesFollowTheSpecWithC4AsMidi60) {
    EXPECT_EQ(RegisterRange::lowPitch(RegisterRange::kLow), 21);
    EXPECT_EQ(RegisterRange::highPitch(RegisterRange::kLow), 52) << "E3";
    EXPECT_EQ(RegisterRange::lowPitch(RegisterRange::kMedium), 55) << "G3";
    EXPECT_EQ(RegisterRange::highPitch(RegisterRange::kMedium), 74) << "D5";
    EXPECT_EQ(RegisterRange::lowPitch(RegisterRange::kHigh), 77) << "F5";
    EXPECT_EQ(RegisterRange::highPitch(RegisterRange::kHigh), 108);
    EXPECT_EQ(RegisterRange::lowPitch(RegisterRange::kAuto), 21);
    EXPECT_EQ(RegisterRange::highPitch(RegisterRange::kAuto), 108);
}

TEST(RegisterRange, maskNoteBlockLimitsEveryInstrumentBand) {
    Pads pads;
    pads.press(LaunchpadGrid::kRegisterRow, 0); // LOW
    ASSERT_EQ(RegisterRange::get(), RegisterRange::kLow);
    std::vector<float> logits(11000 + 5 * 128 + 200, 0.0f);
    RegisterRange::maskNoteBlock(logits, 11000, 5);
    for (int instr = 0; instr < 5; ++instr)
        for (int pitch = 0; pitch < 128; ++pitch)
            EXPECT_EQ(std::isfinite(logits[11000 + instr * 128 + pitch]), pitch >= 21 && pitch <= 52)
                << instr << "/" << pitch;
    EXPECT_TRUE(std::isfinite(logits[0])) << "ids outside the note block are untouched";
    EXPECT_TRUE(std::isfinite(logits.back()));
}

TEST(RegisterRange, maskNoteBlockLeavesTheLogitsAloneWhenNothingWouldBeLeft) {
    Pads pads;
    pads.press(LaunchpadGrid::kRegisterRow, 2); // HIGH: 77..108
    std::vector<float> logits(11000 + 128, -INFINITY);
    for (int pitch = 36; pitch <= 60; ++pitch) // the model may only play 36..60 here
        logits[11000 + pitch] = 0.0f;
    RegisterRange::maskNoteBlock(logits, 11000, 1);
    EXPECT_TRUE(std::isfinite(logits[11000 + 50])) << "the register conflicts: generate anyway";
}

TEST(RegisterRange, autoDoesNotRestrictAnything) {
    EXPECT_TRUE(RegisterRange::isAuto());
    std::vector<float> logits(11000 + 128, 0.0f);
    RegisterRange::maskNoteBlock(logits, 11000, 1);
    for (int pitch = 0; pitch < 128; ++pitch)
        EXPECT_TRUE(std::isfinite(logits[11000 + pitch]));
}

TEST(RegisterGrid, padsSelectTheRegisterAndAutoRestoresTheLastOne) {
    Pads pads;
    EXPECT_TRUE(RegisterRange::isAuto());
    EXPECT_EQ(pads.colourOf(LaunchpadGrid::kRegisterRow, LaunchpadGrid::kSideCol),
              LaunchpadLighting::kGreen) << "AUTO on";
    for (int col = 0; col < 3; ++col)
        EXPECT_EQ(pads.colourOf(6, col), LaunchpadLighting::kOff);

    pads.press(6, 0);
    EXPECT_EQ(RegisterRange::get(), RegisterRange::kLow);
    EXPECT_NE(pads.colourOf(6, 0), LaunchpadLighting::kOff);
    EXPECT_EQ(pads.colourOf(6, 1), LaunchpadLighting::kOff);
    EXPECT_EQ(pads.colourOf(6, LaunchpadGrid::kSideCol), LaunchpadLighting::kRed) << "AUTO off";

    pads.press(6, 2);
    EXPECT_EQ(RegisterRange::get(), RegisterRange::kHigh);

    pads.press(6, LaunchpadGrid::kSideCol); // AUTO on: no limit
    EXPECT_TRUE(RegisterRange::isAuto());
    for (int col = 0; col < 3; ++col)
        EXPECT_EQ(pads.colourOf(6, col), LaunchpadLighting::kOff);
    pads.press(6, LaunchpadGrid::kSideCol); // AUTO off: back to the last selection
    EXPECT_EQ(RegisterRange::get(), RegisterRange::kHigh);
}

TEST(RegisterGrid, worksInEditModeToo) {
    Pads pads;
    pads.grid.setSelectedMode(OrchestrationMode::Edit);
    pads.press(6, 1);
    EXPECT_EQ(RegisterRange::get(), RegisterRange::kMedium);
}

TEST(RegisterGrid, thePressesAreLoggedInWords) {
    using K = PerformanceActivity::LaunchpadAction::Kind;
    const auto name = [](int32_t) { return juce::String(); };
    EXPECT_EQ(PerformanceActivity::describe({K::Register, 0, 1, false, false}, name),
              "Register LOW (A0 to E3)");
    EXPECT_EQ(PerformanceActivity::describe({K::Register, 0, 2, false, false}, name),
              "Register MEDIUM (G3 to D5)");
    EXPECT_EQ(PerformanceActivity::describe({K::Register, 0, 3, false, false}, name),
              "Register HIGH (F5 to C8)");
    EXPECT_EQ(PerformanceActivity::describe({K::RegisterAuto, 0, 0, true, false}, name),
              "Register AUTO on (no limit)");
    EXPECT_EQ(PerformanceActivity::describe({K::Polyphony, 0, 0, false, false}, name),
              "Max polyphony off");

    Pads pads;
    std::vector<juce::String> lines;
    pads.grid.action_callback = [&](const PerformanceActivity::LaunchpadAction &a) {
        lines.push_back(PerformanceActivity::describe(a, name));
    };
    pads.press(6, 0);
    pads.press(6, LaunchpadGrid::kSideCol);
    ASSERT_EQ(lines.size(), 2u);
    EXPECT_EQ(lines[0], "Register LOW (A0 to E3)");
    EXPECT_EQ(lines[1], "Register AUTO on (no limit)");
}

TEST(RegisterSampler, theShapedPitchMaskFollowsTheRegisterBounds) {
    AnticipationSampler s(AnticipationSampler::kDenseVelocityBase);
    AnticipationSampler::Context ctx;
    ctx.pitchLow = 73;
    ctx.pitchHigh = 108;
    const std::vector<float> logits(11776, 0.0f);
    const auto note = s.masked(2, logits, ctx);
    for (size_t i = 0; i < note.size(); ++i) {
        const bool legal = i >= 11128 + 73 && i <= 11128 + 108;
        EXPECT_EQ(std::isfinite(note[i]), legal) << i;
    }
    for (int n = 0; n < 300; ++n) {
        const auto p = s.sample(2, logits, ctx, 1.0f, 1.0f);
        EXPECT_GE(p - 11128, 73);
        EXPECT_LE(p - 11128, 108);
    }
    // an impossible range falls back to the piano range
    ctx.pitchLow = 120;
    ctx.pitchHigh = 125;
    const auto fallback = s.masked(2, logits, ctx);
    EXPECT_TRUE(std::isfinite(fallback[11128 + 60]));
}

} // namespace virtual_orch_test
