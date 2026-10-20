#include <gtest/gtest.h>

#include <map>

#include <VirtualOrch/Clock.h>
#include <VirtualOrch/LaunchpadGrid.h>
#include <VirtualOrch/Metrics.h>
#include <VirtualOrch/reduction/ReductionTransformer.h>

namespace {

/** Reduction stub exposing what the polyphony cap uses. */
class StubReduction : public ReductionTransformer {
public:
    explicit StubReduction(ModelConfig &config) : ReductionTransformer("stub", config) {}
    [[nodiscard]] auto isModelLoaded() const -> bool override { return true; }
    void run() override {}
    using ReductionTransformer::onsetLowerBoundCs;
    using ReductionTransformer::pushOutputToken;
    auto setTime(int32_t t) -> void { currentTime = t; }
};

auto note(int32_t onset, int32_t durCs, int32_t pitch = 60) -> Token {
    return Token{onset, static_cast<int32_t>(Vocab::DurOffset) + durCs,
                 static_cast<int32_t>(Vocab::NoteOffset) + pitch};
}

struct GridFixture {
    LaunchpadGrid grid;
    std::map<int, uint8_t> leds;
    int callbacks = 0;
    uint32_t now = 1000;

    GridFixture() {
        grid.set_led = [this](int midiNote, uint8_t colour) { leds[midiNote] = colour; };
        grid.accompaniment_callback = [this] { ++callbacks; };
        grid.setSelectedMode(OrchestrationMode::Jam);
        grid.refreshAllLeds();
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

TEST(AccompanimentGrid, startsWithNoCapAndAccompanimentOff) {
    GridFixture f;
    EXPECT_FALSE(f.grid.isAccompanimentOn());
    EXPECT_EQ(f.grid.getMaxPolyphony(), 0) << "no cap until a pad is pressed";
    EXPECT_EQ(f.grid.effectivePolyphonyCap(), 0);
    EXPECT_EQ(f.colourOf(7, LaunchpadGrid::kSideCol), LaunchpadLighting::kRed) << "Jam, off";
    for (int col = 0; col < 4; ++col)
        EXPECT_EQ(f.colourOf(7, col), LaunchpadLighting::kOff);
}

TEST(AccompanimentGrid, polyphonyPadsWorkInJamModeWithoutAccompaniment) {
    GridFixture f;
    f.press(7, 2);
    EXPECT_FALSE(f.grid.isAccompanimentOn());
    EXPECT_EQ(f.grid.getMaxPolyphony(), 3);
    EXPECT_EQ(f.grid.effectivePolyphonyCap(), 3) << "the cap applies in plain Jam mode";
    EXPECT_EQ(f.callbacks, 1);
    EXPECT_NE(f.colourOf(7, 2), LaunchpadLighting::kOff);
    EXPECT_EQ(f.colourOf(7, 0), LaunchpadLighting::kOff);

    f.press(7, 0);
    EXPECT_EQ(f.grid.effectivePolyphonyCap(), 1);
    f.press(7, 0); // pressing the selected pad again clears the cap
    EXPECT_EQ(f.grid.getMaxPolyphony(), 0);
    EXPECT_EQ(f.grid.effectivePolyphonyCap(), 0);
    EXPECT_EQ(f.colourOf(7, 0), LaunchpadLighting::kOff);
    EXPECT_EQ(f.callbacks, 3);
}

TEST(AccompanimentGrid, accompanimentStartsAtOneUnlessACapWasAlreadyChosen) {
    {
        GridFixture f;
        f.press(7, LaunchpadGrid::kSideCol);
        EXPECT_TRUE(f.grid.isAccompanimentActive());
        EXPECT_EQ(f.grid.getMaxPolyphony(), 1);
        EXPECT_EQ(f.grid.effectivePolyphonyCap(), 1);
        EXPECT_EQ(f.colourOf(7, LaunchpadGrid::kSideCol), LaunchpadLighting::kGreen);
        EXPECT_NE(f.colourOf(7, 0), LaunchpadLighting::kOff) << "pad (7,0) = polyphony 1 lit";
    }
    {
        GridFixture f;
        f.press(7, 2); // a cap chosen in plain Jam
        f.press(7, LaunchpadGrid::kSideCol);
        EXPECT_TRUE(f.grid.isAccompanimentOn());
        EXPECT_EQ(f.grid.getMaxPolyphony(), 3) << "kept, not reset to 1";
        f.press(7, LaunchpadGrid::kSideCol); // accompaniment off again: the cap stays in Jam
        EXPECT_FALSE(f.grid.isAccompanimentOn());
        EXPECT_EQ(f.grid.effectivePolyphonyCap(), 3);
    }
}

TEST(AccompanimentGrid, editModeDisablesAndJamRestores) {
    GridFixture f;
    f.press(7, LaunchpadGrid::kSideCol);
    f.press(7, 2);
    ASSERT_EQ(f.grid.getMaxPolyphony(), 3);

    f.grid.setSelectedMode(OrchestrationMode::Edit);
    EXPECT_FALSE(f.grid.isAccompanimentActive());
    EXPECT_EQ(f.grid.effectivePolyphonyCap(), 0) << "no cap in Edit mode";
    EXPECT_EQ(f.colourOf(7, LaunchpadGrid::kSideCol), LaunchpadLighting::kOff) << "not lit in Edit";
    EXPECT_EQ(f.colourOf(7, 2), LaunchpadLighting::kOff);

    const int before = f.callbacks;
    f.press(7, LaunchpadGrid::kSideCol); // nothing in Edit
    f.press(7, 1);
    EXPECT_EQ(f.callbacks, before);
    EXPECT_TRUE(f.grid.isAccompanimentOn());
    EXPECT_EQ(f.grid.getMaxPolyphony(), 3);

    f.grid.setSelectedMode(OrchestrationMode::Jam);
    EXPECT_TRUE(f.grid.isAccompanimentActive());
    EXPECT_EQ(f.grid.effectivePolyphonyCap(), 3) << "restored";
    EXPECT_EQ(f.colourOf(7, LaunchpadGrid::kSideCol), LaunchpadLighting::kGreen);
    EXPECT_NE(f.colourOf(7, 2), LaunchpadLighting::kOff);
}

TEST(AccompanimentReduction, capRaisesTheNextOnsetToTheRightOffset) {
    ModelConfig config;
    StubReduction rt(config);
    rt.pushOutputToken(note(0, 100));  // ends 100
    rt.pushOutputToken(note(20, 60));  // ends 80
    rt.setTime(50);

    EXPECT_EQ(rt.onsetLowerBoundCs(), 50) << "no cap";
    rt.setMaxPolyphony(3);
    EXPECT_EQ(rt.onsetLowerBoundCs(), 50) << "room left";
    rt.setMaxPolyphony(2);
    EXPECT_EQ(rt.onsetLowerBoundCs(), 80) << "the next offset";
    rt.setMaxPolyphony(1);
    EXPECT_EQ(rt.onsetLowerBoundCs(), 100) << "two sounding, cap 1: the 2nd earliest offset";
    rt.setMaxPolyphony(0);
    EXPECT_EQ(rt.onsetLowerBoundCs(), 50);
}

} // namespace virtual_orch_test
