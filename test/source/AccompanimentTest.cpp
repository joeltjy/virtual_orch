#include <gtest/gtest.h>

#include <map>

#include <VirtualOrch/Clock.h>
#include <VirtualOrch/LaunchpadGrid.h>
#include <VirtualOrch/Metrics.h>
#include <VirtualOrch/PedalClear.h>
#include <VirtualOrch/reduction/ReductionTransformer.h>

namespace {

/** Reduction stub exposing what the polyphony cap and the mode cut use. */
class StubReduction : public ReductionTransformer {
public:
    explicit StubReduction(ModelConfig &config) : ReductionTransformer("stub", config) {}
    [[nodiscard]] auto isModelLoaded() const -> bool override { return true; }
    void run() override {}
    using ReductionTransformer::consumeModeCut;
    using ReductionTransformer::onsetLowerBoundCs;
    using ReductionTransformer::pushOutputToken;
    using ReductionTransformer::rewindInputDataFrom;
    auto setTime(int32_t t) -> void { currentTime = t; }
    [[nodiscard]] auto time() const -> int32_t { return currentTime; }
    auto data() -> std::vector<int32_t> & { return inputData; }
};

auto note(int32_t onset, int32_t durCs, int32_t pitch = 60) -> Token {
    return Token{onset, static_cast<int32_t>(Vocab::DurOffset) + durCs,
                 static_cast<int32_t>(Vocab::NoteOffset) + pitch};
}

struct GridFixture {
    LaunchpadGrid grid;
    std::map<int, uint8_t> leds;
    std::vector<OrchestrationMode> requested;
    int callbacks = 0;
    uint32_t now = 1000;

    GridFixture() {
        grid.set_led = [this](int midiNote, uint8_t colour) { leds[midiNote] = colour; };
        grid.accompaniment_callback = [this] { ++callbacks; };
        grid.mode_callback = [this](OrchestrationMode mode) {
            requested.push_back(mode);
            grid.setSelectedMode(mode); // what AppSession::setOrchestrationMode does
        };
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

constexpr int kRow = LaunchpadGrid::kAccompRow;

} // namespace

namespace virtual_orch_test {

TEST(AccompanimentMode, isJamLikeForOrchestration) {
    EXPECT_FALSE(isJamLike(OrchestrationMode::Edit));
    EXPECT_TRUE(isJamLike(OrchestrationMode::Jam));
    EXPECT_TRUE(isJamLike(OrchestrationMode::Accompaniment));
}

TEST(AccompanimentGrid, modePadAtColumnTwoSelectsAccompaniment) {
    GridFixture f;
    EXPECT_EQ(LaunchpadGrid::modeForPad(LaunchpadGrid::kModeRow, 2), OrchestrationMode::Accompaniment);
    f.press(LaunchpadGrid::kModeRow, LaunchpadGrid::kAccompanimentCol);
    ASSERT_EQ(f.requested.size(), 1u);
    EXPECT_EQ(f.requested[0], OrchestrationMode::Accompaniment);
    EXPECT_EQ(f.colourOf(LaunchpadGrid::kModeRow, LaunchpadGrid::kAccompanimentCol),
              LaunchpadLighting::kWhite);
    EXPECT_EQ(f.colourOf(LaunchpadGrid::kModeRow, LaunchpadGrid::kJamCol), LaunchpadLighting::kOff);
}

TEST(AccompanimentGrid, bottomRowIsOffAndInertOutsideAccompaniment) {
    for (const auto mode: {OrchestrationMode::Edit, OrchestrationMode::Jam}) {
        GridFixture f;
        f.grid.setSelectedMode(mode);
        EXPECT_FALSE(f.grid.isAccompanimentActive());
        EXPECT_EQ(f.grid.effectivePolyphonyCap(), 0);
        f.press(kRow, 2);
        EXPECT_EQ(f.grid.getMaxPolyphony(), 1) << "pads do nothing outside Accompaniment";
        EXPECT_EQ(f.callbacks, 0);
        for (int col = 0; col < LaunchpadGrid::kPolyphonyPadCount; ++col)
            EXPECT_EQ(f.colourOf(kRow, col), LaunchpadLighting::kOff);
        EXPECT_EQ(f.colourOf(kRow, LaunchpadGrid::kSideCol), LaunchpadLighting::kOff)
            << "the old side toggle is gone";
    }
}

TEST(AccompanimentGrid, inAccompanimentPadsSelectThePolyphony) {
    GridFixture f;
    f.grid.setSelectedMode(OrchestrationMode::Accompaniment);
    EXPECT_TRUE(f.grid.isAccompanimentActive());
    EXPECT_EQ(f.grid.effectivePolyphonyCap(), 1) << "default 1";
    EXPECT_EQ(f.colourOf(kRow, 0), LaunchpadLighting::kYellow);
    EXPECT_EQ(f.colourOf(kRow, 1), LaunchpadLighting::kOff);

    f.press(kRow, 3);
    EXPECT_EQ(f.grid.getMaxPolyphony(), 4);
    EXPECT_EQ(f.grid.effectivePolyphonyCap(), 4);
    EXPECT_EQ(f.callbacks, 1);
    EXPECT_EQ(f.colourOf(kRow, 0), LaunchpadLighting::kOff);
    EXPECT_EQ(f.colourOf(kRow, 3), LaunchpadLighting::kYellow);

    f.press(kRow, 4); // not a polyphony pad
    EXPECT_EQ(f.grid.getMaxPolyphony(), 4);
    f.press(kRow, LaunchpadGrid::kSideCol); // nothing there any more
    EXPECT_TRUE(f.grid.isAccompanimentActive());
}

TEST(AccompanimentGrid, polyphonyIsKeptAcrossModes) {
    GridFixture f;
    f.grid.setSelectedMode(OrchestrationMode::Accompaniment);
    f.press(kRow, 2);
    ASSERT_EQ(f.grid.getMaxPolyphony(), 3);

    f.grid.setSelectedMode(OrchestrationMode::Edit);
    EXPECT_EQ(f.grid.effectivePolyphonyCap(), 0);
    EXPECT_EQ(f.colourOf(kRow, 2), LaunchpadLighting::kOff);

    f.grid.setSelectedMode(OrchestrationMode::Accompaniment);
    EXPECT_EQ(f.grid.effectivePolyphonyCap(), 3) << "restored";
    EXPECT_EQ(f.colourOf(kRow, 2), LaunchpadLighting::kYellow);
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

TEST(AccompanimentReduction, keepQueueFlagFollowsTheSetter) {
    ModelConfig config;
    StubReduction rt(config);
    EXPECT_FALSE(rt.keepsQueueOnInput());
    rt.setKeepQueueOnInput(true);
    EXPECT_TRUE(rt.keepsQueueOnInput());
}

TEST(AccompanimentReduction, modeCutDropsGeneratedNotesFromTheCutAndSendsTheMarker) {
    ModelConfig config;
    StubReduction rt(config);
    rt.pushOutputToken(note(100, 50, 60)); // played
    rt.pushOutputToken(note(300, 50, 62)); // generated ahead
    rt.pushOutputToken(note(400, 50, 64));
    Token drained{};
    while (rt.outputTokenQueue.pull(drained)) {
    }

    EXPECT_FALSE(rt.consumeModeCut().has_value()) << "nothing pending";
    rt.requestModeCut(250);
    const auto cut = rt.consumeModeCut();
    ASSERT_TRUE(cut.has_value());
    EXPECT_EQ(*cut, 250);
    EXPECT_FALSE(rt.consumeModeCut().has_value()) << "consumed once";

    Token marker{};
    ASSERT_TRUE(rt.outputTokenQueue.pull(marker));
    EXPECT_TRUE(PedalClear::isPedalClear(marker));
    EXPECT_EQ(marker.time, 250);

    for (const auto &t: rt.getOutputHistory())
        if (t.note != static_cast<int32_t>(Vocab::ClearQueue))
            EXPECT_LT(t.time, 250) << "notes from the cut on are dropped";
}

TEST(AccompanimentReduction, rewindDropsContextEventsFromTheCutAndMovesTheClock) {
    ModelConfig config;
    StubReduction rt(config);
    // Stride 3 (time, duration, note): live note at 100, generated at 300 and 400, live at 200.
    rt.data() = {100, 1050, 11060, 300, 1050, 11062, 400, 1050, 11064, 200, 1050, 11067};
    rt.setTime(400);
    rt.rewindInputDataFrom(250);
    EXPECT_EQ(rt.data(), (std::vector<int32_t>{100, 1050, 11060, 200, 1050, 11067}));
    EXPECT_EQ(rt.time(), 250);
}

} // namespace virtual_orch_test
