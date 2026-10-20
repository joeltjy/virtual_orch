#include <gtest/gtest.h>

#include <chrono>
#include <functional>
#include <mutex>
#include <thread>

#include <VirtualOrch/AppSession.h>
#include <VirtualOrch/Clock.h>
#include <VirtualOrch/Metrics.h>
#include <VirtualOrch/PerformanceActivity.h>
#include <VirtualOrch/widgets/PianoRollView.h>
#include <VirtualOrch/orchestration-models/IodPretrained.h>
#include <VirtualOrch/orchestration-models/IodPretrainedTD.h>

namespace {

using namespace StreamingKv;

auto note(int32_t onset, int32_t durCs, int32_t pitch) -> Token {
    return Token{onset, static_cast<int32_t>(Vocab::DurOffset) + durCs,
                 static_cast<int32_t>(Vocab::NoteOffset) + pitch};
}

/** Brightness of a pixel as seen over black (the snapshot keeps the alpha of what was drawn). */
auto pixelLight(const juce::Colour &c) -> float {
    return c.getBrightness() * c.getFloatAlpha();
}

/** Mean brightness of the row band a pitch occupies (the whole width). */
auto bandBrightness(const juce::Image &image, int32_t pitch, int32_t low, int32_t high) -> float {
    const float rows = static_cast<float>(high - low + 1);
    const float rowH = static_cast<float>(image.getHeight()) / rows;
    const int y0 = static_cast<int>(static_cast<float>(high - pitch) * rowH);
    const int y1 = static_cast<int>(static_cast<float>(high - pitch + 1) * rowH) - 1;
    double sum = 0.0;
    int n = 0;
    for (int y = y0 + 1; y < y1; ++y)
        for (int x = 0; x < image.getWidth(); ++x) {
            sum += pixelLight(image.getPixelAt(x, y));
            ++n;
        }
    return static_cast<float>(sum / std::max(1, n));
}

class FakeBackend : public Backend {
public:
    FakeBackend() { d.layers = 1; d.heads = 1; d.headDim = 1; d.maxRows = 512; d.sinkRows = 4; }
    [[nodiscard]] auto dims() const -> Dims override { return d; }
    auto run(const State &past, const std::vector<int64_t> &ids, State &present,
             std::vector<float> &last) -> bool override {
        present = past;
        for (const auto id: ids) {
            present.key.push_back(static_cast<float>(id));
            present.value.push_back(static_cast<float>(id));
            ++present.rows;
        }
        last.assign(11776, 0.0f);
        for (size_t i = 0; i < last.size(); ++i)
            last[i] = -0.02f * static_cast<float>(i < 10000 ? i : 0);
        return true;
    }
    Dims d;
};

} // namespace

namespace virtual_orch_test {

TEST(PianoRollProvisional, heldNotesAreOutlinesAndLockedInNotesAreSolid) {
    const juce::ScopedJuceInitialiser_GUI gui;
    Metrics metrics;
    Clock clock(metrics);
    PianoRollView roll(clock);
    roll.setBounds(0, 0, 400, 310);
    roll.setPitchRange(50, 80);
    roll.setNoteColours(juce::Colours::white, juce::Colours::white);

    // pitch 60: finished note (locked in); pitch 70: still held (provisional duration 400 cs)
    roll.setNotes({note(-100, 300, 60), note(-100, 400, 70)});

    roll.setProvisionalDurationCs(0); // baseline: everything solid
    const auto plain = roll.createComponentSnapshot(roll.getLocalBounds(), false);
    const float lockedPlain = bandBrightness(plain, 60, 50, 80);
    const float heldPlain = bandBrightness(plain, 70, 50, 80);
    ASSERT_GT(lockedPlain, 0.02f);
    ASSERT_GT(heldPlain, 0.02f);

    roll.setProvisionalDurationCs(400);
    const auto marked = roll.createComponentSnapshot(roll.getLocalBounds(), false);
    const float lockedMarked = bandBrightness(marked, 60, 50, 80);
    const float heldMarked = bandBrightness(marked, 70, 50, 80);
    EXPECT_NEAR(lockedMarked, lockedPlain, lockedPlain * 0.02f) << "a locked-in note is unchanged";
    EXPECT_LT(heldMarked, heldPlain * 0.5f) << "the held note's body is mostly transparent";
    EXPECT_GT(heldMarked, 0.0f) << "but its outline is drawn";
}

TEST(PianoRollProvisional, theDurationSentinelDoesNotDependOnTheModel) {
    // The rule is the pipeline's own: duration == ModelConfig::inputDuration (snapped to the dense
    // grid), the same test hasUnresolvedProvisionalInputNotes uses for every reduction model.
    ModelConfig config;
    EXPECT_TRUE(DenseSampling::isProvisionalDurationToken(
        static_cast<int32_t>(Vocab::DurOffset) + config.inputDuration, config.inputDuration));
    EXPECT_FALSE(DenseSampling::isProvisionalDurationToken(
        static_cast<int32_t>(Vocab::DurOffset) + config.inputDuration + 5, config.inputDuration));
}

TEST(ForceRestart, describesItInTheConsole) {
    using K = PerformanceActivity::LaunchpadAction::Kind;
    EXPECT_EQ(PerformanceActivity::describe({K::ForceRestart, 0, 0, false, false},
                                            [](int32_t) { return juce::String(); }),
              "Force restart");
}

TEST(ForceRestart, topRowButtonZeroRaisesTheSceneCallbackThatRestarts) {
    LaunchpadGrid grid;
    int last = -1;
    grid.scene_callback = [&](int idx) { last = idx; };
    grid.handleSceneInput(0);
    EXPECT_EQ(last, 0);
}

TEST(ForceRestart, resetStateClearsWhatTheOrchestrationModelsRemember) {
    ModelConfig unused;
    for (const bool td: {false, true}) {
        std::unique_ptr<OrchestrationModel> model;
        const auto dummy = std::string(VIRTUAL_ORCH_TEST_RESOURCES) + (td ? "/iod_td_dummy.onnx" : "/iod_dummy.onnx");
        if (td) {
            auto m = std::make_unique<IodPretrainedTD>();
            m->init(dummy.c_str());
            model = std::move(m);
        } else {
            auto m = std::make_unique<IodPretrained>();
            m->init(dummy.c_str());
            model = std::move(m);
        }
        const std::vector<int32_t> instruments{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 19};
        (void) model->getOutput({note(0, 60, 60), note(40, 60, 64)}, instruments, {});
        ASSERT_EQ(model->getInputNotes().size(), 2u) << (td ? "TD" : "IOD");
        model->resetState();
        EXPECT_TRUE(model->getInputNotes().empty()) << (td ? "TD" : "IOD");
        EXPECT_TRUE(model->tokenHistory().empty());
        // and it works again, from a clean list
        (void) model->getOutput({note(0, 60, 62)}, instruments, {});
        EXPECT_EQ(model->getInputNotes().size(), 1u);
    }
}

TEST(ForceRestart, isAStopFollowedByAStart) {
    const juce::ScopedJuceInitialiser_GUI gui;
    AppSession session;
    session.reductionTransformerAnticipated.initWithBackend(std::make_unique<FakeBackend>(), 11776);
    session.setMusicModelArch(MusicModelArch::DenseAnticipatedPiano);
    ASSERT_TRUE(session.isMusicModelLoaded());

    // The message thread is the test thread: restart steps are queued and run by pump().
    std::mutex queueMutex;
    std::vector<std::function<void()>> queue;
    session.postToMessageThread = [&](std::function<void()> f) {
        const std::lock_guard<std::mutex> lock(queueMutex);
        queue.push_back(std::move(f));
    };
    const auto pump = [&](int ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
        std::vector<std::function<void()>> todo;
        {
            const std::lock_guard<std::mutex> lock(queueMutex);
            todo.swap(queue);
        }
        for (auto &f: todo)
            f();
    };

    session.startGeneration();
    pump(200);
    ASSERT_TRUE(session.isMusicThreadRunning());

    // the reduction produces something: its output history fills
    const auto clockNow = static_cast<int32_t>(session.clock.getTime());
    session.activeReduction().inputTokenQueue.push(
        note(clockNow, 20, 60)); // a finished note ending now
    for (int i = 0; i < 100 && session.activeReduction().getOutputHistory().empty(); ++i)
        pump(50);
    ASSERT_FALSE(session.activeReduction().getOutputHistory().empty());

    session.forceRestart();
    for (int i = 0; i < 400 && session.getRestartsCompleted() == 0; ++i)
        pump(50);
    ASSERT_EQ(session.getRestartsCompleted(), 1);
    EXPECT_TRUE(session.isMusicThreadRunning()) << "started again";
    EXPECT_TRUE(session.activeReduction().getOutputHistory().empty())
        << "the stop cleared the history, and the new run has no input";

    // a second press while one is in flight is ignored; a later one works again
    session.forceRestart();
    session.forceRestart();
    for (int i = 0; i < 400 && session.getRestartsCompleted() < 2; ++i)
        pump(50);
    EXPECT_EQ(session.getRestartsCompleted(), 2);

    session.stopGeneration();
    for (int i = 0; i < 100; ++i)
        pump(20);
}

} // namespace virtual_orch_test
