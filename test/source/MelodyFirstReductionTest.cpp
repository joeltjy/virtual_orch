#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <random>

#include <VirtualOrch/reduction/MelodyFirstReductionTransformer.h>

namespace {

using namespace StreamingKv;

/** Fake model: pseudo-random logits seeded by the cache contents, preferring early onsets. */
class FakeBackend : public Backend {
public:
    FakeBackend(size_t vocabIn, int maxRows) : vocab(vocabIn) {
        d.layers = 1; d.heads = 1; d.headDim = 1; d.maxRows = maxRows; d.sinkRows = 4;
    }
    [[nodiscard]] auto dims() const -> Dims override { return d; }
    auto run(const State &past, const std::vector<int64_t> &ids, State &present,
             std::vector<float> &last) -> bool override {
        present = past;
        for (const auto id: ids) {
            present.key.push_back(static_cast<float>(id));
            present.value.push_back(static_cast<float>(id));
            ++present.rows;
        }
        uint32_t seed = 29;
        for (const float k: present.key)
            seed = seed * 1664525u + static_cast<uint32_t>(k) + 1013904223u;
        std::mt19937 gen(seed);
        std::normal_distribution<float> n(0.0f, 1.0f);
        last.resize(vocab);
        for (size_t i = 0; i < last.size(); ++i) {
            last[i] = n(gen);
            if (i < 10000)
                last[i] -= 0.02f * static_cast<float>(i);
        }
        return true;
    }
    size_t vocab;
    Dims d;
};

struct Rig {
    ModelConfig config;
    MelodyFirstReductionTransformer rt{config};
    int32_t now = 1000;
    int32_t tdValue = TemporalDensity::kAuto;

    Rig() {
        rt.initWithBackends(std::make_unique<FakeBackend>(55285, 288), 55285,
                            std::make_unique<FakeBackend>(11776, 512), 11776);
        rt.nowCsProvider = [this] { return now; };
        rt.tdValueProvider = [this] { return tdValue; };
    }

    auto play(int32_t onset, int pitch, int32_t durCs, bool held = false) -> void {
        const int32_t dur = held ? config.inputDuration : durCs;
        Token t{onset, static_cast<int32_t>(Vocab::DurOffset) + dur,
                static_cast<int32_t>(Vocab::NoteOffset) + pitch};
        t.velocity = 90;
        rt.inputTokenQueue.push(t);
    }

    auto drain() -> std::vector<Token> {
        std::vector<Token> out;
        Token t{};
        while (rt.outputTokenQueue.pull(t))
            out.push_back(t);
        return out;
    }
};

} // namespace

namespace virtual_orch_test {

TEST(MelodyFirst, waitsForTheFirstUserNote) {
    Rig rig;
    EXPECT_FALSE(rig.rt.stepOnce());
    EXPECT_TRUE(rig.rt.melodyEventsForTest().empty());
    rig.play(950, 64, 80);
    EXPECT_TRUE(rig.rt.stepOnce());
}

TEST(MelodyFirst, generatesMelodyInternallyAndOnlyReductionNotesGoDownstream) {
    Rig rig;
    rig.play(950, 64, 80);
    ASSERT_TRUE(rig.rt.stepOnce());
    EXPECT_GE(rig.rt.melodyEventsForTest().size(), 1u);
    EXPECT_LE(rig.rt.melodyEventsForTest().size(), 3u) << "three melody notes per loop at most";
    const auto out = rig.drain();
    ASSERT_EQ(out.size(), 1u) << "one reduction note per loop";
    const int32_t dur = out[0].duration - static_cast<int32_t>(Vocab::DurOffset);
    EXPECT_GE(dur, 10);
    EXPECT_LE(dur, 995);
    EXPECT_EQ(dur % 5, 0);
    EXPECT_GE(out[0].getPitch(), 21);
    EXPECT_LE(out[0].getPitch(), 108);
    EXPECT_EQ(out[0].velocity % 8, 0);
    EXPECT_GE(out[0].velocity, 8);
}

TEST(MelodyFirst, userNotesFeedBothModelsAsControls) {
    Rig rig;
    EXPECT_FALSE(rig.rt.stepOnce()); // starts the streams at the app clock (1000)
    rig.now = 1100;
    rig.play(1000, 72, 80); // finished notes arrive at their note-off
    rig.play(1000, 40, 80);
    ASSERT_TRUE(rig.rt.stepOnce());
    // T8 sees melody + bass of the user's notes as AMT control quads on instrument 0
    int controlQuads = 0;
    const auto t8 = rig.rt.melodyStreamForTest().tokens();
    for (size_t i = 1; i + 3 < t8.size(); i += 4)
        if (t8[i] >= 27513 && t8[i] < 37513)
            ++controlQuads;
    EXPECT_GE(controlQuads, 1);
    // the DA model sees the piano as instrument 0 and (later) the melody as instrument 2
    bool piano = false;
    for (const auto id: rig.rt.reductionStreamForTest().tokens())
        if (id >= 11000 && id < 11640 && (id - 11000) / 128 == 0)
            piano = true;
    EXPECT_TRUE(piano);
    EXPECT_FALSE(rig.rt.stateForTest().getPieces().empty());
}

TEST(MelodyFirst, melodyNotesBecomeInstrumentTwoControlsThreeSecondsEarly) {
    Rig rig;
    rig.play(950, 64, 80);
    for (int i = 0; i < 8; ++i) {
        ASSERT_TRUE(rig.rt.stepOnce()) << i;
        rig.now += 40;
    }
    bool melodyInDa = false;
    for (size_t i = 0; i + 3 < rig.rt.reductionStreamForTest().tokens().size(); ++i) {
        const auto id = rig.rt.reductionStreamForTest().tokens()[i];
        if (id >= 11000 && id < 11640 && (id - 11000) / 128 == 2)
            melodyInDa = true;
    }
    EXPECT_TRUE(melodyInDa);
}

TEST(MelodyFirst, melodyDoesNotRunMoreThanThreeSecondsAheadOfTheReduction) {
    Rig rig;
    rig.play(950, 64, 80);
    for (int i = 0; i < 25; ++i) {
        const bool tooFar = rig.rt.melodyIsTooFarAhead();
        const size_t before = rig.rt.melodyEventsForTest().size();
        (void) rig.rt.stepOnce();
        if (tooFar)
            EXPECT_EQ(rig.rt.melodyEventsForTest().size(), before)
                << "no melody while more than 3 s ahead (step " << i << ")";
        rig.now += 30;
        (void) rig.drain();
    }
}

TEST(MelodyFirst, silenceWritesVisibleRestQuads) {
    Rig rig;
    rig.play(950, 64, 80);
    ASSERT_TRUE(rig.rt.stepOnce());
    rig.now += 1000; // 10 s of silence
    ASSERT_TRUE(rig.rt.stepOnce());
    int rests = 0;
    const auto t8 = rig.rt.melodyStreamForTest().tokens();
    for (size_t i = 1; i + 3 < t8.size(); i += 4)
        if (t8[i + 2] == 27512)
            ++rests;
    EXPECT_GE(rests, 8);
}

TEST(MelodyFirst, theMbStreamIsCutToHalfWhenItPassesTwoHundredFiftySixTokens) {
    Rig rig;
    rig.play(950, 64, 80);
    size_t peak = 0;
    for (int i = 0; i < 60; ++i) {
        (void) rig.rt.stepOnce();
        rig.now += 25;
        (void) rig.drain();
        peak = std::max(peak, rig.rt.melodyStreamForTest().tokens().size());
        EXPECT_LE(rig.rt.melodyStreamForTest().tokens().size(), 260u);
        if (i % 5 == 0)
            rig.play(rig.now - 10, 60 + i % 20, 60);
    }
    EXPECT_GT(peak, 100u);
    ASSERT_NE(rig.rt.melodyCacheForTest(), nullptr);
}

TEST(MelodyFirst, aNoteOffRefreshesRAndTheHeldPianoIsProvisionalInTheDaStream) {
    Rig rig;
    rig.play(950, 64, 0, /*held*/ true);
    ASSERT_TRUE(rig.rt.stepOnce());
    ASSERT_EQ(rig.rt.stateForTest().getPieces().size(), 1u);
    EXPECT_TRUE(rig.rt.stateForTest().getPieces()[0].provisional);
    EXPECT_LT(rig.rt.reductionStreamForTest().committedLen(), rig.rt.reductionStreamForTest().tokens().size());

    TokenUpdate update;
    update.oldNote = Token{950, static_cast<int32_t>(Vocab::DurOffset) + rig.config.inputDuration,
                           static_cast<int32_t>(Vocab::NoteOffset) + 64};
    update.newNote = Token{950, static_cast<int32_t>(Vocab::DurOffset) + 90,
                           static_cast<int32_t>(Vocab::NoteOffset) + 64};
    rig.rt.updatesFromFilter.push(update);
    rig.now += 100;
    ASSERT_TRUE(rig.rt.stepOnce());
    EXPECT_FALSE(rig.rt.stateForTest().getPieces()[0].provisional);
    EXPECT_EQ(rig.rt.stateForTest().getPieces()[0].durationCs, 90);
}

TEST(MelodyFirst, polyphonyCapOneAppliesToTheMelodyAndTheReduction) {
    Rig rig;
    rig.rt.setMaxPolyphony(1);
    rig.play(950, 64, 80);
    std::vector<Token> reduction;
    for (int i = 0; i < 10; ++i) {
        (void) rig.rt.stepOnce();
        for (const auto &t: rig.drain())
            reduction.push_back(t);
        rig.now += 20;
    }
    ASSERT_GE(reduction.size(), 3u);
    for (size_t i = 1; i < reduction.size(); ++i)
        EXPECT_GE(reduction[i].time, reduction[i - 1].time + reduction[i - 1].getRealDuration());
    const auto &melody = rig.rt.melodyEventsForTest();
    ASSERT_GE(melody.size(), 3u);
    for (size_t i = 1; i < melody.size(); ++i)
        EXPECT_GE(melody[i].time, melody[i - 1].time + melody[i - 1].getRealDuration());
}

TEST(MelodyFirst, tdTokensFollowTheLaunchpadInTheReductionStream) {
    Rig rig;
    rig.tdValue = 4;
    rig.play(950, 64, 80);
    ASSERT_TRUE(rig.rt.stepOnce());
    bool td = false;
    for (const auto id: rig.rt.reductionStreamForTest().tokens())
        if (id >= 11768) {
            td = true;
            EXPECT_EQ(id, TdTokens::denseIdForLevel(4, 11776));
        }
    EXPECT_TRUE(td);
}

TEST(MelodyFirst, pausedGenerationProducesNothing) {
    Rig rig;
    rig.play(950, 64, 80);
    rig.rt.generationPause.set(true);
    EXPECT_FALSE(rig.rt.stepOnce());
    EXPECT_TRUE(rig.drain().empty());
}

// Optional: both real exports (VIRTUAL_ORCH_TEST_MODELS=<dir>), a few steps end to end.
TEST(MelodyFirstReal, bothRealModelsRunEndToEnd) {
    const char *dir = std::getenv("VIRTUAL_ORCH_TEST_MODELS");
    if (dir == nullptr)
        GTEST_SKIP() << "VIRTUAL_ORCH_TEST_MODELS not set";
    const auto da = std::filesystem::path(dir) / "da_rope_td_T7_dummy_from_rope_ckpt6000_kv.onnx";
    const auto mb = std::filesystem::path(dir) / "mb_phase2_T8_best_step4000_kv.onnx";
    if (! std::filesystem::exists(da) || ! std::filesystem::exists(mb))
        GTEST_SKIP() << "exports missing";
    ModelConfig config;
    MelodyFirstReductionTransformer rt(config);
    if (! rt.init(da.string().c_str(), mb.string().c_str()))
        GTEST_SKIP() << "this ONNX Runtime cannot load the exports";
    int32_t now = 1000;
    rt.nowCsProvider = [&] { return now; };
    for (const int pitch: {72, 67, 64}) {
        Token t{now - 20, static_cast<int32_t>(Vocab::DurOffset) + 80,
                static_cast<int32_t>(Vocab::NoteOffset) + pitch};
        t.velocity = 90;
        rt.inputTokenQueue.push(t);
    }
    for (int i = 0; i < 2; ++i) {
        EXPECT_TRUE(rt.stepOnce()) << i;
        now += 30;
    }
    Token out{};
    int count = 0;
    while (rt.outputTokenQueue.pull(out)) {
        ++count;
        EXPECT_GE(out.getPitch(), 21);
        EXPECT_LE(out.getPitch(), 108);
    }
    EXPECT_EQ(count, 2);
    EXPECT_GE(rt.melodyEventsForTest().size(), 2u);
}

} // namespace virtual_orch_test
