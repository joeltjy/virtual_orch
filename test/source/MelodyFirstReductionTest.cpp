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
        // T8'' is dense (11776), T7'' is AMT-native (55293)
        rt.initWithBackends(std::make_unique<FakeBackend>(11776, 512), 11776,
                            std::make_unique<FakeBackend>(55293, 512), 55293);
        rt.nowCsProvider = [this] { return now; };
        rt.tdValueProvider = [this] { return tdValue; };
    }

    /** Start the streams at the app clock (1000), then let 1 s pass: notes played from now on
     *  begin after the stream origin, as notes do in the app. */
    auto begin() -> void {
        EXPECT_FALSE(rt.stepOnce());
        now = 1100;
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

auto countToken(const std::vector<int32_t> &tokens, int32_t id) -> int {
    return static_cast<int>(std::count(tokens.begin(), tokens.end(), id));
}

} // namespace

namespace virtual_orch_test {

TEST(MelodyFirst, waitsForTheFirstUserNote) {
    Rig rig;
    EXPECT_FALSE(rig.rt.stepOnce());
    EXPECT_TRUE(rig.rt.melodyEventsForTest().empty());
    rig.begin();
    rig.play(1000, 64, 40);
    EXPECT_TRUE(rig.rt.stepOnce());
}

TEST(MelodyFirst, generatesMelodyInternallyAndOnlyReductionNotesGoDownstream) {
    Rig rig;
    rig.begin();
    rig.play(1000, 64, 40);
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

TEST(MelodyFirst, theTwoModelsUseTheirOwnVocabs) {
    Rig rig;
    rig.begin();
    rig.play(1000, 64, 40);
    ASSERT_TRUE(rig.rt.stepOnce());
    const auto t8 = rig.rt.melodyStreamForTest().tokens();
    const auto t7 = rig.rt.reductionStreamForTest().tokens();
    EXPECT_EQ(rig.rt.melodyStreamForTest().vocabKind(), AnticipatedStream::Vocab::Dense);
    EXPECT_EQ(rig.rt.reductionStreamForTest().vocabKind(), AnticipatedStream::Vocab::Amt);
    EXPECT_NE(t8[0], 55027) << "T8'' has no mode token: it starts with the first quad";
    EXPECT_EQ(t7[0], 55027) << "T7'' starts with the ANTICIPATE flag";
    for (const auto id: t8)
        EXPECT_LT(id, 11776) << "dense ids only";
    for (const auto id: t7)
        EXPECT_LT(id, 55293);
}

TEST(MelodyFirst, userNotesFeedBothModelsAsGivenEvents) {
    Rig rig;
    EXPECT_FALSE(rig.rt.stepOnce()); // starts the streams at the app clock (1000)
    rig.now = 1100;
    rig.play(1000, 72, 80); // finished notes arrive at their note-off
    rig.play(1000, 40, 80);
    ASSERT_TRUE(rig.rt.stepOnce());
    EXPECT_FALSE(rig.rt.stateForTest().getPieces().empty());

    // T8'': the melody + bass R as instrument-0 events in the dense vocab
    bool given8 = false;
    for (size_t i = 0; i + 3 < rig.rt.melodyStreamForTest().tokens().size(); ++i) {
        const auto id = rig.rt.melodyStreamForTest().tokens()[i];
        if (id >= 11000 && id < 11128)
            given8 = true;
    }
    EXPECT_TRUE(given8);
    // T7'': the piano as instrument-0 EVENTS (11000 + pitch, velocity 55028 + v), not controls
    const auto t7 = rig.rt.reductionStreamForTest().tokens();
    bool pianoEvent = false;
    for (size_t i = 1; i + 3 < t7.size(); ++i)
        if (t7[i + 2] == 11000 + 72 && t7[i + 3] >= 55028 && t7[i + 3] < 55028 + 128)
            pianoEvent = true;
    EXPECT_TRUE(pianoEvent);
}

TEST(MelodyFirst, melodyNotesBecomeAnticipatedControlsOfTheReductionModel) {
    Rig rig;
    rig.begin();
    rig.play(1000, 64, 40);
    for (int i = 0; i < 8; ++i) {
        ASSERT_TRUE(rig.rt.stepOnce()) << i;
        rig.now += 40;
    }
    int controlQuads = 0;
    const auto t7 = rig.rt.reductionStreamForTest().tokens();
    for (size_t i = 1; i + 3 < t7.size(); ++i)
        if (t7[i] >= 27513 && t7[i] < 37513 && t7[i + 1] >= 37513 && t7[i + 1] < 38513)
            ++controlQuads;
    EXPECT_GE(controlQuads, 1) << "T8'''s notes appear as AMT control quads";
}

TEST(MelodyFirst, melodyDoesNotRunMoreThanThreeSecondsAheadOfTheReduction) {
    Rig rig;
    rig.begin();
    rig.play(1000, 64, 40);
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

TEST(MelodyFirst, silenceWritesVisibleRestsOnlyInTheAmtStream) {
    Rig rig;
    rig.begin();
    rig.play(1000, 64, 40);
    ASSERT_TRUE(rig.rt.stepOnce());
    rig.now += 1000; // 10 s of silence
    ASSERT_TRUE(rig.rt.stepOnce());
    EXPECT_GE(countToken(rig.rt.reductionStreamForTest().tokens(), 27512), 8)
        << "T7'' sees one REST per second";
    EXPECT_EQ(countToken(rig.rt.melodyStreamForTest().tokens(), 27512), 0) << "T8'' has no rests";
}

TEST(MelodyFirst, aNoteOffRefreshesRAndTheHeldNoteIsProvisionalInBothStreams) {
    Rig rig;
    rig.begin();
    rig.play(1000, 64, 0, /*held*/ true);
    ASSERT_TRUE(rig.rt.stepOnce());
    ASSERT_EQ(rig.rt.stateForTest().getPieces().size(), 1u);
    EXPECT_TRUE(rig.rt.stateForTest().getPieces()[0].provisional);
    EXPECT_LT(rig.rt.reductionStreamForTest().committedLen(),
              rig.rt.reductionStreamForTest().tokens().size());

    TokenUpdate update;
    update.oldNote = Token{1000, static_cast<int32_t>(Vocab::DurOffset) + rig.config.inputDuration,
                           static_cast<int32_t>(Vocab::NoteOffset) + 64};
    update.newNote = Token{1000, static_cast<int32_t>(Vocab::DurOffset) + 90,
                           static_cast<int32_t>(Vocab::NoteOffset) + 64};
    rig.rt.updatesFromFilter.push(update);
    rig.now += 100;
    ASSERT_TRUE(rig.rt.stepOnce());
    EXPECT_FALSE(rig.rt.stateForTest().getPieces()[0].provisional);
    EXPECT_EQ(rig.rt.stateForTest().getPieces()[0].durationCs, 90);
    EXPECT_EQ(rig.rt.reductionStreamForTest().committedLen(),
              rig.rt.reductionStreamForTest().tokens().size())
        << "nothing is held any more in the T7'' stream";
}

TEST(MelodyFirst, theMelodyStageIsTwoVoicesAndTheCapAppliesToBothStages) {
    {
        Rig rig; // no accompaniment cap: at most 2 sounding in the melody
        rig.begin();
    rig.play(1000, 64, 40);
        for (int i = 0; i < 12; ++i) {
            (void) rig.rt.stepOnce();
            rig.now += 20;
            (void) rig.drain();
        }
        const auto &melody = rig.rt.melodyEventsForTest();
        ASSERT_GE(melody.size(), 4u);
        for (const auto &probe: melody) {
            int sounding = 0;
            for (const auto &other: melody)
                if (other.time <= probe.time && probe.time < other.time + other.getRealDuration())
                    ++sounding;
            EXPECT_LE(sounding, 2) << "never more than two voices at " << probe.time;
        }
    }
    {
        Rig rig;
        rig.rt.setMaxPolyphony(1);
        rig.begin();
    rig.play(1000, 64, 40);
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
}

TEST(MelodyFirst, tdTokensFollowTheLaunchpadWithEachModelsOwnIds) {
    Rig rig;
    rig.tdValue = 4;
    rig.begin();
    rig.play(1000, 64, 40);
    ASSERT_TRUE(rig.rt.stepOnce());
    bool dense = false, amt = false;
    for (const auto id: rig.rt.melodyStreamForTest().tokens())
        if (id >= 11768) {
            dense = true;
            EXPECT_EQ(id, TdTokens::denseIdForLevel(4, 11776));
        }
    for (const auto id: rig.rt.reductionStreamForTest().tokens())
        if (id >= 55285) {
            amt = true;
            EXPECT_EQ(id, TdTokens::amtIdForLevel(4, 55293));
        }
    EXPECT_TRUE(dense);
    EXPECT_TRUE(amt);
}

TEST(MelodyFirst, pausedGenerationProducesNothing) {
    Rig rig;
    rig.begin();
    rig.play(1000, 64, 40);
    rig.rt.generationPause.set(true);
    EXPECT_FALSE(rig.rt.stepOnce());
    EXPECT_TRUE(rig.drain().empty());
}

// Optional: both real exports, once they exist. VIRTUAL_ORCH_TEST_T7_ONNX / VIRTUAL_ORCH_TEST_T8_ONNX
// point at the T7'' (AMT-native) and T8'' (dense) KV exports.
TEST(MelodyFirstReal, bothRealModelsRunEndToEnd) {
    const char *t7 = std::getenv("VIRTUAL_ORCH_TEST_T7_ONNX");
    const char *t8 = std::getenv("VIRTUAL_ORCH_TEST_T8_ONNX");
    if (t7 == nullptr || t8 == nullptr)
        GTEST_SKIP() << "VIRTUAL_ORCH_TEST_T7_ONNX / VIRTUAL_ORCH_TEST_T8_ONNX not set (no exports yet)";
    ModelConfig config;
    MelodyFirstReductionTransformer rt(config);
    if (! rt.init(t7, t8))
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
}

} // namespace virtual_orch_test
