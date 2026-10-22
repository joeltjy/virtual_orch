#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include <VirtualOrch/reduction/TwoInstrumentAnticipatedReductionTransformer.h>

namespace {

using namespace StreamingKv;

/** Dense-vocab fake model: logits are pseudo-random, seeded by the cache contents. */
class DenseFakeBackend : public Backend {
public:
    explicit DenseFakeBackend(size_t vocabIn) : vocab(vocabIn) {
        d.layers = 1; d.heads = 1; d.headDim = 1; d.maxRows = 512; d.sinkRows = 4;
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
        uint32_t seed = 17;
        for (const float k: present.key)
            seed = seed * 1664525u + static_cast<uint32_t>(k) + 1013904223u;
        std::mt19937 gen(seed);
        std::normal_distribution<float> n(0.0f, 1.0f);
        last.resize(vocab);
        for (size_t i = 0; i < last.size(); ++i) {
            last[i] = n(gen);
            // Prefer early onsets, like a real model (gaps of a few tens of cs)
            if (i < 10000)
                last[i] -= 0.02f * static_cast<float>(i);
        }
        ++calls;
        return true;
    }
    size_t vocab;
    Dims d;
    int calls = 0;
};

struct OnnxBackendHolder {
    StreamingKv::OnnxBackend backend;
};

struct Rig {
    ModelConfig config;
    TwoInstrumentAnticipatedReductionTransformer rt{config};
    int32_t now = 1000;
    int32_t tdValue = TemporalDensity::kAuto;
    DenseFakeBackend *fake = nullptr;

    explicit Rig(size_t vocab = 11776) {
        auto backend = std::make_unique<DenseFakeBackend>(vocab);
        fake = backend.get();
        rt.initWithBackend(std::move(backend), vocab);
        rt.nowCsProvider = [this] { return now; };
        rt.tdValueProvider = [this] { return tdValue; };
    }

    auto playNote(int32_t onset, int32_t pitch, int32_t durCs, bool held = false) -> void {
        const int32_t dur = held ? config.inputDuration : durCs;
        Token t{onset, static_cast<int32_t>(Vocab::DurOffset) + dur,
                static_cast<int32_t>(Vocab::NoteOffset) + pitch};
        t.velocity = 90;
        rt.inputTokenQueue.push(t);
    }

    auto drainOutput() -> std::vector<Token> {
        std::vector<Token> out;
        Token t{};
        while (rt.outputTokenQueue.pull(t))
            out.push_back(t);
        return out;
    }
};

auto isTd(int32_t id) -> bool { return id >= 11768 && id <= 11775; }

} // namespace

namespace virtual_orch_test {

TEST(AnticipatedReduction, waitsForTheFirstUserNote) {
    Rig rig;
    EXPECT_FALSE(rig.rt.stepOnce());
    EXPECT_TRUE(rig.drainOutput().empty());
    rig.playNote(900, 60, 80);
    EXPECT_TRUE(rig.rt.stepOnce());
    EXPECT_EQ(rig.drainOutput().size(), 1u);
}

TEST(AnticipatedReduction, generatesOnlyLegalQ5V8Notes) {
    Rig rig;
    rig.playNote(900, 60, 80);
    int32_t previousOnset = 0;
    for (int i = 0; i < 12; ++i) {
        ASSERT_TRUE(rig.rt.stepOnce()) << i;
        rig.now += 20;
        for (const auto &t: rig.drainOutput()) {
            const int32_t dur = t.duration - static_cast<int32_t>(Vocab::DurOffset);
            EXPECT_GE(dur, 10);
            EXPECT_LE(dur, 995);
            EXPECT_EQ(dur % 5, 0);
            const int32_t pitch = t.getPitch();
            EXPECT_GE(pitch, 21);
            EXPECT_LE(pitch, 108);
            EXPECT_GE(t.velocity, 8);
            EXPECT_LE(t.velocity, 120);
            EXPECT_EQ(t.velocity % 8, 0);
            EXPECT_GE(t.time, previousOnset) << "event clock never goes back";
            EXPECT_GE(t.time, 1000) << "never before the start of the stream";
            previousOnset = t.time;
        }
    }
}

TEST(AnticipatedReduction, controlsEnterOncePastDueAsInstrumentZero) {
    Rig rig;
    // Accompaniment rules: input never rewinds (in Jam it cuts the future and regenerates).
    rig.rt.setKeepQueueOnInput(true);
    rig.playNote(990, 64, 50);
    ASSERT_TRUE(rig.rt.stepOnce()); // first note: clock moves to the sampled onset
    const int32_t clock = rig.rt.streamForTest().clock();
    // A control far in the future stays out; a past-due one goes in on the next step.
    rig.playNote(clock + 3000, 70, 50);
    rig.playNote(clock - 5, 66, 50);
    ASSERT_TRUE(rig.rt.stepOnce());
    const auto tokens = rig.rt.streamForTest().tokens();
    int controls = 0;
    std::set<int32_t> controlPitches;
    for (size_t i = 0; i + 3 < tokens.size(); ++i) {
        if (isTd(tokens[i]))
            continue;
        const int32_t note = tokens[i + 2];
        if (note >= 11000 && note < 11640 && (note - 11000) / 128 == 0) {
            ++controls;
            controlPitches.insert((note - 11000) % 128);
        }
        i += 3;
    }
    EXPECT_GE(controls, 2);
    EXPECT_TRUE(controlPitches.count(64));
    EXPECT_TRUE(controlPitches.count(66));
    EXPECT_FALSE(controlPitches.count(70)) << "its onset is still ahead of the clock";
    EXPECT_GE(rig.rt.streamForTest().pendingCount(), 1u);
}

TEST(AnticipatedReduction, aHeldNoteIsUsedButNotCommittedUntilNoteOff) {
    Rig rig;
    rig.playNote(1000, 60, 80);
    ASSERT_TRUE(rig.rt.stepOnce());
    rig.now = 1100;
    rig.playNote(1050, 64, 0, /*held*/ true);
    // make the held key past due, then keep generating
    for (int i = 0; i < 6; ++i)
        ASSERT_TRUE(rig.rt.stepOnce());
    const auto *cache = rig.rt.cacheForTest();
    ASSERT_NE(cache, nullptr);
    const auto tokens = rig.rt.streamForTest().tokens();
    EXPECT_LT(rig.rt.streamForTest().committedLen(), tokens.size()) << "held key blocks the commit";
    EXPECT_LT(cache->committedFed(), tokens.size());
    EXPECT_GT(cache->scratchFed(), 0u);
    const size_t committedBefore = cache->committedFed();

    // note-off arrives
    TokenUpdate update;
    update.oldNote = Token{1050, static_cast<int32_t>(Vocab::DurOffset) + rig.config.inputDuration,
                           static_cast<int32_t>(Vocab::NoteOffset) + 64};
    update.newNote = Token{1050, static_cast<int32_t>(Vocab::DurOffset) + 120,
                           static_cast<int32_t>(Vocab::NoteOffset) + 64};
    rig.rt.updatesFromFilter.push(update);
    ASSERT_TRUE(rig.rt.stepOnce());
    const auto after = rig.rt.streamForTest().tokens();
    EXPECT_EQ(rig.rt.streamForTest().committedLen(), after.size()) << "nothing is held any more";
    EXPECT_GT(rig.rt.cacheForTest()->committedFed(), committedBefore)
        << "the cache advanced over the confirmed tail";
}

TEST(AnticipatedReduction, tdTokensFollowTheLaunchpad) {
    Rig rig;
    rig.tdValue = 16;
    rig.playNote(1000, 60, 80);
    ASSERT_TRUE(rig.rt.stepOnce());
    const auto tokens = rig.rt.streamForTest().tokens();
    bool seen = false;
    for (const auto id: tokens)
        if (isTd(id)) {
            seen = true;
            EXPECT_EQ(id, TdTokens::denseIdForLevel(16, 11776));
        }
    EXPECT_TRUE(seen);

    // AUTO repeats the last level the user chose
    rig.tdValue = TemporalDensity::kAuto;
    for (int i = 0; i < 5; ++i)
        (void) rig.rt.stepOnce();
    for (const auto id: rig.rt.streamForTest().tokens())
        if (isTd(id))
            EXPECT_EQ(id, TdTokens::denseIdForLevel(16, 11776));
}

TEST(AnticipatedReduction, level64FallsBackWithTheSmallerVocab) {
    Rig rig(11775);
    rig.tdValue = 64;
    rig.playNote(1000, 60, 80);
    ASSERT_TRUE(rig.rt.stepOnce());
    for (const auto id: rig.rt.streamForTest().tokens())
        if (isTd(id))
            EXPECT_EQ(id, 11774);
}

TEST(AnticipatedReduction, polyphonyCapOneMakesTheReductionMonophonic) {
    Rig rig;
    rig.rt.setMaxPolyphony(1);
    rig.playNote(1000, 60, 80);
    std::vector<Token> all;
    for (int i = 0; i < 10; ++i) {
        ASSERT_TRUE(rig.rt.stepOnce()) << i;
        for (const auto &t: rig.drainOutput())
            all.push_back(t);
        if (rig.rt.streamForTest().clock() > 6000)
            break;
    }
    ASSERT_GE(all.size(), 3u);
    for (size_t i = 1; i < all.size(); ++i)
        EXPECT_GE(all[i].time, all[i - 1].time + all[i - 1].getRealDuration())
            << "note " << i << " starts while the previous one still sounds";
}

TEST(AnticipatedReduction, streamsLongerThanOneHundredSecondsRebase) {
    Rig rig;
    rig.playNote(1000, 60, 80);
    for (int i = 0; i < 400; ++i) {
        if (! rig.rt.stepOnce())
            break;
        rig.now += 150;
        if (i % 7 == 0)
            rig.playNote(rig.now, 55 + i % 20, 60);
        (void) rig.drainOutput();
        if (rig.rt.streamForTest().base() > 20000)
            break;
    }
    EXPECT_GT(rig.rt.streamForTest().base(), 20000) << "the origin moved forward at least 3 times";
    EXPECT_EQ(rig.rt.streamForTest().base() % TdTokens::kBoundaryCs, 0);
    for (const auto id: rig.rt.streamForTest().tokens()) {
        EXPECT_GE(id, 0);
        EXPECT_LT(id, 11776);
    }
}

TEST(AnticipatedReduction, theCacheEvictingAnOldNoteRebasesAndOutputTimesStayAbsoluteAndContinuous) {
    Rig rig;
    // Accompaniment rules: input never rewinds (in Jam it cuts the future and regenerates).
    rig.rt.setKeepQueueOnInput(true);
    rig.playNote(1000, 60, 80);
    ASSERT_TRUE(rig.rt.stepOnce()); // starts the stream
    int32_t previousBase = rig.rt.streamForTest().base();
    int32_t previousOnset = 0;
    (void) rig.drainOutput();
    bool earlyRebase = false;
    int rebases = 0;
    for (int i = 0; i < 500 && rebases < 3; ++i) {
        const int32_t clockBefore = rig.rt.streamForTest().clock();
        ASSERT_TRUE(rig.rt.stepOnce()) << i;
        rig.now += 60;
        if (i % 3 == 0)
            rig.playNote(rig.now - 10, 50 + i % 24, 40);
        for (const auto &t: rig.drainOutput()) {
            EXPECT_GE(t.time, previousOnset) << "absolute output times never go back";
            previousOnset = t.time;
        }
        const int32_t base = rig.rt.streamForTest().base();
        if (base != previousBase) {
            ++rebases;
            EXPECT_EQ((base - previousBase) % 3000, 0);
            if (clockBefore - previousBase < 8900)
                earlyRebase = true; // not the 90 s safety net: the eviction rule
            previousBase = base;
        }
        // every time token the model sees is relative to the origin and in range
        for (const auto id: rig.rt.streamForTest().tokens())
            EXPECT_LT(id, 11776);
    }
    EXPECT_GE(rebases, 1);
    EXPECT_TRUE(earlyRebase) << "the first rebase came from an eviction, before 90 s of context";
}

TEST(AnticipatedReduction, pausedGenerationProducesNothing) {
    Rig rig;
    rig.playNote(1000, 60, 80);
    rig.rt.generationPause.set(true);
    EXPECT_FALSE(rig.rt.stepOnce());
    EXPECT_TRUE(rig.drainOutput().empty());
    rig.rt.generationPause.set(false);
    EXPECT_TRUE(rig.rt.stepOnce());
}

} // namespace virtual_orch_test

// Optional: the real T3 KV export (VIRTUAL_ORCH_TEST_MODELS=<dir with the .onnx>). Checks that
// feeding a stream in pieces gives the same logits as one prefill, that the committed + scratch
// cache matches a plain cache, and that the transformer runs end to end on the real graph.
#include <cstdlib>
#include <filesystem>

namespace virtual_orch_test {

TEST(AnticipatedReductionReal, kvPiecesMatchOnePrefillAndTheTransformerRuns) {
    const char *dir = std::getenv("VIRTUAL_ORCH_TEST_MODELS");
    if (dir == nullptr)
        GTEST_SKIP() << "VIRTUAL_ORCH_TEST_MODELS not set";
    const auto path = std::filesystem::path(dir) / "maestro_rope_td_T3_dummy_from_rope_ckpt6000_kv.onnx";
    if (! std::filesystem::exists(path))
        GTEST_SKIP() << "no real T3 export";

    OnnxBackendHolder holder;
    std::string error;
    if (! holder.backend.init(path.string().c_str(), error))
        GTEST_SKIP() << "this ONNX Runtime cannot load the export: " << error;
    ASSERT_EQ(holder.backend.vocabSize(), 11775u);

    // A plausible stream: quads of controls/events and a TD token.
    std::vector<int32_t> tokens;
    for (int i = 0; i < 20; ++i) {
        tokens.insert(tokens.end(), {i * 10, 10000 + 50, 11000 + 128 + 60 + i % 12, 11640 + 64});
        if (i % 5 == 0)
            tokens.push_back(11772);
    }

    CommittedScratchCache oneShot(holder.backend, 512);
    const auto reference = oneShot.logitsAfter(tokens, tokens.size());
    ASSERT_EQ(reference.size(), 11775u);

    CommittedScratchCache pieces(holder.backend, 1);
    const auto incremental = pieces.logitsAfter(tokens, tokens.size());
    ASSERT_EQ(incremental.size(), reference.size());
    float maxDiff = 0.0f;
    for (size_t i = 0; i < reference.size(); ++i)
        maxDiff = std::max(maxDiff, std::abs(reference[i] - incremental[i]));
    EXPECT_LT(maxDiff, 2e-3f);

    // 12 committed tokens, the rest as scratch: same logits as everything committed
    CommittedScratchCache split(holder.backend, 8);
    const auto withScratch = split.logitsAfter(tokens, 12);
    ASSERT_EQ(withScratch.size(), reference.size());
    maxDiff = 0.0f;
    for (size_t i = 0; i < reference.size(); ++i)
        maxDiff = std::max(maxDiff, std::abs(reference[i] - withScratch[i]));
    EXPECT_LT(maxDiff, 2e-3f);
    EXPECT_EQ(split.committedFed(), 12u);

    // the whole transformer on the real graph
    ModelConfig config;
    TwoInstrumentAnticipatedReductionTransformer rt(config);
    ASSERT_TRUE(rt.init(path.string().c_str()));
    int32_t now = 1000;
    rt.nowCsProvider = [&] { return now; };
    Token t{990, static_cast<int32_t>(Vocab::DurOffset) + 80, static_cast<int32_t>(Vocab::NoteOffset) + 60};
    t.velocity = 90;
    rt.inputTokenQueue.push(t);
    for (int i = 0; i < 4; ++i)
        EXPECT_TRUE(rt.stepOnce()) << i;
    Token out{};
    int count = 0;
    while (rt.outputTokenQueue.pull(out)) {
        ++count;
        EXPECT_GE(out.getPitch(), 21);
        EXPECT_LE(out.getPitch(), 108);
    }
    EXPECT_EQ(count, 4);
}

} // namespace virtual_orch_test

namespace virtual_orch_test {

TEST(AnticipatedReduction, thePianoNoteComesFirstAtATieByDefault) {
    Rig rig;
    EXPECT_FALSE(rig.rt.getTieFirstGenerated()) << "default: given (piano) first at a shared onset";
    rig.rt.setTieFirstGenerated(true);
    EXPECT_TRUE(rig.rt.getTieFirstGenerated());
    rig.rt.setTieFirstGenerated(false);
    EXPECT_FALSE(rig.rt.getTieFirstGenerated());

    // A note at exactly the event clock is written before the next generated note.
    rig.playNote(1000, 60, 20);
    ASSERT_TRUE(rig.rt.stepOnce());
    const auto tokens = rig.rt.streamForTest().tokens();
    ASSERT_GE(tokens.size(), 8u);
    EXPECT_EQ((tokens[2] - 11000) / 128, 0) << "the piano note is the first quad";
}

} // namespace virtual_orch_test
