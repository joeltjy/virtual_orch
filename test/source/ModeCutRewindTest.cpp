#include <gtest/gtest.h>

#include <random>

#include <VirtualOrch/PedalClear.h>
#include <VirtualOrch/reduction/AnticipatedStream.h>
#include <VirtualOrch/reduction/MelodyFirstReductionTransformer.h>
#include <VirtualOrch/reduction/MelodyStream.h>
#include <VirtualOrch/reduction/TwoInstrumentAnticipatedReductionTransformer.h>

// Leaving Accompaniment rewinds the KV-cached reduction models to the cut time: the result must
// be the same as if the notes from the cut on had never been generated.

namespace {

using namespace StreamingKv;
using ANote = AnticipatedStream::Note;

/** Pseudo-random logits seeded by the cache contents, preferring early onsets. */
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
        uint32_t seed = 41;
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

auto keyToken(int32_t onset, int pitch, int32_t durCs) -> Token {
    Token t{onset, static_cast<int32_t>(Vocab::DurOffset) + durCs,
            static_cast<int32_t>(Vocab::NoteOffset) + pitch};
    t.velocity = 90;
    return t;
}

/** Quads of a dense anticipated stream (TD tokens skipped): absolute onset + instrument. */
struct Quad {
    int32_t onsetAbs;
    int32_t instrument;
};
auto quadsOf(const AnticipatedStream &s) -> std::vector<Quad> {
    const auto t = s.tokens();
    std::vector<Quad> out;
    for (size_t i = 0; i + 3 < t.size();) {
        if (t[i] >= static_cast<int32_t>(DenseVocab::VocabSize)) { // TD token (ids past the dense vocab)
            ++i;
            continue;
        }
        const int32_t note = t[i + 2] - static_cast<int32_t>(DenseVocab::NoteOffset);
        out.push_back({t[i] + s.base(), note / DenseConfig::MaxPitch});
        i += 4;
    }
    return out;
}

} // namespace

namespace virtual_orch_test {

// ---- AnticipatedStream ----------------------------------------------------------------------

TEST(ModeCutRewind, anticipatedStreamRewindEqualsNeverGenerated) {
    const auto tdA = [] { return 11770; };
    AnticipatedStream a;
    a.tdTokenProvider = tdA;
    a.reset(0);
    a.addControl({100, 50, 64, 64}, false);
    a.appendEvent({50, 40, 60, 64});
    a.insertDue(a.clock(), 0);
    a.appendEvent({150, 40, 61, 64});
    a.insertDue(a.clock(), 0); // control 100
    a.appendEvent({300, 40, 62, 64}); // crosses boundary 200: generated ahead
    a.addControl({320, 50, 66, 64}, false);
    a.appendEvent({450, 40, 63, 64}); // crosses 400
    a.insertDue(a.clock(), 0); // control 320 lands after the generated notes

    AnticipatedStream b; // the same history without the notes from 250 on
    b.tdTokenProvider = tdA;
    b.reset(0);
    b.addControl({100, 50, 64, 64}, false);
    b.appendEvent({50, 40, 60, 64});
    b.insertDue(b.clock(), 0);
    b.appendEvent({150, 40, 61, 64});
    b.insertDue(b.clock(), 0);
    b.addControl({320, 50, 66, 64}, false);

    EXPECT_TRUE(a.rewindTo(250));
    EXPECT_EQ(a.tokens(), b.tokens());
    EXPECT_EQ(a.clock(), b.clock());
    EXPECT_EQ(a.clock(), 150);
    EXPECT_EQ(a.pendingCount(), b.pendingCount()) << "control 320 is pending again";

    // Both continue the same way (TD 200 is written again, control 320 inserted when due).
    for (auto *s: {&a, &b}) {
        s->appendEvent({260, 40, 65, 64});
        s->appendEvent({330, 40, 67, 64});
        s->insertDue(s->clock(), 0);
    }
    EXPECT_EQ(a.tokens(), b.tokens());
    EXPECT_FALSE(a.rewindTo(1000)) << "nothing at or after 1000";
}

TEST(ModeCutRewind, anticipatedStreamDropsTheRewoundMelodyControlsEverywhere) {
    AnticipatedStream a;
    a.reset(0);
    a.appendEvent({50, 40, 60, 64});
    // M control for a melody note at 400 (delta 300: written once the clock passes 100).
    a.addControl({400, 50, 70, 64}, false, 2, 300, 0);
    // M control for a melody note at 600 (still pending).
    a.addControl({600, 50, 71, 64}, false, 2, 300, 0);
    a.appendEvent({150, 40, 61, 64});
    a.insertDue(a.clock(), 0);

    AnticipatedStream b;
    b.reset(0);
    b.appendEvent({50, 40, 60, 64});
    b.appendEvent({150, 40, 61, 64});

    EXPECT_TRUE(a.rewindTo(250, 2));
    EXPECT_EQ(a.tokens(), b.tokens()) << "the early-written M control of 400 is gone";
    EXPECT_EQ(a.pendingCount(), 0u) << "the pending M control of 600 is gone";
    EXPECT_EQ(a.clock(), 150);
}

// ---- MelodyStream ---------------------------------------------------------------------------

TEST(ModeCutRewind, melodyStreamRewindEqualsNeverGenerated) {
    const auto piece = [](int64_t key, int32_t onset, int32_t dur, int pitch) {
        MelodyBassState::Piece p;
        p.key = key; p.onset = onset; p.durationCs = dur; p.pitch = pitch; p.velocity = 64;
        p.line = 0; p.provisional = false;
        return p;
    };
    const std::vector<MelodyBassState::Piece> r{piece(1, 50, 40, 60), piece(2, 350, 40, 62)};

    MelodyStream a;
    a.reset(0);
    a.syncControls(r);
    a.insertDue(50);
    a.appendEvent({60, 50, 70, 80});
    a.appendRest(160);
    a.appendEvent({300, 50, 72, 80}); // generated ahead
    a.appendEvent({400, 50, 74, 80});
    a.insertDue(a.clock()); // piece 2 written after them

    MelodyStream b;
    b.reset(0);
    b.syncControls(r);
    b.insertDue(50);
    b.appendEvent({60, 50, 70, 80});
    b.appendRest(160);
    b.insertDue(b.clock());

    EXPECT_TRUE(a.rewindTo(250));
    EXPECT_EQ(a.tokens(), b.tokens());
    EXPECT_EQ(a.clock(), b.clock());
    EXPECT_EQ(a.pendingCount(), b.pendingCount());

    // R is re-synced (algorithm A ran again): piece 2 is still offered, piece 1 is not.
    a.syncControls(r);
    b.syncControls(r);
    EXPECT_EQ(a.pendingCount(), 1u);
    for (auto *s: {&a, &b}) {
        s->appendEvent({360, 50, 75, 80});
        s->insertDue(s->clock());
    }
    EXPECT_EQ(a.tokens(), b.tokens());
}

// ---- TwoInstrumentAnticipated ---------------------------------------------------------------

TEST(ModeCutRewind, anticipatedReductionRewindsToTheSnapshotBeforeTheCut) {
    ModelConfig config;
    TwoInstrumentAnticipatedReductionTransformer rt{config};
    rt.initWithBackend(std::make_unique<FakeBackend>(11776, 512), 11776);
    int32_t now = 1000;
    rt.nowCsProvider = [&now] { return now; };
    rt.inputTokenQueue.push(keyToken(950, 64, 80));

    // Generate ahead, keeping the stream as it was before each note.
    std::vector<std::vector<int32_t>> before;
    std::vector<int32_t> clockBefore;
    std::vector<Token> notes;
    for (int i = 0; i < 10; ++i) {
        before.push_back(rt.streamForTest().tokens());
        clockBefore.push_back(rt.streamForTest().clock());
        ASSERT_TRUE(rt.stepOnce());
        Token t{};
        while (rt.outputTokenQueue.pull(t))
            if (t.note != static_cast<int32_t>(Vocab::ClearQueue))
                notes.push_back(t);
    }
    ASSERT_EQ(notes.size(), 10u);
    size_t j = 2;
    while (j < notes.size() && notes[j].time == notes[j - 1].time)
        ++j;
    ASSERT_LT(j, notes.size());
    const int32_t cut = notes[j].time;

    rt.setGenerationPause(true); // look at the state right after the rewind
    Token drained{};
    while (rt.outputTokenQueue.pull(drained)) {
    }
    rt.requestModeCut(cut);
    rt.stepOnce();

    EXPECT_EQ(rt.streamForTest().tokens(), before[j]) << "as if note " << j << " never happened";
    EXPECT_EQ(rt.streamForTest().clock(), clockBefore[j]);
    bool marker = false;
    while (rt.outputTokenQueue.pull(drained))
        marker = marker || PedalClear::isPedalClear(drained);
    EXPECT_TRUE(marker) << "the cut marker goes downstream";
    for (const auto &t: rt.getOutputHistory())
        if (t.note != static_cast<int32_t>(Vocab::ClearQueue))
            EXPECT_LT(t.time, cut);

    rt.setGenerationPause(false);
    EXPECT_TRUE(rt.stepOnce()) << "generation resumes (the cache rebuilt)";
}

// ---- MelodyFirst ----------------------------------------------------------------------------

TEST(ModeCutRewind, melodyFirstRewindsBothTheReductionAndThePlannedMelody) {
    ModelConfig config;
    MelodyFirstReductionTransformer rt{config};
    rt.initWithBackends(std::make_unique<FakeBackend>(55285, 288), 55285,
                        std::make_unique<FakeBackend>(11776, 512), 11776);
    int32_t now = 1000;
    rt.nowCsProvider = [&now] { return now; };
    rt.inputTokenQueue.push(keyToken(950, 64, 80));

    std::vector<Token> notes;
    for (int i = 0; i < 10; ++i) {
        rt.stepOnce();
        Token t{};
        while (rt.outputTokenQueue.pull(t))
            if (t.note != static_cast<int32_t>(Vocab::ClearQueue))
                notes.push_back(t);
    }
    ASSERT_GE(notes.size(), 4u);
    const int32_t cut = notes[notes.size() / 2].time;
    ASSERT_GT(notes.back().time, cut - 1);

    rt.setGenerationPause(true);
    Token drained{};
    while (rt.outputTokenQueue.pull(drained)) {
    }
    rt.requestModeCut(cut);
    rt.stepOnce();

    EXPECT_LT(rt.reductionStreamForTest().clock(), cut);
    EXPECT_LT(rt.melodyStreamForTest().clock(), cut);
    for (const auto &m: rt.melodyEventsForTest())
        EXPECT_LT(m.time, cut) << "the planned melody from the cut on is dropped";
    for (const auto &q: quadsOf(rt.reductionStreamForTest())) {
        if (q.instrument == AnticipatedStream::kEventInstrument || q.instrument == 2)
            EXPECT_LT(q.onsetAbs, cut) << "no reduction note / melody control at or after the cut";
    }
    bool marker = false;
    while (rt.outputTokenQueue.pull(drained))
        marker = marker || PedalClear::isPedalClear(drained);
    EXPECT_TRUE(marker);
    for (const auto &t: rt.getOutputHistory())
        if (t.note != static_cast<int32_t>(Vocab::ClearQueue))
            EXPECT_LT(t.time, cut);

    rt.setGenerationPause(false);
    EXPECT_TRUE(rt.stepOnce()) << "generation resumes (both caches rebuilt)";
}

} // namespace virtual_orch_test
