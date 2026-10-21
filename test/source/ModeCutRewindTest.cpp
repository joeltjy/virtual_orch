#include <gtest/gtest.h>

#include <random>

#include <VirtualOrch/PedalClear.h>
#include <VirtualOrch/reduction/AnticipatedStream.h>
#include <VirtualOrch/reduction/MelodyFirstReductionTransformer.h>
#include <VirtualOrch/reduction/TwoInstrumentAnticipatedReductionTransformer.h>

// Leaving Accompaniment rewinds the KV-cached anticipation models to the cut time: the result must
// be the same as if the notes from the cut on had never been generated.

namespace {

using namespace StreamingKv;
using Spec = AnticipatedStream::ControlSpec;

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

auto lineSpec(int line) -> Spec {
    Spec s;
    s.line = line;
    return s;
}

/** Note quads of a stream (TD / ANTICIPATE skipped): absolute onset, instrument, control flag. */
struct Quad {
    int32_t onsetAbs;
    int32_t instrument;
    bool control; // AMT control ids (the T7'' melody controls)
    bool rest;
};
auto quadsOf(const AnticipatedStream &s) -> std::vector<Quad> {
    const auto t = s.tokens();
    std::vector<Quad> out;
    for (size_t i = 0; i + 3 < t.size();) {
        const int32_t id = t[i];
        if (id < 10000) { // event or rest quad: time first
            const bool rest = t[i + 2] == AnticipatedStream::kRestId;
            const int32_t note = t[i + 2] - static_cast<int32_t>(DenseVocab::NoteOffset);
            out.push_back({id + s.base(), rest ? -1 : note / DenseConfig::MaxPitch, false, rest});
            i += 4;
        } else if (id >= AnticipatedStream::kAmtTimeCtl && id < AnticipatedStream::kAmtDurCtl) {
            const int32_t note = t[i + 2] - AnticipatedStream::kAmtNoteCtl;
            out.push_back({id - AnticipatedStream::kAmtTimeCtl + s.base(), note / 128, true, false});
            i += 4;
        } else {
            ++i; // TD or ANTICIPATE
        }
    }
    return out;
}

} // namespace

namespace virtual_orch_test {

// ---- AnticipatedStream ----------------------------------------------------------------------

TEST(ModeCutRewind, streamRewindEqualsNeverGenerated) {
    AnticipatedStream a;
    a.reset(0);
    a.addControl({100, 50, 64, 64}, false, lineSpec(0));
    a.appendEvent({50, 40, 60, 64});
    a.insertDue(a.clock(), 0);
    a.appendEvent({150, 40, 61, 64});
    a.insertDue(a.clock(), 0);            // given 100 (line 0)
    a.appendEvent({300, 40, 62, 64});     // generated ahead, crosses boundary 200
    a.addControl({320, 50, 66, 64}, false, lineSpec(0));
    a.appendEvent({450, 40, 63, 64});     // crosses 400
    a.insertDue(a.clock(), 0);            // given 320 lands after the generated notes

    AnticipatedStream b; // the same history without the notes from 250 on
    b.reset(0);
    b.addControl({100, 50, 64, 64}, false, lineSpec(0));
    b.appendEvent({50, 40, 60, 64});
    b.insertDue(b.clock(), 0);
    b.appendEvent({150, 40, 61, 64});
    b.insertDue(b.clock(), 0);
    b.addControl({320, 50, 66, 64}, false, lineSpec(0));

    EXPECT_TRUE(a.rewindTo(250));
    EXPECT_EQ(a.tokens(), b.tokens());
    EXPECT_EQ(a.clock(), b.clock());
    EXPECT_EQ(a.clock(), 150);
    EXPECT_EQ(a.pendingCount(), b.pendingCount()) << "given 320 is pending again";
    EXPECT_TRUE(a.eventEmittedAt(150));

    // Both continue the same way: TD 200 written again, given 320 inserted when due (its line's
    // monophony end is the recomputed one).
    for (auto *s: {&a, &b}) {
        s->appendEvent({260, 40, 65, 64});
        s->appendEvent({330, 40, 67, 64});
        s->insertDue(s->clock(), 0);
        s->addControl({340, 50, 69, 64}, false, lineSpec(0));
        s->appendEvent({360, 40, 70, 64});
        s->insertDue(s->clock(), 0);
    }
    EXPECT_EQ(a.tokens(), b.tokens());
    EXPECT_FALSE(a.rewindTo(1000)) << "nothing at or after 1000";
}

TEST(ModeCutRewind, amtRestsAfterTheCutAreDroppedAndTheClockFollowsTheKeptOnes) {
    AnticipatedStream a(AnticipatedStream::Vocab::Amt);
    a.reset(0);
    a.appendEvent({50, 40, 60, 64});
    a.appendRest(150);
    a.appendRest(250);
    a.appendEvent({280, 40, 61, 64});

    AnticipatedStream b(AnticipatedStream::Vocab::Amt);
    b.reset(0);
    b.appendEvent({50, 40, 60, 64});
    b.appendRest(150);

    EXPECT_TRUE(a.rewindTo(200));
    EXPECT_EQ(a.tokens(), b.tokens());
    EXPECT_EQ(a.clock(), 150);
    EXPECT_TRUE(a.eventEmittedAt(50));
    EXPECT_FALSE(a.eventEmittedAt(280));
}

TEST(ModeCutRewind, droppedControlsAreRemovedEverywhere) {
    Spec melody;
    melody.deltaCs = 300;
    melody.controlToken = true;
    AnticipatedStream a(AnticipatedStream::Vocab::Amt);
    a.reset(0);
    a.appendEvent({50, 40, 60, 64});
    a.addControl({400, 50, 70, 64}, false, melody); // written once the clock passes 100
    a.addControl({600, 50, 71, 64}, false, melody); // still pending
    a.appendEvent({150, 40, 61, 64});
    a.insertDue(a.clock(), 0);

    AnticipatedStream b(AnticipatedStream::Vocab::Amt);
    b.reset(0);
    b.appendEvent({50, 40, 60, 64});
    b.appendEvent({150, 40, 61, 64});

    const int32_t cut = 250;
    EXPECT_TRUE(a.rewindTo(cut, [cut](const Spec &spec, const AnticipatedStream::Note &n) {
        return spec.controlToken && n.onsetCs >= cut;
    }));
    EXPECT_EQ(a.tokens(), b.tokens()) << "the early-written control of 400 is gone";
    EXPECT_EQ(a.pendingCount(), 0u) << "the pending control of 600 is gone";
    EXPECT_EQ(a.clock(), 150);
}

// ---- TwoInstrumentAnticipated ---------------------------------------------------------------

TEST(ModeCutRewind, anticipatedReductionRewindsToTheSnapshotBeforeTheCut) {
    ModelConfig config;
    TwoInstrumentAnticipatedReductionTransformer rt{config};
    rt.initWithBackend(std::make_unique<FakeBackend>(11776, 512), 11776);
    int32_t now = 1000;
    rt.nowCsProvider = [&now] { return now; };
    rt.inputTokenQueue.push(keyToken(900, 64, 80));

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
    rt.initWithBackends(std::make_unique<FakeBackend>(11776, 512), 11776,
                        std::make_unique<FakeBackend>(55293, 512), 55293);
    int32_t now = 1000;
    rt.nowCsProvider = [&now] { return now; };
    EXPECT_FALSE(rt.stepOnce()); // start the streams at 1000
    now = 1100;
    rt.inputTokenQueue.push(keyToken(1000, 64, 40));

    std::vector<Token> notes;
    for (int i = 0; i < 12; ++i) {
        rt.stepOnce();
        now += 10;
        Token t{};
        while (rt.outputTokenQueue.pull(t))
            if (t.note != static_cast<int32_t>(Vocab::ClearQueue))
                notes.push_back(t);
    }
    ASSERT_GE(notes.size(), 4u);
    const int32_t cut = notes[notes.size() / 2].time;
    bool melodyAhead = false;
    for (const auto &m: rt.melodyEventsForTest())
        melodyAhead = melodyAhead || m.time >= cut;

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
    for (const auto &q: quadsOf(rt.melodyStreamForTest()))
        if (q.instrument == AnticipatedStream::kEventInstrument)
            EXPECT_LT(q.onsetAbs, cut) << "no T8'' note at or after the cut";
    for (const auto &q: quadsOf(rt.reductionStreamForTest())) {
        if (q.rest || q.instrument == AnticipatedStream::kEventInstrument)
            EXPECT_LT(q.onsetAbs, cut) << "no T7'' note or rest at or after the cut";
        if (q.control)
            EXPECT_LT(q.onsetAbs, cut) << "no melody control at or after the cut";
    }
    bool marker = false;
    while (rt.outputTokenQueue.pull(drained))
        marker = marker || PedalClear::isPedalClear(drained);
    EXPECT_TRUE(marker);
    for (const auto &t: rt.getOutputHistory())
        if (t.note != static_cast<int32_t>(Vocab::ClearQueue))
            EXPECT_LT(t.time, cut);
    juce::ignoreUnused(melodyAhead);

    rt.setGenerationPause(false);
    EXPECT_TRUE(rt.stepOnce()) << "generation resumes (both caches rebuilt)";
}

} // namespace virtual_orch_test
