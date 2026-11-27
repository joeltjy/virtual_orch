#include <gtest/gtest.h>

#include <JuceHeader.h>

#include <algorithm>
#include <memory>

#include <VirtualOrch/reduction/ChordTheory.h>
#include <VirtualOrch/reduction/GroupChord.h>
#include <VirtualOrch/reduction/GroupTokenEngine.h>
#include <VirtualOrch/reduction/HybridGroupReductionTransformer.h>
#include <VirtualOrch/reduction/StreamingKvCache.h>

/**
 * Hybrid v1.4: GroupTokenEngine (v1.2's engine with the v1.4 group tokens) against a scripted backend: the WAIT block
 * [SDUR][BASS][SHAPE x N][SVEL] at a fire (precomputed or decoded), inference when no candidate matches, free groups
 * [onset][dur][BASS][SHAPE x N][vel], and that a group leaves the engine as ONE note carrying its shape.
 */
namespace {

using Engine = GroupTokenEngine;
using namespace StreamingKv;

constexpr int32_t kBass = 48;
const int32_t kMajRoot3 = GroupChord::offsetOf({0, 4, 7}, 3); // C E G over the bass

/** Scripted m2_v4: the logits depend on the last token only; the engine's masks pick the right field. */
class GroupScript : public Backend {
public:
    GroupScript() {
        d.layers = 1;
        d.heads = 1;
        d.headDim = 1;
        d.maxRows = 512;
        d.sinkRows = 4;
    }
    [[nodiscard]] auto dims() const -> Dims override { return d; }

    auto logitsAfter(int64_t last) const -> std::vector<float> {
        std::vector<float> l(Engine::kGroupVocab, 0.0f);
        const auto set = [&](int32_t id, float v) { l[static_cast<size_t>(id)] = v; };
        const auto t = static_cast<int32_t>(last);
        const bool velocity = t >= Engine::kVel0 && t < Engine::kWait0;
        const bool kTok = t >= Engine::kK0 && t < Engine::kChordExpired;
        const bool shape = t >= Engine::kShape0Group && t < Engine::kGroupVocab;
        if (velocity || kTok) {
            // the onset step (a WAIT or an onset), or the block's SDUR after the target piano quad
            if (waitD > 0)
                set(Engine::waitToken(waitD), 20.0f);
            set(onsetRel, 15.0f);
            set(Engine::kDur0 + 80, 20.0f);
            set(Engine::kChordExpired, -10.0f);
        } else if (t >= Engine::kWait0 && t < Engine::kK0) {
            set(Engine::kToken(k, false), 20.0f);
        } else if (t <= Engine::kOnsetMax) {
            set(Engine::kDur0 + 50, 20.0f); // a free note's duration
        } else if ((t >= Engine::kDur0 && t < Engine::kPiano0) || t == Engine::kDurUnknown) {
            const int pianoPitches[] = {60, 62, 64, 65, 67};
            for (int i = 0; i < 5; ++i)
                set(Engine::kPiano0 + pianoPitches[i], 10.0f - static_cast<float>(i));
            set(Engine::kOrch0 + kBass, 20.0f); // BASS / a free note's pitch
        } else if (t >= Engine::kOrch0 && t < Engine::kVel0) {
            set(Engine::kShape0Group + shapeOffset, 20.0f); // after the bass: the group's shape
            set(Engine::kVel0 + 64, 10.0f);
        } else if (shape) {
            set(Engine::kVel0 + 64, 20.0f);
        } else if (t >= Engine::kPiano0 && t < Engine::kOrch0) {
            set(Engine::kVel0 + 64, 20.0f);
        }
        return l;
    }

    auto run(const State &past, const std::vector<int64_t> &ids, State &present, std::vector<float> &out) -> bool override {
        present = past;
        for (const auto id: ids) {
            present.key.push_back(static_cast<float>(id));
            present.value.push_back(static_cast<float>(id));
            ++present.rows;
        }
        out = logitsAfter(ids.back());
        return true;
    }

    auto runBatched(const BatchState &past, const std::vector<int64_t> &ids, BatchState &present,
                    std::vector<float> &logits) -> bool override {
        const auto B = static_cast<size_t>(past.batch);
        const auto r = static_cast<size_t>(past.rows);
        const size_t n = ids.size() / B;
        present.batch = past.batch;
        present.rows = past.rows + static_cast<int>(n);
        present.key.clear();
        for (size_t b = 0; b < B; ++b) {
            present.key.insert(present.key.end(), past.key.begin() + static_cast<std::ptrdiff_t>(b * r),
                               past.key.begin() + static_cast<std::ptrdiff_t>((b + 1) * r));
            for (size_t i = 0; i < n; ++i)
                present.key.push_back(static_cast<float>(ids[b * n + i]));
        }
        present.value = present.key;
        logits.clear();
        for (size_t b = 0; b < B; ++b) {
            const auto l = logitsAfter(ids[b * n + n - 1]);
            logits.insert(logits.end(), l.begin(), l.end());
        }
        ++batchedCalls;
        return true;
    }

    Dims d;
    int32_t waitD = 30;
    int32_t onsetRel = 10;
    int32_t k = 3;
    int32_t shapeOffset = kMajRoot3;
    int batchedCalls = 0;
};

struct GroupRig {
    GroupScript model;
    Engine engine;
    std::vector<Engine::OutNote> played;

    explicit GroupRig(bool precompute = true) {
        engine.setBackend(&model);
        Engine::Settings s;
        s.chordTokens = true;
        s.precompute = precompute;
        s.seed = 3;
        engine.setSettings(s);
        engine.reset(0);
    }
    auto run(int32_t from, int32_t to) -> void {
        for (int32_t t = from; t <= to; ++t) {
            auto out = engine.step(t);
            played.insert(played.end(), out.begin(), out.end());
        }
    }
};

/** The block [SDUR][BASS][SHAPE][SVEL] right after the target piano quad at `at` (written [DUR?]). */
auto expectGroupBlockAfterPianoQuad(const std::vector<int32_t> &toks, int at, int32_t pianoPitch) -> void {
    ASSERT_GE(static_cast<int>(toks.size()), at + 8);
    EXPECT_EQ(toks[static_cast<size_t>(at + 1)], Engine::kDurUnknown);
    EXPECT_EQ(toks[static_cast<size_t>(at + 2)], Engine::kPiano0 + pianoPitch);
    EXPECT_EQ(toks[static_cast<size_t>(at + 4)], Engine::kDur0 + 80);
    EXPECT_EQ(toks[static_cast<size_t>(at + 5)], Engine::kOrch0 + kBass);
    EXPECT_EQ(toks[static_cast<size_t>(at + 6)], Engine::kShape0Group + kMajRoot3);
    EXPECT_EQ(toks[static_cast<size_t>(at + 7)], Engine::kVel0 + 64);
}

} // namespace

TEST(GroupTokenEngine, FireServedFromThePrecomputedGroupBlock) {
    GroupRig rig;
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 6); // the group closes at 5; [WAIT 30][K 3]
    ASSERT_TRUE(rig.engine.hasPendingWait());
    rig.run(7, 25); // anticipation + the batched block stages (SDUR, BASS, SHAPE, SVEL)
    ASSERT_TRUE(rig.engine.precomputeComplete());
    EXPECT_TRUE(rig.played.empty()); // silent while the WAIT is pending

    rig.engine.noteOn(30, 60, 80);
    const auto out = rig.engine.step(30);
    ASSERT_EQ(out.size(), 1u); // the group goes out as ONE note: its bass + shape, not decoded
    EXPECT_EQ(out[0].pitch, kBass);
    EXPECT_EQ(out[0].shape, kMajRoot3);
    EXPECT_EQ(out[0].onsetCs, 30);
    EXPECT_EQ(out[0].durationCs, 80); // SDUR
    EXPECT_EQ(out[0].velocity, 64);   // SVEL
    EXPECT_TRUE(out[0].precomputed);
    EXPECT_TRUE(out[0].fromChord);
    EXPECT_EQ(rig.engine.stats().precomputeHits, 1);
    EXPECT_EQ(rig.engine.stats().chordBlocks, 1);
    // context: piano quad, WAIT, K, target piano quad ([DUR?]), the 4 block tokens
    const auto toks = rig.engine.tokens();
    ASSERT_EQ(toks.size(), 4u + 2u + 4u + 4u);
    EXPECT_EQ(toks[4], Engine::waitToken(30));
    EXPECT_EQ(toks[5], Engine::kToken(3, false));
    EXPECT_EQ(toks[6], 30);
    expectGroupBlockAfterPianoQuad(toks, 6, 60);
}

TEST(GroupTokenEngine, FireDecodedOnTheSpotWithoutPrecompute) {
    GroupRig rig(false);
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 20);
    ASSERT_TRUE(rig.engine.hasPendingWait());
    rig.engine.noteOn(30, 64, 80);
    rig.run(30, 33); // the fire, then the block decoded one step later
    ASSERT_EQ(rig.played.size(), 1u);
    EXPECT_EQ(rig.played[0].pitch, kBass);
    EXPECT_EQ(rig.played[0].shape, kMajRoot3);
    EXPECT_FALSE(rig.played[0].precomputed);
    expectGroupBlockAfterPianoQuad(rig.engine.tokens(), 6, 64);
}

TEST(GroupTokenEngine, NoMatchingCandidateInfersAChordThatFitsThePiano) {
    GroupRig rig;
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 25);
    ASSERT_TRUE(rig.engine.precomputeComplete());
    // F# A C# at the fire: C E G (the candidates' chord) with them is no triad or seventh -> inferred
    rig.engine.noteOn(30, 54, 80);
    rig.engine.noteOn(30, 57, 80);
    rig.engine.noteOn(30, 61, 80);
    rig.run(30, 40);
    ASSERT_EQ(rig.played.size(), 1u);
    const auto &n = rig.played[0];
    const uint16_t mask = n.shape >= 0 ? GroupChord::pitchClassMask(n.pitch, n.shape)
                                       : static_cast<uint16_t>(1u << (n.pitch % 12));
    const auto piano = ChordTheory::maskOf({54, 57, 61});
    EXPECT_TRUE(ChordTheory::recognized(static_cast<ChordTheory::Pcs>(piano | mask))) << "mask " << mask;
    EXPECT_EQ(rig.engine.stats().precomputeHits, 0);
}

TEST(GroupTokenEngine, AFreeGroupIsOneNoteWithItsShape) {
    GroupRig rig;
    rig.model.waitD = 0; // no WAIT: the onset step draws a free onset
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 12);
    ASSERT_FALSE(rig.played.empty());
    const auto &n = rig.played.front();
    EXPECT_EQ(n.pitch, kBass);
    EXPECT_EQ(n.shape, kMajRoot3);
    EXPECT_FALSE(n.fromChord);
    // the context has the group as 5 tokens: [onset][dur][BASS][SHAPE][vel]
    const auto toks = rig.engine.tokens();
    const auto it = std::find(toks.begin(), toks.end(), Engine::kShape0Group + kMajRoot3);
    ASSERT_NE(it, toks.end());
    EXPECT_EQ(*(it - 1), Engine::kOrch0 + kBass);
    EXPECT_EQ(*(it + 1), Engine::kVel0 + 64);
}

/** waitCeiling (2026-11-27): the scripted onset step puts ~99% on [WAIT 30]; capped at 0.5, about half the first
 *  onset-step draws over many seeds are a free onset instead (off: all WAITs). */
TEST(GroupTokenEngine, WaitCeilingCapsTheWaitShareAtTheOnsetStep) {
    const auto waits = [](float ceiling) {
        int n = 0;
        for (uint64_t seed = 1; seed <= 200; ++seed) {
            GroupRig rig;
            auto s = rig.engine.settings();
            s.seed = seed;
            s.waitCeiling = ceiling;
            rig.engine.setSettings(s);
            rig.engine.reset(0);
            rig.engine.noteOn(0, 60, 80);
            rig.run(0, 6);
            // the first onset-step token, right after the piano quad (a free note may be followed by a WAIT)
            const auto toks = rig.engine.tokens();
            n += toks.size() > 4 && toks[4] >= Engine::kWait0 && toks[4] < Engine::kK0 ? 1 : 0;
        }
        return n;
    };
    EXPECT_GE(waits(0.0f), 195);
    const int capped = waits(0.5f);
    EXPECT_GE(capped, 70);
    EXPECT_LE(capped, 130);
}

TEST(GroupTokenEngine, HybridV14LoadsTheGroupModels) {
    const auto dir = juce::File::getSpecialLocation(juce::File::userHomeDirectory).getChildFile("Documents/virtual_orch_models");
    const auto m1 = dir.getChildFile("rt_continuation_chordal.onnx");
    const auto m2 = dir.getChildFile("rt_simultaneous_chordal.onnx");
    const auto bank = dir.getChildFile("voicing_bank_v4.json");
    if (! m1.existsAsFile() || ! m2.existsAsFile() || ! bank.existsAsFile())
        GTEST_SKIP() << "rt_continuation_chordal / rt_simultaneous_chordal / voicing_bank_v4 not on this machine";
    ModelConfig config;
    HybridGroupReductionTransformer hy(config);
    ASSERT_TRUE(hy.loadVoicingBank(bank.getFullPathName().toStdString())) << hy.lastInitError;
    ASSERT_TRUE(hy.init(m1.getFullPathName().toRawUTF8(), m2.getFullPathName().toRawUTF8())) << hy.lastInitError;
    EXPECT_TRUE(hy.isModelLoaded());
    EXPECT_TRUE(hy.isChordModel());
    EXPECT_STREQ(hy.label(), "Hybrid v1.4");
    // the OT / playback decode with the same bank: a C major 4-note group over C3 gives 4 pitches with C on the bottom
    Token t{0, static_cast<int32_t>(Vocab::DurOffset) + 50, static_cast<int32_t>(Vocab::NoteOffset) + 48, 80};
    t.noteId = 1;
    t.chordShape = GroupChord::offsetOf({0, 4, 7}, 4);
    const auto notes = GroupChord::expand(t);
    ASSERT_EQ(notes.size(), 4u);
    EXPECT_EQ(notes.front().getPitch(), 48);
    GroupChord::setSharedBank(nullptr);
}
