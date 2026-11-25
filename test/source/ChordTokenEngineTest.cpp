#include <gtest/gtest.h>

#include <JuceHeader.h>

#include <algorithm>
#include <memory>
#include <set>

#include <VirtualOrch/reduction/DenseTypes.h>
#include <VirtualOrch/reduction/HybridInformativeWaitReductionTransformer.h>
#include <VirtualOrch/reduction/InformativeWaitEngine.h>
#include <VirtualOrch/reduction/StreamingKvCache.h>
#include <VirtualOrch/reduction/VoicingBank.h>

/**
 * Hybrid v1.2: the chord-token model 2 (m2c, vocab 14071; orchestration/overnight/pkg/wait3.py, spec B3f) in
 * InformativeWaitEngine's chordTokens mode, against a scripted backend: [WAIT d][K k,+1], the CHORD block at a fire
 * (a matching chord candidate, or inferred from the piano notes), the leftover quad, [LATE d] or [EXPIRED] at a close, then a re-wait or a give-up, the 6 / 12 cs floors and the
 * [DUR?] rule; and the voicing bank.
 */
namespace {

using Engine = InformativeWaitEngine;
using namespace StreamingKv;

constexpr int32_t kMaj = 7; // VoicingBank::kTypeNames index
constexpr int32_t kBass = 48;
// C major (roots are true pitch classes, wait3 since the 2026-11-24 fix)
const int32_t kCMajor3 = Engine::chordToken(kMaj, 0, 3);

/** Scripted m2c: the logits depend on the last token only; the engine's masks pick the right field. */
class ChordScript : public Backend {
public:
    ChordScript() {
        d.layers = 1;
        d.heads = 1;
        d.headDim = 1;
        d.maxRows = 512;
        d.sinkRows = 4;
    }
    [[nodiscard]] auto dims() const -> Dims override { return d; }

    auto logitsAfter(int64_t last) const -> std::vector<float> {
        std::vector<float> l(Engine::kChordVocab, 0.0f);
        const auto set = [&](int32_t id, float v) { l[static_cast<size_t>(id)] = v; };
        const auto t = static_cast<int32_t>(last);
        const bool velocity = t >= Engine::kVel0 && t < Engine::kWait0;
        const bool kTok = t >= Engine::kK0 && t < Engine::kChordExpired;
        const bool late = t >= Engine::kLate0 && t < Engine::kDurUnknown;
        if (t == Engine::kChordExpired) {
            // after [EXPIRED]: a re-wait (WAIT) or a give-up (CHORD)
            if (expiredThenWait)
                set(Engine::waitToken(waitD), 20.0f);
            else
                set(chordTok, 20.0f);
        } else if (velocity || kTok || late) {
            // the onset step (a WAIT or an onset), a CHORD (after the target piano quad), or at a close: LATE or EXPIRED
            if (waitD > 0)
                set(Engine::waitToken(waitD), 20.0f);
            set(onsetRel, 15.0f);
            set(chordTok, 20.0f);
            if (chordTok2 >= 0)
                set(chordTok2, chordLogit2);
            set(Engine::lateToken(lateD), 20.0f);
            set(Engine::kChordExpired, late ? expiredAfterLate : expiredAfterWait);
        } else if (t >= Engine::kWait0 && t < Engine::kK0) {
            set(Engine::kToken(k, plus), 20.0f);
            set(Engine::kToken(k, ! plus), 10.0f);
        } else if (t <= Engine::kOnsetMax) {
            set(Engine::kDur0 + 50, 20.0f); // the leftover's (or an orchestra note's) duration
        } else if (t >= Engine::kChord0) {
            set(Engine::kDur0 + 80, 20.0f); // SDUR
        } else if ((t >= Engine::kDur0 && t < Engine::kPiano0) || t == Engine::kDurUnknown) {
            const int pianoPitches[] = {60, 62, 64, 65, 67};
            for (int i = 0; i < 5; ++i)
                set(Engine::kPiano0 + pianoPitches[i], 10.0f - static_cast<float>(i));
            set(Engine::kOrch0 + kBass, 20.0f); // BASS (and the leftover's pitch)
        } else if (t >= Engine::kPiano0 && t < Engine::kVel0) {
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
        ++calls;
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
    bool plus = false;
    int32_t chordTok = kCMajor3;
    int32_t chordTok2 = -1; // a second chord candidate, with this logit
    float chordLogit2 = 0.0f;
    int32_t lateD = 12;
    float expiredAfterWait = 0.0f; // the [EXPIRED] logit at a WAIT's close (LATE d has 20)
    float expiredAfterLate = 0.0f; // ... at a LATE's close
    bool expiredThenWait = false;  // after [EXPIRED]: WAIT (re-wait) instead of CHORD (give up)
    int calls = 0;
    int batchedCalls = 0;
};

auto testBank() -> std::shared_ptr<VoicingBank> {
    auto bank = std::make_shared<VoicingBank>();
    bank->add(kMaj, 3, 0, {{0, 4, 7}}); // root position: bass_iv 0
    return bank;
}

struct ChordRig {
    ChordScript model;
    Engine engine;
    std::vector<Engine::Event> events;
    std::vector<Engine::OutNote> played;

    explicit ChordRig(bool precompute = true, int32_t grace = 0) {
        engine.setBackend(&model);
        Engine::Settings s;
        s.chordTokens = true;
        s.voicings = testBank();
        s.precompute = precompute;
        s.expiryGraceCs = grace;
        s.seed = 3;
        engine.setSettings(s);
        engine.onEvent = [this](const Engine::Event &e) { events.push_back(e); };
        engine.reset(0);
    }
    auto run(int32_t from, int32_t to) -> void {
        for (int32_t t = from; t <= to; ++t) {
            auto out = engine.step(t);
            played.insert(played.end(), out.begin(), out.end());
        }
    }
    auto count(const std::string &what) const -> int {
        return static_cast<int>(std::count_if(events.begin(), events.end(), [&](const auto &e) { return e.what == what; }));
    }
    /** Index of the first occurrence of `token` in the context, or -1. */
    auto find(int32_t token) const -> int {
        const auto toks = engine.tokens();
        const auto it = std::find(toks.begin(), toks.end(), token);
        return it == toks.end() ? -1 : static_cast<int>(it - toks.begin());
    }
};

auto pitchesOf(const std::vector<Engine::OutNote> &notes) -> std::vector<int32_t> {
    std::vector<int32_t> p;
    for (const auto &n: notes)
        p.push_back(n.pitch);
    std::sort(p.begin(), p.end());
    return p;
}

/** The block [CHORD][SDUR][BASS][SVEL] right after the piano quad at `at` (the target quad, written [DUR?]). */
auto expectBlockAfterPianoQuad(const std::vector<int32_t> &toks, int at, int32_t pianoPitch) -> void {
    ASSERT_GE(static_cast<int>(toks.size()), at + 8);
    EXPECT_EQ(toks[static_cast<size_t>(at + 1)], Engine::kDurUnknown);
    EXPECT_EQ(toks[static_cast<size_t>(at + 2)], Engine::kPiano0 + pianoPitch);
    EXPECT_EQ(toks[static_cast<size_t>(at + 4)], kCMajor3);
    EXPECT_EQ(toks[static_cast<size_t>(at + 5)], Engine::kDur0 + 80);
    EXPECT_EQ(toks[static_cast<size_t>(at + 6)], Engine::kOrch0 + kBass);
    EXPECT_EQ(toks[static_cast<size_t>(at + 7)], Engine::kVel0 + 64);
}

} // namespace

TEST(ChordTokenEngine, TokenIdsMatchWait3) {
    EXPECT_EQ(Engine::waitToken(3), 11768);
    EXPECT_EQ(Engine::waitToken(300), 11867);
    EXPECT_EQ(Engine::kToken(1, false), 11868);
    EXPECT_EQ(Engine::kToken(1, true), 11869);
    EXPECT_EQ(Engine::kToken(4, true), 11875);
    EXPECT_EQ(Engine::kOf(11873), 3);
    EXPECT_TRUE(Engine::plusOf(11873));
    EXPECT_EQ(Engine::kChordExpired, 11876);
    EXPECT_EQ(Engine::lateToken(3), 11877);
    EXPECT_EQ(Engine::lateToken(99), 11909);
    EXPECT_EQ(Engine::lateToken(150), 11909); // clamped to 99
    EXPECT_EQ(Engine::lateValue(Engine::lateToken(13)), 12);
    EXPECT_EQ(Engine::kDurUnknown, 11910);
    EXPECT_EQ(Engine::chordToken(0, 0, 1), 11911);
    EXPECT_EQ(Engine::chordToken(17, 11, 10), 14070);
    EXPECT_EQ(Engine::kChordVocab, 14071);
    const auto v = Engine::chordValue(Engine::chordToken(12, 5, 7));
    EXPECT_EQ(v.type, 12);
    EXPECT_EQ(v.root, 5);
    EXPECT_EQ(v.n, 7);
    // the type table and its pitch-class counts (k)
    EXPECT_EQ(VoicingBank::typeIndex("maj"), kMaj);
    EXPECT_EQ(VoicingBank::typeIndex("augmaj7"), 17);
    EXPECT_EQ(VoicingBank::pitchClassCount(0), 1);
    EXPECT_EQ(VoicingBank::pitchClassCount(4), 2);
    EXPECT_EQ(VoicingBank::pitchClassCount(10), 3);
    EXPECT_EQ(VoicingBank::pitchClassCount(16), 4);
}

TEST(ChordTokenEngine, VoicingBankDecodesExactNearestAndStacked) {
    const auto bank = testBank();
    std::mt19937_64 rng(1);
    // exact key: C major, bass C3 -> bass_iv 0 -> [0, 4, 7]
    auto d = bank->decode(kMaj, 0, 3, kBass, rng);
    EXPECT_EQ(d.source, VoicingBank::Decoded::Source::Bank);
    EXPECT_EQ(d.pitches, (std::vector<int32_t>{48, 52, 55}));
    // no maj|5|8: the nearest n with the same type and bass position
    d = bank->decode(kMaj, 0, 5, kBass, rng);
    EXPECT_EQ(d.source, VoicingBank::Decoded::Source::NearestN);
    EXPECT_EQ(d.pitches, (std::vector<int32_t>{48, 52, 55}));
    // no dom7 at all: the chord tones stacked upward from the bass (C7 on C3)
    d = bank->decode(11, 0, 5, kBass, rng);
    EXPECT_EQ(d.source, VoicingBank::Decoded::Source::Stacked);
    EXPECT_EQ(d.pitches, (std::vector<int32_t>{48, 52, 55, 58, 60}));
    // JSON in the bank file's format
    VoicingBank parsed;
    std::string error;
    ASSERT_TRUE(parsed.loadJson(R"({"format": "x", "bank": {"min|3|0": [[0, 3, 7], [0, 15, 19]], "unison|2|0": [[0, 12]]}})", error))
        << error;
    EXPECT_EQ(parsed.size(), 2u);
    d = parsed.decode(VoicingBank::typeIndex("unison"), 0, 2, 60, rng); // unison of C on C4: bass_iv 0
    EXPECT_EQ(d.pitches, (std::vector<int32_t>{60, 72}));
    EXPECT_FALSE(parsed.loadJson("{\"nobank\": 1}", error));
}

TEST(ChordTokenEngine, RealBankFileRootPositionMajorIsAMajorTriad) {
    const juce::File file("/Users/Joel/orchestration/overnight/results/voicing_bank_m2c.json");
    if (! file.existsAsFile())
        GTEST_SKIP() << "voicing bank not on this machine";
    VoicingBank bank;
    std::string error;
    ASSERT_TRUE(bank.loadJson(file.loadFileAsString().toStdString(), error)) << error;
    EXPECT_GT(bank.size(), 200u);
    std::mt19937_64 rng(2);
    for (int i = 0; i < 20; ++i) {
        // D major (root 2) with the bass on D3 (50): every voicing is D / F# / A only
        const auto d = bank.decode(kMaj, 2, 3, 50, rng);
        ASSERT_EQ(d.source, VoicingBank::Decoded::Source::Bank);
        std::set<int> pcs;
        for (const auto p: d.pitches)
            pcs.insert(p % 12);
        EXPECT_EQ(pcs, (std::set<int>{2, 6, 9}));
        EXPECT_EQ(d.pitches.front(), 50);
    }
}

TEST(ChordTokenEngine, FireServedFromThePrecomputedChordBlock) {
    ChordRig rig;
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 6); // the group closes at 5; [WAIT 30][K 3]
    ASSERT_TRUE(rig.engine.hasPendingWait());
    rig.run(7, 22); // 1 anticipation call + 4 batched (-> CHORD | pitch; re-tiled
                    // [pitch, vel, CHORD] -> SDUR; -> BASS; -> SVEL)
    ASSERT_TRUE(rig.engine.precomputeComplete());
    EXPECT_EQ(rig.model.batchedCalls, 4);
    EXPECT_TRUE(rig.played.empty()); // silent while the WAIT is pending

    rig.engine.noteOn(30, 60, 80);
    const auto out = rig.engine.step(30);
    EXPECT_EQ(pitchesOf(out), (std::vector<int32_t>{48, 52, 55})); // the bank voicing on BASS
    for (const auto &n: out) {
        EXPECT_EQ(n.onsetCs, 30);
        EXPECT_EQ(n.durationCs, 80); // SDUR
        EXPECT_EQ(n.velocity, 64);   // SVEL
        EXPECT_TRUE(n.precomputed);
    }
    EXPECT_EQ(rig.engine.stats().precomputeHits, 1);
    EXPECT_EQ(rig.engine.stats().chordMatched, 1);
    EXPECT_EQ(rig.engine.stats().chordBlocks, 1);
    // context: piano quad, WAIT, K, target piano quad ([DUR?]), the 4 block tokens (not the 3 notes)
    const auto toks = rig.engine.tokens();
    ASSERT_EQ(toks.size(), 4u + 2u + 4u + 4u);
    EXPECT_EQ(toks[1], Engine::kDurUnknown); // the first note was still down when the WAIT was written
    EXPECT_EQ(toks[4], Engine::waitToken(30));
    EXPECT_EQ(toks[5], Engine::kToken(3, false));
    EXPECT_EQ(toks[6], 30);
    expectBlockAfterPianoQuad(toks, 6, 60);
}

TEST(ChordTokenEngine, FireDecodedOnTheSpotWithoutPrecompute) {
    ChordRig rig(false);
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 20);
    ASSERT_TRUE(rig.engine.hasPendingWait());
    // no finished precompute = no match: CHORD inferred from the piano (E, k = 3, in C major: C major, step 1);
    // n from the model, then SDUR / BASS / SVEL, all in the fire's own step
    rig.engine.noteOn(28, 64, 80);
    const auto out = rig.engine.step(28);
    EXPECT_EQ(pitchesOf(out), (std::vector<int32_t>{48, 52, 55}));
    for (const auto &n: out) {
        EXPECT_EQ(n.onsetCs, 28);
        EXPECT_FALSE(n.precomputed);
    }
    EXPECT_EQ(rig.engine.stats().fireDecoded, 1);
    EXPECT_EQ(rig.engine.stats().chordMatched, 0);
    EXPECT_EQ(rig.engine.stats().inferSteps[1], 1);
    expectBlockAfterPianoQuad(rig.engine.tokens(), 6, 64);
}

TEST(ChordTokenEngine, AllPianoNotesOfTheOnsetComeBeforeTheBlock) {
    ChordRig rig;
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 22);
    ASSERT_TRUE(rig.engine.precomputeComplete());
    rig.engine.noteOn(30, 64, 80); // the chord's notes come in together
    rig.engine.noteOn(30, 60, 80);
    const auto out = rig.engine.step(30);
    EXPECT_EQ(pitchesOf(out), (std::vector<int32_t>{48, 52, 55})); // C E with the C major candidate: matches
    EXPECT_EQ(rig.engine.stats().chordMatched, 1);
    const auto toks = rig.engine.tokens();
    ASSERT_EQ(toks.size(), 4u + 2u + 8u + 4u);
    EXPECT_EQ(toks[6], 30);
    EXPECT_EQ(toks[7], Engine::kDurUnknown);
    EXPECT_EQ(toks[8], Engine::kPiano0 + 60);
    EXPECT_EQ(toks[10], 30);
    EXPECT_EQ(toks[11], Engine::kDurUnknown); // every piano quad of T is [DUR?]
    EXPECT_EQ(toks[12], Engine::kPiano0 + 64);
    EXPECT_EQ(toks[14], kCMajor3);
    // a release now does not rewrite them (the block follows)
    rig.engine.noteOff(30, 64, 10);
    rig.run(31, 33);
    EXPECT_EQ(rig.engine.tokens()[11], Engine::kDurUnknown);
}

TEST(ChordTokenEngine, MatchRuleTakesTheBestScoringCandidateThatFits) {
    ChordRig rig;
    rig.model.chordTok2 = Engine::chordToken(kMaj, 7, 3); // G major, a little less likely than C major
    rig.model.chordLogit2 = 19.5f;
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 22);
    ASSERT_TRUE(rig.engine.precomputeComplete());
    rig.engine.noteOn(30, 62, 80); // D F#: with C major C D E F# G (no), with G major G B D F# = Gmaj7 (yes)
    rig.engine.noteOn(30, 66, 80);
    rig.run(30, 30);
    EXPECT_EQ(rig.engine.stats().chordMatched, 1);
    EXPECT_EQ(rig.engine.tokens()[14], Engine::chordToken(kMaj, 7, 3));
    // with G instead, both fit and C major has the higher score
    ChordRig rig2;
    rig2.model.chordTok2 = Engine::chordToken(kMaj, 7, 3);
    rig2.model.chordLogit2 = 19.5f;
    rig2.engine.noteOn(0, 60, 80);
    rig2.run(0, 22);
    rig2.engine.noteOn(30, 67, 80); // G: in both
    rig2.run(30, 30);
    EXPECT_EQ(rig2.engine.tokens()[10], kCMajor3);
}

TEST(ChordTokenEngine, NoMatchInfersTheChordFromThePianoNotes) {
    ChordRig rig;
    rig.engine.noteOn(0, 70, 80); // Bb: the key around it
    rig.run(0, 22);
    ASSERT_TRUE(rig.engine.precomputeComplete());
    // C Eb F with K 3: no triad holds them, so no candidate can match; inferred F7 (step 2, a seventh accepted for k 3)
    rig.engine.noteOn(30, 60, 80);
    rig.engine.noteOn(30, 63, 80);
    rig.engine.noteOn(30, 65, 80);
    rig.run(30, 31);
    EXPECT_EQ(rig.engine.stats().chordMatched, 0);
    EXPECT_EQ(rig.engine.stats().inferSteps[2], 1);
    const auto toks = rig.engine.tokens();
    ASSERT_GE(toks.size(), 4u + 2u + 12u + 4u);
    const auto cv = Engine::chordValue(toks[18]); // after the three piano quads
    EXPECT_EQ(cv.type, 11);
    EXPECT_EQ(cv.root, 5);
    EXPECT_EQ(toks[19], Engine::kDur0 + 80); // then SDUR / BASS / SVEL from the model
    EXPECT_EQ(toks[20], Engine::kOrch0 + kBass);
    EXPECT_FALSE(rig.played.empty());
}

TEST(ChordTokenEngine, PlusOneWritesTheLeftoverQuadAtTheTargetOnset) {
    ChordRig rig;
    rig.model.plus = true;
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 30); // [WAIT 30][K 3,+1]; the branches also decode the leftover: 7 batched calls
    ASSERT_TRUE(rig.engine.precomputeComplete());
    EXPECT_EQ(rig.model.batchedCalls, 7);
    rig.engine.noteOn(31, 60, 80);
    const auto out = rig.engine.step(31);
    ASSERT_EQ(out.size(), 4u); // 3 chord notes + the leftover
    EXPECT_EQ(rig.engine.stats().precomputeHits, 1);
    const auto toks = rig.engine.tokens();
    EXPECT_EQ(toks[5], Engine::kToken(3, true));
    expectBlockAfterPianoQuad(toks, 6, 60);
    // the leftover quad: onset T (31 - base 0), its own duration / pitch / velocity
    ASSERT_EQ(toks.size(), 4u + 2u + 4u + 4u + 4u);
    EXPECT_EQ(toks[14], 31);
    EXPECT_EQ(toks[15], Engine::kDur0 + 50);
    EXPECT_EQ(toks[16], Engine::kOrch0 + kBass);
}

TEST(ChordTokenEngine, ALatePianistGetsALateTokenAndStillFires) {
    ChordRig rig;
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 37); // the window [24, 36] closes empty: [LATE 12] at 36 -> new window [42, 54]
    EXPECT_EQ(rig.engine.stats().lates, 1);
    EXPECT_EQ(rig.engine.stats().expired, 0);
    ASSERT_TRUE(rig.engine.hasPendingWait());
    EXPECT_TRUE(rig.played.empty()); // the orchestra holds
    rig.run(38, 44);                 // the precompute starts again after the LATE
    rig.engine.noteOn(45, 64, 80);
    rig.run(45, 46);
    EXPECT_EQ(rig.engine.stats().fires, 1);
    EXPECT_EQ(pitchesOf(rig.played), (std::vector<int32_t>{48, 52, 55}));
    const auto toks = rig.engine.tokens();
    ASSERT_GE(toks.size(), 7u + 8u);
    EXPECT_EQ(toks[4], Engine::waitToken(30));
    EXPECT_EQ(toks[6], Engine::lateToken(12));
    EXPECT_EQ(toks[7], 45);
    expectBlockAfterPianoQuad(toks, 7, 64);
}

TEST(ChordTokenEngine, AfterTwoLatesExpiredIsForcedAndACHORDGivesUp) {
    ChordRig rig;
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 74); // closes at 36 (LATE 12), 54 (LATE 12), 72: [EXPIRED] forced, the model's CHORD, the block
    EXPECT_EQ(rig.engine.stats().lates, 2);
    EXPECT_EQ(rig.engine.stats().giveUps, 1);
    EXPECT_EQ(rig.engine.stats().expiredChosen, 0);
    ASSERT_GE(rig.played.size(), 3u);
    std::vector<Engine::OutNote> chord(rig.played.begin(), rig.played.begin() + 3);
    EXPECT_EQ(pitchesOf(chord), (std::vector<int32_t>{48, 52, 55}));
    for (const auto &n: chord)
        EXPECT_GE(n.onsetCs, 72); // now, not in the past
    const auto toks = rig.engine.tokens();
    ASSERT_GE(toks.size(), 13u);
    EXPECT_EQ(toks[6], Engine::lateToken(12));
    EXPECT_EQ(toks[7], Engine::lateToken(12));
    EXPECT_EQ(toks[8], Engine::kChordExpired);
    EXPECT_EQ(toks[9], kCMajor3); // the block right after EXPIRED, no piano quad
    EXPECT_EQ(toks[11], Engine::kOrch0 + kBass);
}

TEST(ChordTokenEngine, TheModelMayDrawExpiredAtAWaitCloseAndThePianoNeverComes) {
    ChordRig rig;
    rig.model.expiredAfterWait = 30.0f; // never comes: [EXPIRED] over [LATE 12]
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 38); // the window [24, 36] closes empty: [EXPIRED], then the model's CHORD and the block, now
    EXPECT_EQ(rig.engine.stats().lates, 0);
    EXPECT_EQ(rig.engine.stats().expiredChosen, 1);
    EXPECT_EQ(rig.engine.stats().giveUps, 1);
    ASSERT_GE(rig.played.size(), 3u);
    std::vector<Engine::OutNote> chord(rig.played.begin(), rig.played.begin() + 3);
    EXPECT_EQ(pitchesOf(chord), (std::vector<int32_t>{48, 52, 55}));
    for (const auto &n: chord)
        EXPECT_GE(n.onsetCs, 36);
    const auto toks = rig.engine.tokens();
    ASSERT_GE(toks.size(), 11u);
    EXPECT_EQ(toks[4], Engine::waitToken(30));
    EXPECT_EQ(toks[6], Engine::kChordExpired);
    EXPECT_EQ(toks[7], kCMajor3);
    EXPECT_EQ(toks[8], Engine::kDur0 + 80);
    EXPECT_EQ(toks[9], Engine::kOrch0 + kBass);
}

TEST(ChordTokenEngine, TheModelMayDrawExpiredAtALateClose) {
    ChordRig rig;
    rig.model.expiredAfterLate = 30.0f; // one LATE, then it gives up before the second
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 56); // closes at 36 (LATE 12) and 54: [EXPIRED], CHORD, the block
    EXPECT_EQ(rig.engine.stats().lates, 1);
    EXPECT_EQ(rig.engine.stats().expiredChosen, 1);
    EXPECT_EQ(rig.engine.stats().giveUps, 1);
    EXPECT_EQ(rig.played.size(), 3u);
    const auto toks = rig.engine.tokens();
    ASSERT_GE(toks.size(), 9u);
    EXPECT_EQ(toks[6], Engine::lateToken(12));
    EXPECT_EQ(toks[7], Engine::kChordExpired);
    EXPECT_EQ(toks[8], kCMajor3);
}

TEST(ChordTokenEngine, NoLatesAllowedForcesExpired) {
    ChordRig rig;
    Engine::Settings s = rig.engine.settings();
    s.maxLates = 0;
    rig.engine.setSettings(s);
    rig.engine.reset(0);
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 38);
    EXPECT_EQ(rig.engine.stats().lates, 0);
    EXPECT_EQ(rig.engine.stats().expiredChosen, 0); // forced, not drawn
    EXPECT_EQ(rig.engine.stats().giveUps, 1);
    EXPECT_EQ(rig.find(Engine::kChordExpired), 6);
}

TEST(ChordTokenEngine, APianoOnsetAlreadyPastTheCloseSetsTheLateValue) {
    ChordRig rig(true, 30); // expiry grace 30 cs: the close at 36 is decided only by 66 or a later onset
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 49);
    ASSERT_TRUE(rig.engine.hasPendingWait());
    EXPECT_EQ(rig.engine.stats().lates, 0);
    rig.engine.noteOn(50, 64, 80); // 14 cs after the close: [LATE 15] (d = arrival - close), and it fires
    rig.run(50, 51);
    EXPECT_EQ(rig.engine.stats().lates, 1);
    EXPECT_EQ(rig.engine.stats().lateForced, 1);
    EXPECT_EQ(rig.engine.stats().fires, 1);
    EXPECT_EQ(rig.find(Engine::lateToken(15)), 6);
    expectBlockAfterPianoQuad(rig.engine.tokens(), 7, 64);
}

TEST(ChordTokenEngine, ExpiredThenAWaitReWaitsWithTheSameK) {
    ChordRig rig(false);
    rig.model.waitD = 300;
    rig.model.k = 2;
    rig.model.expiredAfterWait = 30.0f;
    rig.model.expiredThenWait = true;
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 310); // [WAIT 300][K 2] closes at 306: [EXPIRED], then the model's WAIT with the same K
    EXPECT_EQ(rig.engine.stats().chains, 1);
    EXPECT_EQ(rig.engine.stats().lates, 0);
    EXPECT_EQ(rig.count("orch"), 0);
    const auto toks = rig.engine.tokens();
    ASSERT_EQ(toks.size(), 4u + 5u);
    EXPECT_EQ(toks[4], Engine::waitToken(300));
    EXPECT_EQ(toks[5], Engine::kToken(2, false));
    EXPECT_EQ(toks[6], Engine::kChordExpired);
    EXPECT_EQ(toks[7], Engine::waitToken(300));
    EXPECT_EQ(toks[8], Engine::kToken(2, false));
}

TEST(ChordTokenEngine, WaitsUnderSixCsAreMaskedAndThePlusOneUnderTwelve) {
    {
        ChordRig rig;
        rig.model.waitD = 3; // the model's favourite is under the floor: an orchestra onset instead
        rig.engine.noteOn(0, 60, 80);
        rig.run(0, 8);
        EXPECT_EQ(rig.engine.stats().waitsDrawn, 0);
        EXPECT_GE(rig.count("orch"), 1);
    }
    {
        ChordRig rig;
        rig.model.waitD = 9;
        rig.model.plus = true; // +1 preferred, but a leftover needs d >= 12 cs
        rig.engine.noteOn(0, 60, 80);
        rig.run(0, 6);
        EXPECT_EQ(rig.find(Engine::waitToken(9)), 4);
        EXPECT_EQ(rig.find(Engine::kToken(3, false)), 5);
    }
}

TEST(ChordTokenEngine, ReleaseReplacesDurUnknownOnlyBeforeAModelWrittenUnit) {
    ChordRig rig;
    rig.engine.noteOn(0, 60, 80);
    rig.engine.absorb(5); // written at the group's close, key down: [DUR?]
    auto toks = rig.engine.tokens();
    ASSERT_EQ(toks.size(), 4u);
    EXPECT_EQ(toks[1], Engine::kDurUnknown);
    rig.engine.noteOff(0, 60, 30);
    rig.engine.absorb(30); // nothing model-written after it yet: the real duration
    toks = rig.engine.tokens();
    EXPECT_EQ(toks[1], Engine::kDur0 + 30);

    rig.engine.noteOn(100, 62, 80);
    rig.engine.absorb(105);
    rig.engine.mirrorNote({106, 50, 67, 64}); // an orchestra quad after it (model 1's note)
    rig.engine.absorb(107);
    rig.engine.noteOff(100, 62, 40);
    rig.engine.absorb(150);
    toks = rig.engine.tokens();
    ASSERT_EQ(toks.size(), 12u);
    EXPECT_EQ(toks[5], Engine::kDurUnknown); // stays [DUR?]
    EXPECT_EQ(toks[10], Engine::kOrch0 + 67);
}

TEST(ChordTokenEngine, ModelOneSeesAFiredChordAsPlainQuads) {
    // Hybrid: model 2's output notes (the decoded chord) are mirrored into model 1's own context, as for v1.1
    ChordRig m2;
    m2.engine.noteOn(0, 60, 80);
    m2.run(0, 22);
    m2.engine.noteOn(30, 60, 80);
    const auto chord = m2.engine.step(30);
    ASSERT_EQ(chord.size(), 3u);

    ChordScript m1Model;
    Engine m1;
    m1.setBackend(&m1Model);
    Engine::Settings s;
    s.continuation = true;
    m1.setSettings(s);
    m1.reset(0);
    m1.noteOn(0, 60, 80);
    m1.noteOn(30, 60, 80);
    m1.absorb(35);
    for (const auto &n: chord)
        m1.mirrorNote(n);
    m1.absorb(36);
    const auto toks = m1.tokens();
    ASSERT_EQ(toks.size(), 4u * 5u); // two piano quads, three orchestra quads
    std::vector<int32_t> pitches;
    for (size_t i = 0; i + 3 < toks.size(); i += 4) {
        EXPECT_LT(toks[i + 1], Engine::kPiano0); // real durations (model 1's estimate), never [DUR?]
        if (toks[i + 2] >= Engine::kOrch0) {
            EXPECT_EQ(toks[i], 30);
            EXPECT_EQ(toks[i + 1], Engine::kDur0 + 80);
            pitches.push_back(toks[i + 2] - Engine::kOrch0);
        }
    }
    EXPECT_EQ(pitches, (std::vector<int32_t>{48, 52, 55}));
    for (const auto t: toks)
        EXPECT_LT(t, Engine::kWait0); // nothing of model 2's vocabulary
}

TEST(ChordTokenEngine, HybridRunsTheChordModelAndSwitchesToIt) {
    ModelConfig config;
    HybridInformativeWaitReductionTransformer hy(config);
    auto silent = std::make_unique<ChordScript>();
    silent->waitD = 0;
    silent->onsetRel = 40;
    hy.setChordModel(true);
    hy.setVoicingBank(testBank());
    hy.initWithBackends(std::move(silent), std::make_unique<ChordScript>());
    EXPECT_TRUE(hy.isChordModel());
    EXPECT_STREQ(hy.label(), "Hybrid v1.2");
    int32_t now = 0;
    int64_t wall = 0;
    int64_t silentFor = INT64_MAX;
    hy.nowCsProvider = [&] { return now; };
    hy.wallMsProvider = [&] { return wall; };
    hy.pianoSilentForMs = [&] { return silentFor; };
    hy.stepOnce();
    for (int i = 0; i < 15; ++i) {
        Token t{now, static_cast<int32_t>(DenseVocab::DurOffset) + 20, static_cast<int32_t>(DenseVocab::NoteOffset) + 60};
        t.velocity = 80;
        hy.inputTokenQueue.push(t);
        silentFor = 0;
        for (int k = 0; k < 10; ++k) {
            hy.stepOnce();
            now += 1;
            wall += 10;
        }
    }
    EXPECT_FALSE(hy.usesSilentModel());
}
