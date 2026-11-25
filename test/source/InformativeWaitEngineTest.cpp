#include <gtest/gtest.h>

#include <JuceHeader.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include <VirtualOrch/OpLog.h>
#include <VirtualOrch/PolyphonyCap.h>
#include <VirtualOrch/reduction/DenseTypes.h>
#include <VirtualOrch/reduction/HybridInformativeWaitReductionTransformer.h>
#include <VirtualOrch/reduction/InformativeWaitEngine.h>
#include <VirtualOrch/reduction/StreamingKvCache.h>
#include <VirtualOrch/TemporalDensity.h>
#include <VirtualOrch/reduction/TdController.h>

/**
 * Informative WAIT (overnight model 2): the WAIT / COUNT / EXPIRED state machine in the three modes,
 * the batched anticipation precompute, and (when the export and a Python sample are on this
 * machine) the real ONNX model: batched == batch-1 logits and C++ == Python probabilities.
 */
namespace {

using Engine = InformativeWaitEngine;
using namespace StreamingKv;

/** Scripted model: the logits depend on the last token only. */
class ScriptBackend : public Backend {
public:
    ScriptBackend() {
        d.layers = 1;
        d.heads = 1;
        d.headDim = 1;
        d.maxRows = 512;
        d.sinkRows = 4;
    }
    [[nodiscard]] auto dims() const -> Dims override { return d; }

    auto logitsAfter(int64_t last) const -> std::vector<float> {
        std::vector<float> l(Engine::kVocab, 0.0f);
        const auto set = [&](int32_t id, float v) { l[static_cast<size_t>(id)] = v; };
        const bool afterVelocity = last >= Engine::kVel0 && last < Engine::kWait0;
        if (afterVelocity || last == Engine::kExpired || (last >= Engine::kCount0 && last < Engine::kExpired)) {
            // onset step: a WAIT of `waitD` cs (or an onset at `onsetRel` when waitD == 0); with onsetSlope the
            // nearer onsets are likelier (logit -slope x onset), as a real model's
            if (*onsetSlope > 0.0f)
                for (int32_t o = 0; o <= Engine::kOnsetMax; ++o)
                    set(o, -*onsetSlope * static_cast<float>(o));
            if (*waitD > 0)
                set(Engine::waitToken(*waitD), 20.0f);
            set(*onsetRel, 15.0f);
        } else if (last >= Engine::kWait0 && last < Engine::kCount0) {
            set(Engine::countToken(*count), 20.0f);
        } else if (last <= Engine::kOnsetMax) {
            set(Engine::kDur0 + 50, 20.0f); // duration 50 cs
        } else if (last >= Engine::kDur0 && last < Engine::kPiano0) {
            const int pianoPitches[] = {60, 62, 64, 65, 67};
            for (int i = 0; i < 5; ++i)
                set(Engine::kPiano0 + pianoPitches[i], 10.0f - static_cast<float>(i));
            set(Engine::kOrch0 + 60, 9.0f);
            set(Engine::kOrch0 + 64, 8.8f);
            set(Engine::kOrch0 + 67, 8.6f);
        } else if (last >= Engine::kPiano0 && last < Engine::kVel0) {
            set(Engine::kVel0 + 64, 20.0f);
        }
        return l;
    }

    auto run(const State &past, const std::vector<int64_t> &ids, State &present, std::vector<float> &last)
        -> bool override {
        present = past;
        for (const auto id: ids) {
            present.key.push_back(static_cast<float>(id));
            present.value.push_back(static_cast<float>(id));
            ++present.rows;
        }
        ++calls;
        last = logitsAfter(ids.back());
        return true;
    }

    auto runBatched(const BatchState &past, const std::vector<int64_t> &ids, BatchState &present,
                    std::vector<float> &logits) -> bool override {
        if (! batching)
            return false;
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
    std::shared_ptr<int32_t> waitD = std::make_shared<int32_t>(30);
    std::shared_ptr<int32_t> count = std::make_shared<int32_t>(2);
    std::shared_ptr<int32_t> onsetRel = std::make_shared<int32_t>(10);
    std::shared_ptr<float> onsetSlope = std::make_shared<float>(0.0f);
    bool batching = true;
    int calls = 0;
    int batchedCalls = 0;
};

struct Rig {
    ScriptBackend model;
    Engine engine;
    std::vector<Engine::Event> events;
    std::vector<Engine::OutNote> played;

    explicit Rig(Engine::Mode mode) {
        engine.setBackend(&model);
        Engine::Settings s;
        s.mode = mode;
        s.seed = 7;
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
};

TEST(InformativeWaitEngine, TokenIds) {
    EXPECT_EQ(Engine::waitToken(3), 11768);
    EXPECT_EQ(Engine::waitToken(300), 11867);
    EXPECT_EQ(Engine::waitToken(31), Engine::waitToken(30)); // nearest multiple of 3
    EXPECT_EQ(Engine::waitValue(Engine::waitToken(30)), 30);
    EXPECT_EQ(Engine::countToken(1), 11868);
    EXPECT_EQ(Engine::countToken(10), 11877);
    EXPECT_EQ(Engine::kExpired, 11878);
    EXPECT_EQ(Engine::q5(52), 50);
    EXPECT_EQ(Engine::q8(70), 72);
}

TEST(InformativeWaitEngine, FireServedFromPrecomputedBranch) {
    Rig rig(Engine::Mode::Unconstrained);
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 6); // the group closes at 5 cs; the model draws [WAIT 30][COUNT 2]
    ASSERT_TRUE(rig.engine.hasPendingWait());
    EXPECT_EQ(rig.engine.stats().waitsDrawn, 1);
    rig.run(7, 22); // 1 anticipation call + 3k = 6 batched calls
    ASSERT_TRUE(rig.engine.precomputeComplete());
    const auto top = rig.engine.anticipatedPitches();
    ASSERT_EQ(top.size(), 5u);
    EXPECT_EQ(top.front(), 60);
    EXPECT_EQ(rig.model.batchedCalls, 6);
    EXPECT_TRUE(rig.played.empty()); // the orchestra waits

    rig.engine.noteOn(30, 60, 80); // in the window [24, 36]
    const auto out = rig.engine.step(30);
    ASSERT_EQ(out.size(), 2u); // exactly COUNT notes, at once, on the piano onset
    for (const auto &n: out) {
        EXPECT_EQ(n.onsetCs, 30);
        EXPECT_TRUE(n.precomputed);
    }
    EXPECT_LT(out[0].pitch, out[1].pitch); // ascending chord
    EXPECT_EQ(rig.engine.stats().precomputeHits, 1);
    EXPECT_EQ(rig.engine.stats().antTop1, 1);
    EXPECT_FALSE(rig.engine.hasPendingWait());
    // tokens: piano quad, WAIT, COUNT, first piano note, 2 orchestra quads (onset 30 = model time 30)
    const auto toks = rig.engine.tokens();
    ASSERT_EQ(toks.size(), 4u + 2u + 4u + 8u);
    EXPECT_EQ(toks[4], Engine::waitToken(30));
    EXPECT_EQ(toks[5], Engine::countToken(2));
    EXPECT_EQ(toks[6], 30);
    EXPECT_EQ(toks[8], Engine::kPiano0 + 60);
    EXPECT_EQ(toks[10], 30);
    EXPECT_GE(toks[12], Engine::kOrch0);
}

// The duration and pitch biases reach the precomputed branches, from the committed orchestra notes only.
// The scripted model wants duration 50 and pitches 60 / 64 / 67; a strong τ makes the bias decisive.
TEST(InformativeWaitEngine, PrecomputedBranchesAreBiasedByCommittedNotesOnly) {
    Rig rig(Engine::Mode::Unconstrained);
    auto s = rig.engine.settings();
    s.durationBias = true;
    s.pitchBias = true;
    s.durationTau = 20.0f;
    s.durationTauPrime = 0.0f;
    s.pitchTau = 20.0f;
    s.pitchTauPrime = 0.0f;
    rig.engine.setSettings(s);
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 22); // [WAIT 30][COUNT 2], precomputed with no orchestra note committed yet
    ASSERT_TRUE(rig.engine.precomputeComplete());
    rig.engine.noteOn(30, 60, 80);
    const auto first = rig.engine.step(30);
    ASSERT_EQ(first.size(), 2u);
    for (const auto &n: first) {
        EXPECT_TRUE(n.precomputed);
        // nothing committed: no bias, so the branch's 2nd note keeps duration 50 although its 1st note
        // (uncommitted while it was drawn) has it
        EXPECT_EQ(n.durationCs, 50);
    }

    // the fire committed both notes (duration 50): the next precompute's branches avoid their durations
    // and pitches (the model's pitches 60 / 64 / 67 are close, so which two were played is sampled)
    rig.run(31, 59);
    ASSERT_TRUE(rig.engine.hasPendingWait());
    ASSERT_TRUE(rig.engine.precomputeComplete());
    rig.engine.noteOn(60, 60, 80);
    const auto second = rig.engine.step(60);
    ASSERT_EQ(second.size(), 2u);
    for (const auto &n: second) {
        EXPECT_TRUE(n.precomputed);
        EXPECT_NE(n.durationCs, 50);
        EXPECT_NE(n.pitch, first[0].pitch);
        EXPECT_NE(n.pitch, first[1].pitch);
    }

    // the same run without the biases plays the model's choices
    Rig plain(Engine::Mode::Unconstrained);
    plain.engine.noteOn(0, 60, 80);
    plain.run(0, 29);
    plain.engine.noteOn(30, 60, 80);
    (void) plain.engine.step(30);
    plain.run(31, 59);
    plain.engine.noteOn(60, 60, 80);
    const auto unbiased = plain.engine.step(60);
    ASSERT_EQ(unbiased.size(), 2u);
    for (const auto &n: unbiased)
        EXPECT_EQ(n.durationCs, 50);
}

TEST(InformativeWaitEngine, AFireIsTimedFromTheKeyLandingToTheLastOrchestraNote) {
    Rig rig(Engine::Mode::Unconstrained);
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 22); // [WAIT 30][COUNT 2], precomputed
    ASSERT_TRUE(rig.engine.precomputeComplete());
    // the key landed 3 ms before the engine applies it
    const auto landed = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now().time_since_epoch()).count() - 3000;
    rig.engine.noteOn(30, 60, 80, landed);
    rig.engine.step(30);
    const auto placed = std::find_if(rig.events.begin(), rig.events.end(), [](const auto &e) { return e.what == "placed"; });
    ASSERT_NE(placed, rig.events.end());
    EXPECT_EQ(placed->a, 2);         // both COUNT notes, at the fire
    EXPECT_GE(placed->b, 3000);      // from the key, not from the fire
    EXPECT_LT(placed->b, 3000000);
    EXPECT_EQ(placed->text, "precomputed|key");
}

TEST(InformativeWaitEngine, ADecodedFireIsPlacedAfterItsLastNote) {
    Rig rig(Engine::Mode::Unconstrained);
    auto s = rig.engine.settings();
    s.outsideTopFiveAsRankOne = false;
    rig.engine.setSettings(s);
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 30);
    rig.engine.noteOn(31, 50, 80); // not anticipated: decoded one note per step; landing unknown
    rig.run(31, 33);
    const auto placed = std::find_if(rig.events.begin(), rig.events.end(), [](const auto &e) { return e.what == "placed"; });
    ASSERT_NE(placed, rig.events.end());
    EXPECT_EQ(placed->a, 2);
    EXPECT_EQ(placed->text, "decoded|fire");
}

TEST(InformativeWaitEngine, APitchOutsideTheTopFivePlaysTheRankOneBranch) {
    Rig rig(Engine::Mode::Unconstrained);
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 30); // [WAIT 30][COUNT 2], precomputed for the top 5 (60 first)
    ASSERT_TRUE(rig.engine.precomputeComplete());
    rig.played.clear();
    rig.engine.noteOn(31, 50, 80); // 50 is not in the top 5
    const auto out = rig.engine.step(31);
    ASSERT_EQ(out.size(), 2u) << "all COUNT notes at once, from the rank-1 branch";
    for (const auto &n: out) {
        EXPECT_TRUE(n.precomputed);
        EXPECT_EQ(n.onsetCs, 31);
    }
    EXPECT_EQ(rig.engine.stats().fireDecoded, 0);
    EXPECT_EQ(rig.engine.stats().precomputeHits, 1);
    EXPECT_EQ(rig.engine.stats().antTop5, 0) << "still scored as outside the top 5";
}

TEST(InformativeWaitEngine, MissDecodesOnTheSpot) {
    Rig rig(Engine::Mode::Unconstrained);
    auto s = rig.engine.settings();
    s.outsideTopFiveAsRankOne = false; // decode for the pitch actually played
    rig.engine.setSettings(s);
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 30);
    rig.engine.noteOn(31, 50, 80); // not anticipated
    rig.played.clear();
    rig.run(31, 33); // decoded one note per step after the fire
    const auto &out = rig.played;
    ASSERT_EQ(out.size(), 2u);
    EXPECT_FALSE(out[0].precomputed);
    EXPECT_EQ(out[0].onsetCs, 31);
    EXPECT_LT(out[0].pitch, out[1].pitch);
    EXPECT_EQ(rig.engine.stats().fireDecoded, 1);
    EXPECT_EQ(rig.engine.stats().antTop5, 0);
}

TEST(InformativeWaitEngine, NoPianoWritesExpired) {
    Rig rig(Engine::Mode::Unconstrained);
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 36);
    EXPECT_TRUE(rig.engine.hasPendingWait()); // expires after 30 + 6
    rig.run(37, 37);
    EXPECT_EQ(rig.engine.stats().expired, 1);
    const auto toks = rig.engine.tokens();
    EXPECT_NE(std::find(toks.begin(), toks.end(), Engine::kExpired), toks.end());
    // the model draws again after the expiry (another WAIT, measured from the expiry at 36)
    rig.run(38, 38);
    EXPECT_EQ(rig.engine.stats().waitsDrawn, 2);
}

TEST(InformativeWaitEngine, AWiderWindowExpiresLater) {
    Rig rig(Engine::Mode::Unconstrained);
    auto s = rig.engine.settings();
    s.fireWindowCs = 10;
    rig.engine.setSettings(s);
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 40);
    EXPECT_TRUE(rig.engine.hasPendingWait()); // expires after 30 + 10
    rig.run(41, 41);
    EXPECT_EQ(rig.engine.stats().expired, 1);
}

TEST(InformativeWaitEngine, AKeyThatLandedInTheWindowButArrivesLateStillFires) {
    Rig rig(Engine::Mode::Unconstrained);
    auto s = rig.engine.settings();
    s.expiryGraceCs = 20;
    rig.engine.setSettings(s);
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 45); // the window [24, 36] closed at 36: within the grace, the WAIT is still pending
    EXPECT_TRUE(rig.engine.hasPendingWait());
    EXPECT_EQ(rig.engine.stats().expired, 0);
    rig.engine.noteOn(33, 62, 80); // the key landed at 33 (in the window), reported at 46
    rig.run(46, 50);
    EXPECT_EQ(rig.engine.stats().fires, 1);
    EXPECT_EQ(rig.engine.stats().expired, 0);
}

TEST(InformativeWaitEngine, AfterTheGraceTheWaitExpiresAtTheWindowsEnd) {
    Rig rig(Engine::Mode::Unconstrained);
    auto s = rig.engine.settings();
    s.expiryGraceCs = 20;
    rig.engine.setSettings(s);
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 57);
    EXPECT_EQ(rig.engine.stats().expired, 1);
    const auto ex = std::find_if(rig.events.begin(), rig.events.end(), [](const auto &e) { return e.what == "expired"; });
    ASSERT_NE(ex, rig.events.end());
    EXPECT_EQ(ex->atCs, 36); // written at the window's end, not when it was decided
}

TEST(InformativeWaitEngine, NoWaitAfterExpiryUntilTheNextPianoNote) {
    Rig rig(Engine::Mode::Unconstrained);
    auto s = rig.engine.settings();
    s.noWaitAfterExpiry = true;
    rig.engine.setSettings(s);
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 41);
    EXPECT_EQ(rig.engine.stats().expired, 1);
    rig.run(42, 80); // the model would WAIT again: no WAIT is legal until a piano note
    EXPECT_EQ(rig.engine.stats().waitsDrawn, 1);
    EXPECT_FALSE(rig.engine.hasPendingWait());
    rig.engine.noteOn(81, 62, 80);
    rig.run(81, 90);
    EXPECT_EQ(rig.engine.stats().waitsDrawn, 2);
}

TEST(InformativeWaitEngine, EarlyPianoIsInputAndTheWaitStays) {
    Rig rig(Engine::Mode::Unconstrained);
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 9);
    rig.engine.noteOn(10, 62, 80); // before the window [24, 36]
    rig.run(10, 20);
    EXPECT_EQ(rig.engine.stats().earlyPiano, 1);
    EXPECT_TRUE(rig.engine.hasPendingWait());
    EXPECT_TRUE(rig.played.empty());
    rig.run(21, 29);
    rig.engine.noteOn(30, 62, 80);
    const auto out = rig.engine.step(30);
    EXPECT_EQ(out.size(), 2u);
    EXPECT_EQ(rig.engine.stats().fires, 1);
}

TEST(InformativeWaitEngine, ChordNotesAfterTheStack) {
    Rig rig(Engine::Mode::Unconstrained);
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 29);
    rig.engine.noteOn(30, 64, 80);
    rig.engine.step(30);
    rig.engine.noteOn(32, 55, 80); // same group (within 5 cs): after the orchestra stack
    rig.run(31, 36);
    const auto toks = rig.engine.tokens();
    // ... first piano note (64), 2 orchestra notes, then the rest of the chord (55) at onset 30
    ASSERT_GE(toks.size(), 4u + 2u + 4u + 8u + 4u);
    EXPECT_EQ(toks[8], Engine::kPiano0 + 64);
    EXPECT_EQ(toks[18], 30);
    EXPECT_EQ(toks[20], Engine::kPiano0 + 55);
}

TEST(InformativeWaitEngine, HeldKeyGetsAnEstimatedDuration) {
    Rig rig(Engine::Mode::Unconstrained);
    *rig.model.waitD = 0; // no WAITs (keep the stream simple)
    *rig.model.onsetRel = 9000;
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 6);
    EXPECT_EQ(rig.engine.tokens()[1], Engine::kDur0 + 30); // nothing released yet: 30 cs
    rig.engine.noteOff(0, 60, 120);
    rig.run(7, 7);
    EXPECT_EQ(rig.engine.tokens()[1], Engine::kDur0 + 30); // not rewritten (the cache never re-feeds)
    rig.engine.noteOn(100, 62, 80);
    rig.run(100, 106);
    const auto toks = rig.engine.tokens();
    const auto at = std::find(toks.begin(), toks.end(), Engine::kPiano0 + 62);
    ASSERT_NE(at, toks.end());
    EXPECT_EQ(*(at - 1), Engine::kDur0 + 120); // the pianist's released notes: median 120
}

TEST(InformativeWaitEngine, RewriteModeRewritesTheHeldDuration) {
    Rig rig(Engine::Mode::Unconstrained);
    auto s = rig.engine.settings();
    s.rewriteHeldDurations = true;
    rig.engine.setSettings(s);
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 6);
    EXPECT_EQ(rig.engine.tokens()[1], Engine::kDur0 + 30); // held: the estimate (30 cs before any release)
    rig.engine.noteOff(0, 60, 120);
    rig.run(7, 7);
    EXPECT_EQ(rig.engine.tokens()[1], Engine::kDur0 + 120);
}

TEST(InformativeWaitEngine, PianoCutsAFutureOrchestraNote) {
    Rig rig(Engine::Mode::Unconstrained);
    *rig.model.waitD = 0; // the model plays instead of waiting
    *rig.model.onsetRel = 40;
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 5);
    ASSERT_EQ(rig.played.size(), 1u);
    EXPECT_EQ(rig.played[0].onsetCs, 40);
    rig.engine.noteOn(20, 62, 80); // the piano at 20 <= 40 - 5: the note at 40 goes
    rig.engine.step(20);
    EXPECT_EQ(rig.engine.takeCancelFrom(), 40);
    EXPECT_EQ(rig.engine.stats().cutNotes, 1);
}

TEST(InformativeWaitEngine, LineLegatoRemovesTheProvisionalWait) {
    Rig rig(Engine::Mode::LineLegato);
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 6); // free draw (no WAIT in A): a note at 10, 50 cs, provisional [WAIT 51][COUNT 1]
    ASSERT_EQ(rig.played.size(), 1u);
    EXPECT_EQ(rig.played[0].onsetCs, 10);
    EXPECT_GE(rig.played[0].pitch, 55);
    EXPECT_LE(rig.played[0].pitch, 84);
    EXPECT_TRUE(rig.engine.hasPendingWait());
    rig.run(7, 62); // no piano by the note's end (61): removed, the next note starts at 61
    ASSERT_EQ(rig.played.size(), 2u);
    EXPECT_EQ(rig.played[1].onsetCs, 61);
    EXPECT_EQ(rig.engine.stats().removed, 1);
    const auto toks = rig.engine.tokens();
    // piano, note 1, note 2, the new provisional WAIT/COUNT: the removed pair is gone
    EXPECT_EQ(toks.size(), 4u + 4u + 4u + 2u);
    EXPECT_EQ(toks[8], 61);
}

TEST(InformativeWaitEngine, LineFireStacksOneNote) {
    Rig rig(Engine::Mode::LineLegato);
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 40);
    rig.engine.noteOn(58, 62, 80); // window [55, 61]
    const auto out = rig.engine.step(58);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].onsetCs, 58);
    EXPECT_EQ(rig.engine.stats().fires, 1);
    rig.run(59, 64); // the chord closes: the fired note gets its own provisional WAIT
    EXPECT_TRUE(rig.engine.hasPendingWait());
    const auto toks = rig.engine.tokens();
    // the fired WAIT stays in the context (piano, note, WAIT, COUNT, piano, note, WAIT, COUNT)
    EXPECT_EQ(std::count(toks.begin(), toks.end(), Engine::countToken(1)), 2);
}

TEST(InformativeWaitEngine, LineRestsDrawsItsOwnWaitWithCountOne) {
    Rig rig(Engine::Mode::LineRests);
    *rig.model.count = 4; // ignored: COUNT is forced to 1 in line B
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 6);
    EXPECT_EQ(rig.engine.stats().waitsDrawn, 1);
    const auto toks = rig.engine.tokens();
    EXPECT_EQ(toks.back(), Engine::countToken(1));
    rig.run(7, 29);
    rig.engine.noteOn(30, 60, 80);
    const auto out = rig.engine.step(30);
    EXPECT_EQ(out.size(), 1u);
}

TEST(InformativeWaitEngine, NoBatchingStillAnticipates) {
    Rig rig(Engine::Mode::Unconstrained);
    rig.model.batching = false;
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 29);
    EXPECT_FALSE(rig.engine.precomputeComplete());
    rig.engine.noteOn(30, 60, 80);
    rig.played.clear();
    rig.run(30, 32);
    const auto &out = rig.played;
    EXPECT_EQ(out.size(), 2u);
    EXPECT_FALSE(out[0].precomputed);
    EXPECT_EQ(rig.engine.stats().antTop1, 1); // top-5 still scored
}

TEST(InformativeWaitEngine, ContextWindowRebases) {
    Rig rig(Engine::Mode::Unconstrained);
    Engine::Settings s = rig.engine.settings();
    s.maxTokens = 40;
    s.keepTokens = 20;
    rig.engine.setSettings(s);
    for (int32_t t = 0; t < 2000; t += 50) {
        rig.engine.noteOn(t, 60, 80);
        rig.run(t, t + 49);
    }
    EXPECT_GT(rig.engine.stats().contextRebuilds, 0);
    EXPECT_LE(rig.engine.tokens().size(), 60u);
    EXPECT_EQ(rig.engine.tokens().front(), 0); // the window starts at onset 0, as training lines
}

TEST(InformativeWaitEngine, ContinuationModeNeverWaitsAndMayShareThePianoOnset) {
    Rig rig(Engine::Mode::Unconstrained);
    auto s = rig.engine.settings();
    s.continuation = true;
    rig.engine.setSettings(s);
    *rig.model.waitD = 30; // a WAIT is the model's favourite, but model 1 has none
    *rig.model.onsetRel = 0; // and it likes to share the piano's onset
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 8);
    EXPECT_EQ(rig.engine.stats().waitsDrawn, 0);
    ASSERT_FALSE(rig.played.empty());
    EXPECT_GE(rig.played[0].onsetCs, 5); // never in the past (the group closed at 5 cs)
    const auto toks = rig.engine.tokens();
    for (const auto t: toks)
        EXPECT_LT(t, Engine::kWait0);
}

TEST(HybridInformativeWait, SwitchesToTheWaitModelAfterOneSecondOfPlayingAndBackOnSilence) {
    ModelConfig config;
    HybridInformativeWaitReductionTransformer hy(config);
    auto silent = std::make_unique<ScriptBackend>();
    *silent->waitD = 0;
    *silent->onsetRel = 40;
    auto wait = std::make_unique<ScriptBackend>();
    hy.initWithBackends(std::move(silent), std::move(wait));
    int32_t now = 0;
    int64_t wall = 0;
    int64_t silentFor = INT64_MAX;
    hy.nowCsProvider = [&] { return now; };
    hy.wallMsProvider = [&] { return wall; };
    hy.pianoSilentForMs = [&] { return silentFor; };
    hy.stepOnce();
    EXPECT_TRUE(hy.usesSilentModel());
    // the pianist plays for 1.2 s: model 2 takes over
    for (int i = 0; i < 12; ++i) {
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
    EXPECT_EQ(hy.reason(), HybridInformativeWaitReductionTransformer::Reason::Playing);
    // silent for 0.6 s: back to model 1
    silentFor = 600;
    hy.stepOnce();
    EXPECT_TRUE(hy.usesSilentModel());
    EXPECT_EQ(hy.reason(), HybridInformativeWaitReductionTransformer::Reason::Stopped);
}

TEST(InformativeWaitEngine, AKeyReleasedBeforeItsNoteIsAppliedKeepsItsDuration) {
    // Input that waited (Hybrid v1.1 paused under the pedal, Nov 23 2026): the key went down and up
    // before the engine applied either. The release must reach the note, not be dropped and leave
    // it "held" for as long as it waited (995 cs).
    Rig rig(Engine::Mode::Unconstrained);
    auto s = rig.engine.settings();
    s.continuation = true;
    rig.engine.setSettings(s);
    rig.engine.noteOn(10, 60, 80);
    rig.engine.noteOff(10, 60, 20);
    rig.engine.absorb(1600); // applied 16 s late
    const auto toks = rig.engine.tokens();
    ASSERT_GE(toks.size(), 4u);
    EXPECT_EQ(toks[1], Engine::kDur0 + 20);
    EXPECT_EQ(toks[2], Engine::kPiano0 + 60);
}

TEST(InformativeWaitEngine, OrchestraPitchesStayInTheRegisterRange) {
    Rig rig(Engine::Mode::Unconstrained);
    auto s = rig.engine.settings();
    s.continuation = true;
    s.pitchLow = 70; // the model prefers 60 / 64 / 67: none of them is allowed
    s.pitchHigh = 80;
    rig.engine.setSettings(s);
    *rig.model.waitD = 0;
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 400);
    ASSERT_FALSE(rig.played.empty());
    for (const auto &n: rig.played) {
        EXPECT_GE(n.pitch, 70);
        EXPECT_LE(n.pitch, 80);
    }
}

TEST(InformativeWaitEngine, TheLegacyLogitAdjustmentsLeaveTheWaitDecisionAlone) {
    // model 2 with every adjustment on still draws its WAITs (only onsets, durations and pitches change)
    Rig rig(Engine::Mode::Unconstrained);
    auto s = rig.engine.settings();
    s.pitchBias = s.durationBias = s.onsetGapBias = s.shortGap = s.logitTd = true;
    s.tdLevel = 8;
    rig.engine.setSettings(s);
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 6);
    EXPECT_TRUE(rig.engine.hasPendingWait());
    EXPECT_EQ(rig.engine.stats().waitsDrawn, 1);
}

TEST(InformativeWaitEngine, ModelOneDrawsWithTheLegacyBiasesOn) {
    Rig rig(Engine::Mode::Unconstrained);
    auto s = rig.engine.settings();
    s.continuation = true;
    s.pitchBias = s.durationBias = s.onsetGapBias = s.shortGap = s.logitTd = true;
    s.tdLevel = 8;
    rig.engine.setSettings(s);
    *rig.model.waitD = 0;
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 600);
    EXPECT_GE(rig.played.size(), 3u);
    EXPECT_TRUE(rig.engine.lastError.empty()) << rig.engine.lastError;
}

namespace TdTest {

/** Model 1 with flat onset logits: the mean gap between its onsets, drawn with `gamma` at TD 8. */
auto meanGap(std::optional<float> gamma, std::pair<std::optional<double>, int> *sensitivity = nullptr) -> double {
    Rig rig(Engine::Mode::Unconstrained);
    auto s = rig.engine.settings();
    s.continuation = true;
    s.tdLevel = 8;
    s.tdGamma = gamma;
    rig.engine.setSettings(s);
    *rig.model.waitD = 0;
    *rig.model.onsetRel = 0; // only onset 0 (in the past) stands out: every legal onset is equally likely
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 3000);
    if (sensitivity != nullptr)
        *sensitivity = rig.engine.takeSensitivity();
    std::vector<int32_t> onsets;
    for (const auto &n: rig.played)
        onsets.push_back(n.onsetCs);
    std::sort(onsets.begin(), onsets.end());
    onsets.erase(std::unique(onsets.begin(), onsets.end()), onsets.end());
    EXPECT_GE(onsets.size(), 3u);
    return onsets.size() < 2 ? 0.0
                             : static_cast<double>(onsets.back() - onsets.front()) / static_cast<double>(onsets.size() - 1);
}

} // namespace TdTest

TEST(InformativeWaitEngine, TdGammaShortensOrLengthensTheOnsetGaps) {
    std::pair<std::optional<double>, int> none, plus;
    const double flat = TdTest::meanGap(std::nullopt, &none);
    const double denser = TdTest::meanGap(3.0f, &plus);
    const double sparser = TdTest::meanGap(-2.0f);
    EXPECT_LT(denser, 0.7 * flat);
    EXPECT_GT(sparser, flat);
    // sensitivity is recorded only for draws made with a gamma, and is positive (more gamma, more density)
    EXPECT_EQ(none.second, 0);
    ASSERT_TRUE(plus.first.has_value());
    EXPECT_GT(plus.second, 0);
    EXPECT_GT(*plus.first, 0.0);
}

TEST(HybridInformativeWait, TicksOneTdControllerPerModelAndKeepsGammaOnPedalRelease) {
    ModelConfig config;
    HybridInformativeWaitReductionTransformer hy(config);
    auto silent = std::make_unique<ScriptBackend>();
    *silent->waitD = 0;
    *silent->onsetRel = 0;
    *silent->onsetSlope = 1.0f; // the next onset, as a dense real model: model 1 has no per-draw limit (as v1), and
    // without a clock the ahead throttle (outputMaxAheadSeconds) never stops it running ahead
    hy.initWithBackends(std::move(silent), std::make_unique<ScriptBackend>());
    int32_t now = 0;
    int64_t wall = 0;
    int32_t td = TemporalDensity::kAuto;
    hy.nowCsProvider = [&] { return now; };
    hy.wallMsProvider = [&] { return wall; };
    hy.pianoSilentForMs = [] { return INT64_MAX; };
    hy.tdValueProvider = [&] { return td; };
    Token t{0, static_cast<int32_t>(DenseVocab::DurOffset) + 20, static_cast<int32_t>(DenseVocab::NoteOffset) + 60};
    t.velocity = 80;
    hy.inputTokenQueue.push(t);
    const auto runFor = [&](int ms) {
        for (int i = 0; i < ms / 10; ++i) {
            hy.stepOnce();
            now += 1;
            wall += 10;
        }
    };
    runFor(2500);
    auto views = hy.getTdControlStates();
    ASSERT_EQ(views.size(), 2u);
    EXPECT_EQ(views[0].model, "m1");
    EXPECT_EQ(views[1].model, "m2");
    EXPECT_EQ(views[0].state.why, "TD AUTO");
    EXPECT_FALSE(hy.tdGamma(true).has_value());

    // TD 64: both controllers start from the warm start (startGamma(64) = 4, Nov 25 2026). Model 1 draws, so its
    // gamma moves from there by at most one step per 2 s update (up to kMaxStepUp, down kMaxStep); model 2 is not
    // drawing and stays frozen at the warm start.
    td = 64;
    runFor(6500);
    views = hy.getTdControlStates();
    ASSERT_EQ(views.size(), 2u);
    const double start = TdController::startGamma(64);
    EXPECT_FALSE(views[0].state.frozen) << views[0].state.why;
    EXPECT_NE(views[0].state.gamma, start) << "model 1's controller updated";
    EXPECT_LE(views[0].state.gamma, start + 3 * TdController::kMaxStepUp + 1e-9);
    EXPECT_GE(views[0].state.gamma, start - 3 * TdController::kMaxStep - 1e-9);
    EXPECT_TRUE(views[1].state.frozen);
    EXPECT_DOUBLE_EQ(views[1].state.gamma, start);
    ASSERT_TRUE(hy.tdGamma(true).has_value());
    EXPECT_FLOAT_EQ(*hy.tdGamma(true), static_cast<float>(views[0].state.gamma));

    // a pedal release (generation resumes) only soft-resets: a positive gamma is kept for both models
    const auto m1Before = *hy.tdGamma(true);
    const auto m2Before = *hy.tdGamma(false);
    ASSERT_GT(m1Before, 0.0f);
    hy.setGenerationPause(true, "test");
    hy.setGenerationPause(false, "test");
    hy.stepOnce();
    EXPECT_FLOAT_EQ(*hy.tdGamma(true), m1Before);
    EXPECT_FLOAT_EQ(*hy.tdGamma(false), m2Before);
}

TEST(HybridInformativeWait, AModeCutKeepsTheContextAndGenerationContinuesWithoutNewInput) {
    // A TD press or pedal-up (a mode cut) used to restart both streams: nothing was generated
    // until the pianist played again (Nov 24 2026). Now only the notes from the cut on go.
    ModelConfig config;
    HybridInformativeWaitReductionTransformer hy(config);
    auto silent = std::make_unique<ScriptBackend>();
    *silent->waitD = 0;
    *silent->onsetRel = 0;
    *silent->onsetSlope = 1.0f; // the next onset, as a dense real model: model 1 has no per-draw limit (as v1), and
    // without a clock the ahead throttle (outputMaxAheadSeconds) never stops it running ahead
    hy.initWithBackends(std::move(silent), std::make_unique<ScriptBackend>());
    int32_t now = 0;
    int64_t wall = 0;
    hy.nowCsProvider = [&] { return now; };
    hy.wallMsProvider = [&] { return wall; };
    hy.pianoSilentForMs = [] { return INT64_MAX; };
    hy.tdValueProvider = [] { return TemporalDensity::kAuto; };
    Token t{0, static_cast<int32_t>(DenseVocab::DurOffset) + 20, static_cast<int32_t>(DenseVocab::NoteOffset) + 60};
    t.velocity = 80;
    hy.inputTokenQueue.push(t);
    const auto runFor = [&](int cs) {
        for (int i = 0; i < cs; ++i) {
            hy.stepOnce();
            now += 1;
            wall += 10;
        }
    };
    runFor(300);
    ASSERT_FALSE(hy.getOutputHistory().empty()) << "model 1 continues the piano";
    const int32_t cut = now;
    hy.requestModeCut(cut, "TD press (test)");
    runFor(300); // no new piano note: a wiped context would draw nothing (an empty engine waits for the piano)
    const auto history = hy.getOutputHistory();
    const auto drawnAfterCut = std::count_if(history.begin(), history.end(), [cut](const Token &t) {
        return t.time >= cut && t.note >= static_cast<int32_t>(Vocab::NoteOffset) && t.note < static_cast<int32_t>(Vocab::Rest);
    });
    EXPECT_GT(drawnAfterCut, 0) << "generation continues from the kept context";
}

TEST(HybridInformativeWait, ACutBringsModelOneBackFromFarAhead) {
    // Nov 24 2026: after the piano cut model 1's far notes, its lower bound stayed at the furthest onset
    // ever drawn, so it kept drawing tens of seconds ahead and model 2 was switched out at once.
    ModelConfig config;
    HybridInformativeWaitReductionTransformer hy(config);
    auto silent = std::make_unique<ScriptBackend>();
    auto *model1 = silent.get();
    *silent->waitD = 0;
    *silent->onsetRel = 3000; // first, model 1 likes an onset 30 s in
    hy.initWithBackends(std::move(silent), std::make_unique<ScriptBackend>());
    int32_t now = 0;
    int64_t wall = 0;
    hy.nowCsProvider = [&] { return now; };
    hy.wallMsProvider = [&] { return wall; };
    hy.pianoSilentForMs = [] { return INT64_MAX; };
    hy.tdValueProvider = [] { return TemporalDensity::kAuto; };
    const auto piano = [&](int32_t at, int pitch) {
        Token t{at, static_cast<int32_t>(DenseVocab::DurOffset) + 20, static_cast<int32_t>(DenseVocab::NoteOffset) + pitch};
        t.velocity = 80;
        hy.inputTokenQueue.push(t);
    };
    const auto runFor = [&](int cs) {
        for (int i = 0; i < cs; ++i) {
            hy.stepOnce();
            now += 1;
            wall += 10;
        }
    };
    piano(0, 60);
    runFor(50);
    const auto far = hy.getOutputHistory();
    ASSERT_TRUE(std::any_of(far.begin(), far.end(), [](const Token &t) { return t.time >= 2900; }))
        << "model 1 drew far ahead";
    // the pianist plays: the far notes are cut (Jam); model 1 now prefers the nearest onset
    *model1->onsetRel = 0;
    *model1->onsetSlope = 1.0f;
    piano(now, 62);
    runFor(1); // the cut (the first draw may still use logits cached before the fake model changed)
    const auto farAt = [](const std::vector<Token> &h) {
        return std::count_if(h.begin(), h.end(), [](const Token &t) { return t.time >= 2900; });
    };
    const auto farAfterCut = farAt(hy.getOutputHistory());
    EXPECT_LE(farAfterCut, 1) << "the far notes were cut";
    runFor(99);
    const auto after = hy.getOutputHistory();
    EXPECT_TRUE(std::any_of(after.begin(), after.end(), [&](const Token &t) {
        return t.time > 50 && t.time < now + 100 && t.note >= static_cast<int32_t>(Vocab::NoteOffset)
               && t.note < static_cast<int32_t>(Vocab::Rest);
    })) << "model 1 draws near the present again";
    EXPECT_EQ(farAt(after), farAfterCut) << "and none is drawn far ahead after it";
}

// ---- the real export (skipped unless the files are on this machine) ----

auto home() -> std::string {
    const char *h = std::getenv("HOME");
    return h != nullptr ? h : "";
}

auto modelPath() -> std::string {
    const char *p = std::getenv("IW_MODEL");
    return p != nullptr ? p : home() + "/Documents/virtual_orch_models/rt_simultaneous_latest.onnx";
}

auto readTokens(const std::string &path) -> std::vector<int32_t> {
    std::ifstream in(path);
    std::vector<int32_t> t;
    int32_t x = 0;
    while (in >> x)
        t.push_back(x);
    return t;
}

TEST(InformativeWaitOnnx, BatchedBranchesEqualBatchOne) {
    const auto tokensPath = home() + "/orchestration/samples/overnight_112226/v2/m2_r10__unconstrained/sample_00/tokens.txt";
    if (! juce::File(modelPath()).existsAsFile() || ! juce::File(tokensPath).existsAsFile())
        GTEST_SKIP() << "export or sample tokens not on this machine";
    OnnxBackend backend;
    std::string err;
    ASSERT_TRUE(backend.init(modelPath().c_str(), err)) << err;
    ASSERT_EQ(backend.vocabSize(), static_cast<size_t>(Engine::kVocab));
    auto toks = readTokens(tokensPath);
    toks.resize(std::min<size_t>(toks.size(), 400));
    CommittedScratchCache cache(backend);
    const State *state = cache.stateAfter(toks, toks.size());
    ASSERT_NE(state, nullptr);
    const State one = *state;
    BatchState tiled;
    ASSERT_TRUE(tileState(one, backend.dims(), 5, tiled));
    const std::vector<int64_t> feed{11060, 11064, 11067, 11072, 11048};
    BatchState next;
    std::vector<float> batched;
    const auto t0 = std::chrono::steady_clock::now();
    ASSERT_TRUE(backend.runBatched(tiled, feed, next, batched));
    const double msB = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    float worst = 0;
    double msOne = 0;
    for (size_t b = 0; b < 5; ++b) {
        State present;
        std::vector<float> last;
        const auto t1 = std::chrono::steady_clock::now();
        ASSERT_TRUE(backend.run(one, {feed[b]}, present, last));
        msOne += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        for (size_t i = 0; i < last.size(); ++i)
            worst = std::max(worst, std::abs(last[i] - batched[b * last.size() + i]));
    }
    EXPECT_LT(worst, 1e-3f);
    // three tokens per branch (the precompute's [piano pitch, velocity, T] feed)
    {
        std::vector<int64_t> feed3;
        for (const auto f: feed)
            feed3.insert(feed3.end(), {f, 11640 + 64, 300});
        BatchState next3;
        std::vector<float> batched3;
        ASSERT_TRUE(backend.runBatched(tiled, feed3, next3, batched3));
        EXPECT_EQ(next3.rows, tiled.rows + 3);
        float worst3 = 0;
        for (size_t b = 0; b < 5; ++b) {
            State present;
            std::vector<float> last;
            ASSERT_TRUE(backend.run(one, {feed3[b * 3], feed3[b * 3 + 1], feed3[b * 3 + 2]}, present, last));
            for (size_t i = 0; i < last.size(); ++i)
                worst3 = std::max(worst3, std::abs(last[i] - batched3[b * last.size() + i]));
        }
        EXPECT_LT(worst3, 1e-3f);
        std::cout << "[iw] 3 tokens per branch: max |logit diff| " << worst3 << std::endl;
    }
    std::cout << "[iw] batch-5 call " << msB << " ms, 5 batch-1 calls " << msOne << " ms (avg " << msOne / 5
              << "), max |logit diff| " << worst << std::endl;
}

TEST(InformativeWaitOnnx, CommitFromScratchMatchesRefeeding) {
    // committed prefix lagging 12 tokens behind, growing by 1-5 tokens per call: the cache that cuts C out
    // of W must give the same logits as the one that feeds the committed tokens again
    const auto tokensPath = home() + "/orchestration/samples/overnight_112226/v2/m2_r10__unconstrained/sample_00/tokens.txt";
    if (! juce::File(modelPath()).existsAsFile() || ! juce::File(tokensPath).existsAsFile())
        GTEST_SKIP() << "export or sample tokens not on this machine";
    OnnxBackend backend;
    std::string err;
    ASSERT_TRUE(backend.init(modelPath().c_str(), err)) << err;
    const auto toks = readTokens(tokensPath);
    CommittedScratchCache plain(backend), reuse(backend);
    reuse.reuseScratchOnCommit = true;
    float worst = 0;
    size_t n = 20;
    int callsPlain = 0, callsReuse = 0;
    for (int step = 0; step < 24 && n < toks.size(); ++step) {
        const std::vector<int32_t> desired(toks.begin(), toks.begin() + static_cast<std::ptrdiff_t>(n));
        const size_t committed = n > 12 ? n - 12 : 0;
        const auto a = plain.logitsAfter(desired, committed);
        const auto b = reuse.logitsAfter(desired, committed);
        ASSERT_EQ(a.size(), b.size());
        for (size_t i = 0; i < a.size(); ++i)
            worst = std::max(worst, std::abs(a[i] - b[i]));
        n += 1 + static_cast<size_t>(step % 5);
    }
    callsPlain = plain.backendCalls();
    callsReuse = reuse.backendCalls();
    EXPECT_LT(worst, 1e-3f);
    std::cout << "[iw] commit-from-scratch: max |logit diff| " << worst << ", backend calls " << callsReuse
              << " vs " << callsPlain << ", tokens fed " << reuse.tokensFed() << " vs " << plain.tokensFed() << std::endl;
}

TEST(InformativeWaitOnnx, HybridV11LoadsBothModels) {
    const auto dir = home() + "/Documents/virtual_orch_models/";
    const auto m1 = dir + "rt_continuation_latest.onnx";
    const auto m2 = dir + "rt_simultaneous_latest.onnx";
    if (! juce::File(m1).existsAsFile() || ! juce::File(m2).existsAsFile())
        GTEST_SKIP() << "rt_continuation_latest / rt_simultaneous_latest not on this machine";
    ModelConfig config;
    HybridInformativeWaitReductionTransformer hy(config);
    ASSERT_TRUE(hy.init(m1.c_str(), m2.c_str())) << hy.lastInitError;
    EXPECT_TRUE(hy.isModelLoaded());
    // the wrong way round is refused (vocab check)
    HybridInformativeWaitReductionTransformer wrong(config);
    EXPECT_FALSE(wrong.init(m2.c_str(), m1.c_str()));
}

TEST(InformativeWaitOnnx, HybridV12LoadsTheChordModel) {
    const auto dir = home() + "/Documents/virtual_orch_models/";
    const auto m1 = dir + "rt_continuation_latest.onnx";
    const auto m2 = dir + "simultaneous_chordtokens_m2c_r2b_kv.onnx";
    const auto bank = dir + "voicing_bank_m2c.json";
    if (! juce::File(m1).existsAsFile() || ! juce::File(m2).existsAsFile() || ! juce::File(bank).existsAsFile())
        GTEST_SKIP() << "rt_continuation_latest / simultaneous_chordtokens_m2c_r2b_kv / voicing_bank_m2c not on this machine";
    ModelConfig config;
    HybridInformativeWaitReductionTransformer hy(config);
    hy.loadVoicingBank(bank);
    ASSERT_TRUE(hy.init(m1.c_str(), m2.c_str())) << hy.lastInitError;
    EXPECT_TRUE(hy.isModelLoaded());
    EXPECT_TRUE(hy.isChordModel()); // vocab 14071
    EXPECT_STREQ(hy.label(), "Hybrid v1.2");
}

TEST(InformativeWaitOnnx, MatchesPythonSampler) {
    // The Python sampler's trace: for each pick, the top-25 raw probabilities after tokens[:pos].
    const auto tracePath = home() + "/orchestration/samples/overnight_112226/v2/m2_r10_p30_q03__unconstrained/sample_00/trace.json.gz";
    if (! juce::File(modelPath()).existsAsFile() || ! juce::File(tracePath).existsAsFile())
        GTEST_SKIP() << "export or trace not on this machine";
    juce::FileInputStream raw(juce::File{tracePath});
    juce::GZIPDecompressorInputStream gz(&raw, false, juce::GZIPDecompressorInputStream::gzipFormat);
    const auto trace = juce::JSON::parse(gz.readEntireStreamAsString());
    ASSERT_TRUE(trace["meta"].isObject());
    std::vector<int32_t> toks;
    for (const auto &t: *trace["meta"]["tokens"].getArray())
        toks.push_back(static_cast<int32_t>(t));
    OnnxBackend backend;
    std::string err;
    ASSERT_TRUE(backend.init(modelPath().c_str(), err)) << err;
    CommittedScratchCache cache(backend);
    int checked = 0;
    float worst = 0;
    for (const auto &step: *trace["steps"].getArray()) {
        if (step["t"].toString() != "pick")
            continue;
        const int pos = static_cast<int>(step["pos"]);
        if (pos < 8 || pos > 480 || pos > static_cast<int>(toks.size()))
            continue;
        const std::vector<int32_t> prefix(toks.begin(), toks.begin() + pos);
        const auto &logits = cache.logitsAfter(prefix, prefix.size());
        ASSERT_EQ(logits.size(), static_cast<size_t>(Engine::kVocab));
        const float peak = *std::max_element(logits.begin(), logits.end());
        double z = 0;
        for (const float l: logits)
            z += std::exp(static_cast<double>(l - peak));
        for (const auto &pair: *step["raw_top"].getArray()) {
            const int id = static_cast<int>(pair[0]);
            const auto p = static_cast<float>(static_cast<double>(pair[1]));
            const auto mine = static_cast<float>(std::exp(static_cast<double>(logits[static_cast<size_t>(id)] - peak)) / z);
            worst = std::max(worst, std::abs(mine - p));
        }
        if (++checked >= 60)
            break;
    }
    ASSERT_GT(checked, 20);
    EXPECT_LT(worst, 2e-3f) << "C++ ONNX vs Python (CUDA, fp32) top-25 probabilities";
    std::cout << "[iw] " << checked << " picks vs Python: max |p diff| " << worst << std::endl;
}

} // namespace

// ---------------------------------------------------------------- rewind / cut

namespace {
/** A continuation engine (Hybrid's model 1): no WAITs, the next onset the likeliest, so it plays densely. */
auto denseRig() -> std::unique_ptr<Rig> {
    auto rig = std::make_unique<Rig>(Engine::Mode::Unconstrained);
    auto s = rig->engine.settings();
    s.continuation = true;
    rig->engine.setSettings(s);
    *rig->model.waitD = 0;
    *rig->model.onsetSlope = 1.0f;
    return rig;
}

/** Index (in tokens()) of the first orchestra note with an onset at or after `cut`; size() if none. */
auto firstOrchFrom(const std::vector<int32_t> &toks, int32_t base, int32_t cut) -> size_t {
    for (size_t i = 2; i < toks.size(); ++i)
        if (toks[i] >= Engine::kOrch0 && toks[i] < Engine::kOrch0 + 128 && toks[i - 2] + base >= cut)
            return i - 2;
    return toks.size();
}
} // namespace

TEST(InformativeWaitRewind, DropGeneratedFromLeavesExactlyWhatWasThereBeforeTheCut) {
    auto rig = denseRig();
    rig->engine.noteOn(0, 60, 80);
    rig->run(0, 100);
    rig->engine.noteOn(110, 64, 80);
    rig->run(101, 200);
    const auto before = rig->engine.tokens();
    const int32_t cut = 150;
    const size_t keep = firstOrchFrom(before, rig->engine.base(), cut);
    ASSERT_LT(keep, before.size()) << "the model drew past the cut";

    rig->engine.dropGeneratedFrom(cut);
    // as if nothing had been generated from the cut on: the same tokens up to it, nothing after
    EXPECT_EQ(rig->engine.tokens(), std::vector<int32_t>(before.begin(), before.begin() + static_cast<std::ptrdiff_t>(keep)));
}

TEST(InformativeWaitRewind, PianoNotesAfterTheCutAreWrittenBackInOrder) {
    auto rig = denseRig();
    rig->engine.noteOn(0, 60, 80);
    rig->run(0, 160);
    rig->engine.noteOn(170, 64, 80); // after the cut: the pianist's note stays
    rig->run(161, 200);
    rig->engine.dropGeneratedFrom(150);
    const auto toks = rig->engine.tokens();
    EXPECT_EQ(firstOrchFrom(toks, rig->engine.base(), 150), toks.size()) << "no generated note from the cut on";
    const auto piano64 = std::find(toks.begin(), toks.end(), Engine::kPiano0 + 64);
    ASSERT_NE(piano64, toks.end());
    EXPECT_EQ(*(piano64 - 2) + rig->engine.base(), 170);
    EXPECT_EQ(piano64 + 2, toks.end()) << "the piano note is now the last event";
}

TEST(InformativeWaitRewind, GenerationResumesFromNowAfterARewind) {
    auto rig = denseRig();
    rig->engine.noteOn(0, 60, 80);
    rig->run(0, 200);
    rig->engine.dropGeneratedFrom(150);
    rig->played.clear();
    rig->run(201, 260);
    ASSERT_FALSE(rig->played.empty()) << "the context is kept: the model goes on without new input";
    for (const auto &n: rig->played)
        EXPECT_GE(n.onsetCs, 201) << "no note in the past";
}

TEST(InformativeWaitRewind, AnInterruptedWaitLeavesNoUnansweredWaitInTheContext) {
    // A mode cut or a model switch interrupts a pending WAIT: training lines never hold a WAIT / COUNT
    // that neither fired nor EXPIRED, so the context must not either.
    Rig rig(Engine::Mode::Unconstrained);
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 6);
    ASSERT_TRUE(rig.engine.hasPendingWait());
    rig.engine.interrupt();
    rig.engine.dropGeneratedFrom(6);
    EXPECT_FALSE(rig.engine.hasPendingWait());
    const auto toks = rig.engine.tokens();
    ASSERT_FALSE(toks.empty());
    const bool endsWithWaitOrCount = toks.back() >= Engine::kWait0 && toks.back() < Engine::kExpired;
    EXPECT_FALSE(endsWithWaitOrCount) << "a WAIT / COUNT left dangling";
}

TEST(InformativeWaitRewind, AccompanimentKeepsTheGeneratedNoteAPianoNoteWouldCut) {
    Rig rig(Engine::Mode::Unconstrained);
    auto s = rig.engine.settings();
    s.inputCut = Engine::Settings::InputCut::Accompaniment;
    rig.engine.setSettings(s);
    *rig.model.waitD = 0;
    *rig.model.onsetRel = 40;
    rig.engine.noteOn(0, 60, 80);
    rig.run(0, 5);
    ASSERT_EQ(rig.played.size(), 1u);
    rig.engine.noteOn(20, 62, 80); // in Jam this cuts the note at 40
    rig.engine.step(20);
    EXPECT_EQ(rig.engine.takeCancelFrom(), INT32_MAX);
    EXPECT_EQ(rig.engine.stats().cutNotes, 0);
}

TEST(InformativeWaitRewind, EditCutsOnlyNotesFromTheKeptStretchOn) {
    for (const int32_t keepCs: {10, 30}) {
        Rig rig(Engine::Mode::Unconstrained);
        auto s = rig.engine.settings();
        s.inputCut = Engine::Settings::InputCut::Edit;
        s.editKeepCs = keepCs;
        rig.engine.setSettings(s);
        *rig.model.waitD = 0;
        *rig.model.onsetRel = 40;
        rig.engine.noteOn(0, 60, 80);
        rig.run(0, 5);
        ASSERT_EQ(rig.played.size(), 1u);
        rig.engine.noteOn(20, 62, 80); // the note at 40: cut when 40 >= 20 + keepCs
        rig.engine.step(20);
        if (keepCs == 10)
            EXPECT_EQ(rig.engine.takeCancelFrom(), 40) << "keep 10: 40 is past 30, cut";
        else
            EXPECT_EQ(rig.engine.takeCancelFrom(), INT32_MAX) << "keep 30: 40 is within 50, kept";
    }
}

TEST(HybridInformativeWait, AModeCutRemovesExactlyTheGeneratedNotesFromTheCut) {
    ModelConfig config;
    HybridInformativeWaitReductionTransformer hy(config);
    auto silent = std::make_unique<ScriptBackend>();
    *silent->waitD = 0;
    *silent->onsetRel = 0;
    *silent->onsetSlope = 1.0f;
    hy.initWithBackends(std::move(silent), std::make_unique<ScriptBackend>());
    int32_t now = 0;
    int64_t wall = 0;
    hy.nowCsProvider = [&] { return now; };
    hy.wallMsProvider = [&] { return wall; };
    hy.pianoSilentForMs = [] { return INT64_MAX; };
    hy.tdValueProvider = [] { return TemporalDensity::kAuto; };
    Token t{0, static_cast<int32_t>(DenseVocab::DurOffset) + 20, static_cast<int32_t>(DenseVocab::NoteOffset) + 60};
    t.velocity = 80;
    hy.inputTokenQueue.push(t);
    for (int i = 0; i < 300; ++i, ++now, wall += 10)
        hy.stepOnce();
    const auto isNote = [](const Token &x) {
        return x.note >= static_cast<int32_t>(Vocab::NoteOffset) && x.note < static_cast<int32_t>(Vocab::Rest);
    };
    const auto before = hy.getOutputHistory();
    const int32_t cut = now - 50;
    ASSERT_TRUE(std::any_of(before.begin(), before.end(), [&](const Token &x) { return isNote(x) && x.time >= cut; }))
        << "notes were generated after the cut";

    hy.setGenerationPause(true, "test: look right after the cut");
    hy.requestModeCut(cut, "TD press (test)");
    hy.stepOnce();
    const auto after = hy.getOutputHistory();
    std::vector<Token> expected;
    for (const auto &x: before)
        if (isNote(x) && x.time < cut)
            expected.push_back(x);
    std::vector<Token> kept;
    for (const auto &x: after)
        if (isNote(x))
            kept.push_back(x);
    ASSERT_EQ(kept.size(), expected.size()) << "the notes before the cut stay, the ones from it go";
    for (size_t i = 0; i < kept.size(); ++i) {
        EXPECT_EQ(kept[i].time, expected[i].time);
        EXPECT_EQ(kept[i].note, expected[i].note);
    }
}

TEST(HybridInformativeWait, InAccompanimentThePianistIsIgnored) {
    // Accompaniment: model 1 keeps continuing its own music. The pianist's notes reach neither model,
    // and playing does not bring in model 2.
    ModelConfig config;
    HybridInformativeWaitReductionTransformer hy(config);
    auto silent = std::make_unique<ScriptBackend>();
    *silent->waitD = 0;
    *silent->onsetRel = 0;
    *silent->onsetSlope = 1.0f; // the next onset, as a dense real model
    hy.initWithBackends(std::move(silent), std::make_unique<ScriptBackend>());
    int32_t now = 0;
    int64_t wall = 0;
    int64_t silentFor = INT64_MAX;
    hy.nowCsProvider = [&] { return now; };
    hy.wallMsProvider = [&] { return wall; };
    hy.pianoSilentForMs = [&] { return silentFor; };
    hy.tdValueProvider = [] { return TemporalDensity::kAuto; };
    const auto pianoNotesInContext = [&] {
        const auto data = hy.getInputData();
        int n = 0;
        for (size_t i = 0; i + 3 < data.size(); i += 4)
            n += DenseQuantize::instrumentOfNoteToken(data[i + 2]) == DenseConfig::PianoReductionInputInstrument ? 1 : 0;
        return n;
    };
    const auto play = [&](int pitch) {
        Token t{now, static_cast<int32_t>(DenseVocab::DurOffset) + 20, static_cast<int32_t>(DenseVocab::NoteOffset) + pitch};
        t.velocity = 80;
        hy.inputTokenQueue.push(t);
        silentFor = 0;
        for (int k = 0; k < 10; ++k, ++now, wall += 10)
            hy.stepOnce();
    };
    play(60); // one note in Jam: it goes in
    ASSERT_EQ(pianoNotesInContext(), 1);
    hy.setKeepQueueOnInput(true); // Accompaniment
    for (int i = 0; i < 20; ++i) // 2 s of playing
        play(62 + i % 5);
    EXPECT_EQ(pianoNotesInContext(), 1) << "no note played in Accompaniment reaches the models";
    EXPECT_TRUE(hy.usesSilentModel()) << "model 2 never takes over";
    EXPECT_EQ(hy.reason(), HybridInformativeWaitReductionTransformer::Reason::Accompaniment);
    const auto before = hy.getOutputHistory().size();
    for (int i = 0; i < 300; ++i, ++now, wall += 10)
        hy.stepOnce();
    EXPECT_GT(hy.getOutputHistory().size(), before) << "model 1 keeps going";
}

TEST(HybridInformativeWait, TheOperationLogGetsAnRtDrawPerNoteAndLoggingChangesNothing) {
    // Hybrid v1.1 wrote no RT_DRAW / RT_WAIT, so the Replay page had none (Nov 24 2026).
    const auto run = [](bool logging) {
        ModelConfig config;
        HybridInformativeWaitReductionTransformer hy(config);
        auto silent = std::make_unique<ScriptBackend>();
        *silent->waitD = 0;
        *silent->onsetRel = 0;
        *silent->onsetSlope = 1.0f;
        hy.initWithBackends(std::move(silent), std::make_unique<ScriptBackend>());
        int32_t now = 0;
        int64_t wall = 0;
        hy.nowCsProvider = [&] { return now; };
        hy.wallMsProvider = [&] { return wall; };
        hy.pianoSilentForMs = [] { return INT64_MAX; };
        hy.tdValueProvider = [] { return TemporalDensity::kAuto; };
        std::vector<std::string> lines;
        if (logging)
            OpLog::setCapture([&](const std::string &line) { lines.push_back(line); });
        Token t{0, static_cast<int32_t>(DenseVocab::DurOffset) + 20, static_cast<int32_t>(DenseVocab::NoteOffset) + 60};
        t.velocity = 80;
        hy.inputTokenQueue.push(t);
        for (int i = 0; i < 300; ++i, ++now, wall += 10)
            hy.stepOnce();
        OpLog::setCapture(nullptr);
        std::vector<std::array<int32_t, 3>> notes;
        for (const auto &tok: hy.getOutputHistory())
            if (tok.note >= static_cast<int32_t>(Vocab::NoteOffset) && tok.note < static_cast<int32_t>(Vocab::Rest))
                notes.push_back({tok.time, tok.duration, tok.note});
        return std::make_pair(notes, lines);
    };
    const auto [plain, none] = run(false);
    const auto [logged, lines] = run(true);
    ASSERT_FALSE(plain.empty());
    // the seed differs per run object only through threadRun (not used here): the same draws either way
    EXPECT_EQ(plain, logged) << "recording the draws must not change them";
    int draws = 0;
    for (const auto &l: lines)
        if (l.find("\"op\":\"RT_DRAW\"") != std::string::npos) {
            ++draws;
            EXPECT_NE(l.find("\"model\":\"m1\""), std::string::npos) << l;
            for (const char *field: {"\"field\":\"onset\"", "\"field\":\"dur\"", "\"field\":\"pitch\"", "\"field\":\"vel\""})
                EXPECT_NE(l.find(field), std::string::npos) << field << " in " << l;
        }
    EXPECT_GE(draws, static_cast<int>(logged.size())) << "an RT_DRAW for every note (and for notes cut later)";
}

TEST(InformativeWaitEngine, AMonophonicCapFiresOneNoteWhateverTheCount) {
    // The Launchpad's monophonic pad (max polyphony 1): a fired WAIT with COUNT 2 places one note,
    // precomputed or decoded.
    for (const int pianoPitch: {60, 50}) { // anticipated (precomputed) / not (decoded)
        Rig rig(Engine::Mode::Unconstrained);
        auto s = rig.engine.settings();
        s.outsideTopFiveAsRankOne = false;
        s.maxFireNotes = 1;
        rig.engine.setSettings(s);
        rig.engine.noteOn(0, 60, 80);
        rig.run(0, 30); // [WAIT 30][COUNT 2]
        rig.played.clear();
        rig.engine.noteOn(31, pianoPitch, 80);
        rig.run(31, 33);
        const auto atFire = std::count_if(rig.played.begin(), rig.played.end(), [](const auto &n) { return n.onsetCs == 31; });
        EXPECT_EQ(atFire, 1) << "piano " << pianoPitch;
        const auto placed = std::find_if(rig.events.begin(), rig.events.end(), [](const auto &e) { return e.what == "placed"; });
        ASSERT_NE(placed, rig.events.end());
        EXPECT_EQ(placed->a, 1);
    }
}

TEST(PolyphonyCapMono, MonophonicStaysAboveC4) {
    // The Mono pad: the line from C#4 up; other caps and no cap leave the register alone.
    EXPECT_EQ(PolyphonyCap::lowestPitch(1, 21, 108), 61);
    EXPECT_EQ(PolyphonyCap::lowestPitch(1, 77, 108), 77) << "register HIGH is above already";
    EXPECT_EQ(PolyphonyCap::lowestPitch(1, 21, 52), 21) << "register LOW keeps its range";
    EXPECT_EQ(PolyphonyCap::lowestPitch(0, 21, 108), 21);
}
