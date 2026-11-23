#include <gtest/gtest.h>

#include <JuceHeader.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include <VirtualOrch/reduction/InformativeWaitEngine.h>
#include <VirtualOrch/reduction/StreamingKvCache.h>

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
            // onset step: a WAIT of `waitD` cs (or an onset at `onsetRel` when waitD == 0)
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

TEST(InformativeWaitEngine, MissDecodesOnTheSpot) {
    Rig rig(Engine::Mode::Unconstrained);
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
    EXPECT_EQ(rig.engine.tokens()[1], Engine::kDur0 + 5); // held: the time it was down when written
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

// ---- the real export (skipped unless the files are on this machine) ----

auto home() -> std::string {
    const char *h = std::getenv("HOME");
    return h != nullptr ? h : "";
}

auto modelPath() -> std::string {
    const char *p = std::getenv("IW_MODEL");
    return p != nullptr ? p : home() + "/Documents/virtual_orch_models/informative_wait_m2_r10.onnx";
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

TEST(InformativeWaitOnnx, MatchesPythonSampler) {
    // The Python sampler's trace: for each pick, the top-25 raw probabilities after tokens[:pos].
    const auto tracePath = home() + "/orchestration/samples/overnight_112226/v2/m2_r10__unconstrained/sample_00/trace.json.gz";
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
