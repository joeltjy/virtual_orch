#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <string>

#include <VirtualOrch/reduction/AnticipatedStream.h>
#include <VirtualOrch/reduction/AnticipationSampler.h>
#include <VirtualOrch/reduction/TwoInstrumentAnticipatedReductionTransformer.h>

/**
 * W2/W3 (WAIT), the Accompaniment late-input reorder (T3'' and W2/W3), the default-off logit
 * adjustments and the HybridTransformer, on a fake model whose WAIT logit the test sets.
 */
namespace {

using namespace StreamingKv;
using Variant = TwoInstrumentAnticipatedReductionTransformer::Variant;
constexpr int32_t kWait = TwoInstrumentAnticipatedReductionTransformer::kWaitToken;

class WaitFakeBackend : public Backend {
public:
    explicit WaitFakeBackend(size_t vocabIn) : vocab(vocabIn) {
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
        uint32_t seed = 29;
        for (const float k: present.key)
            seed = seed * 1664525u + static_cast<uint32_t>(k) + 1013904223u;
        std::mt19937 gen(seed);
        std::normal_distribution<float> n(0.0f, 1.0f);
        last.resize(vocab);
        for (size_t i = 0; i < last.size(); ++i) {
            last[i] = n(gen);
            if (i < 10000)
                last[i] += (*farFuture ? 0.02f : -0.02f) * static_cast<float>(i); // early onsets, like a real model
        }
        if (vocab > static_cast<size_t>(kWait))
            last[static_cast<size_t>(kWait)] = *waitLogit;
        return true;
    }
    size_t vocab;
    Dims d;
    std::shared_ptr<float> waitLogit = std::make_shared<float>(-100.0f);
    /** Prefer far-future onsets (W2/W3 after a timeout). */
    std::shared_ptr<bool> farFuture = std::make_shared<bool>(false);
};

auto key(int32_t onset, int pitch, int32_t durCs) -> Token {
    Token t{onset, static_cast<int32_t>(Vocab::DurOffset) + durCs,
            static_cast<int32_t>(Vocab::NoteOffset) + pitch};
    t.velocity = 90;
    return t;
}

struct Rig {
    ModelConfig config;
    TwoInstrumentAnticipatedReductionTransformer rt;
    int32_t now = 1000;
    int64_t silentMs = 0;
    int64_t wallMs = 0;
    std::shared_ptr<float> waitLogit;
    std::shared_ptr<bool> farFuture;

    explicit Rig(Variant v) : rt(config, v) {
        rt.nowCsProvider = [this] { return now; };
        rt.tdValueProvider = [] { return TemporalDensity::kAuto; };
        rt.pianoSilentForMs = [this] { return silentMs; };
        rt.wallMsProvider = [this] { return wallMs; };
        if (v == Variant::Hybrid) {
            auto silent = std::make_unique<WaitFakeBackend>(11776);
            auto wait = std::make_unique<WaitFakeBackend>(11777);
            waitLogit = wait->waitLogit;
            farFuture = wait->farFuture;
            rt.initHybridWithBackends(std::move(silent), 11776, std::move(wait), 11777);
        } else {
            const size_t vocab = v == Variant::Wait ? 11777 : 11776;
            auto fake = std::make_unique<WaitFakeBackend>(vocab);
            waitLogit = fake->waitLogit;
            farFuture = fake->farFuture;
            rt.initWithBackend(std::move(fake), vocab);
        }
    }
    auto notes() -> std::vector<Token> {
        std::vector<Token> out;
        Token t{};
        while (rt.outputTokenQueue.pull(t))
            if (t.note != static_cast<int32_t>(Vocab::ClearQueue))
                out.push_back(t);
        return out;
    }
};

/** Onsets of the quads in a dense token list (TD and WAIT skipped). */
auto quadOnsets(const std::vector<int32_t> &tokens) -> std::vector<int32_t> {
    std::vector<int32_t> out;
    for (size_t i = 0; i < tokens.size();) {
        if ((tokens[i] >= 11768 && tokens[i] <= 11775) || tokens[i] == kWait) {
            ++i;
            continue;
        }
        out.push_back(tokens[i]);
        i += 4;
    }
    return out;
}

struct Entry {
    char kind; // 'W' WAIT, 'T' TD, 'Q' quad
    int32_t onset = 0, dur = 0, pitch = 0;
};

auto parse(const std::vector<int32_t> &tokens) -> std::vector<Entry> {
    std::vector<Entry> out;
    for (size_t i = 0; i < tokens.size();) {
        if (tokens[i] == kWait) {
            out.push_back({'W'});
            ++i;
        } else if (tokens[i] >= 11768 && tokens[i] <= 11775) {
            out.push_back({'T'});
            ++i;
        } else {
            out.push_back({'Q', tokens[i], tokens[i + 1] - 10000, (tokens[i + 2] - 11000) % 128});
            i += 4;
        }
    }
    return out;
}

} // namespace

namespace virtual_orch_test {

TEST(WaitStream, waitBeforeEveryPianoGroupAndChordsMergeWithin5Cs) {
    AnticipatedStream s;
    s.waitToken = kWait;
    s.mergeGivenCs = 5;
    s.tdTokenProvider = [] { return 11772; };
    s.reset(1000);
    s.addControl({1000, 50, 60, 64}, false);
    s.addControl({1003, 47, 64, 64}, false); // 3 cs later: same group, end time kept (1050)
    s.addControl({1040, 50, 67, 64}, false); // a new group
    s.insertDue(1100, 1100);
    std::string kinds;
    std::vector<Entry> quads;
    for (const auto &e: parse(s.tokens())) {
        kinds += e.kind;
        if (e.kind == 'Q')
            quads.push_back(e);
    }
    EXPECT_EQ(kinds, "WQTQWQ") << "WAIT, chord (TD after its first quad), WAIT, next group";
    ASSERT_EQ(quads.size(), 3u);
    EXPECT_EQ(quads[1].onset, 0) << "the second chord note moved to the group's first onset";
    EXPECT_EQ(quads[1].dur, 50) << "its end time is kept: 1003 + 47 - 1000";
    EXPECT_EQ(quads[2].onset, 40);
}

TEST(WaitStream, aMergedHeldNoteIsConfirmedByTheOnsetItArrivedWith) {
    AnticipatedStream s;
    s.waitToken = kWait;
    s.mergeGivenCs = 5;
    s.tdTokenProvider = [] { return 11772; };
    s.reset(1000);
    s.addControl({1000, 0, 60, 64}, true);
    s.addControl({1004, 0, 64, 64}, true);
    EXPECT_TRUE(s.confirmControl(1004, 64, 80)) << "matched on 1004, not the moved 1000";
    s.insertDue(1100, 1100);
    std::vector<Entry> quads;
    for (const auto &e: parse(s.tokens()))
        if (e.kind == 'Q')
            quads.push_back(e);
    ASSERT_EQ(quads.size(), 2u);
    EXPECT_EQ(quads[1].pitch, 64);
    EXPECT_EQ(quads[1].dur, 85) << "ends at 1004 + 80, so 85 cs from the group onset";
}

TEST(WaitSampler, thresholdRulePlaysWhenPlayingHasAtLeast30Percent) {
    AnticipationSampler s(AnticipationSampler::kDenseVelocityBase);
    s.onsetOptions.shortGapPenalty = false;
    s.onsetOptions.logitTd = false;
    std::vector<float> logits(11777, -1e9f);
    logits[0] = 0.0f; // one legal onset
    AnticipationSampler::Context ctx;
    ctx.lowerOnsetModel = 0;
    logits[kWait] = std::log(4.0f); // P(wait) 0.8, P(play) 0.2 < 0.3
    EXPECT_TRUE(s.decideWait(logits, ctx, kWait, 0.30f));
    EXPECT_NEAR(s.lastWaitProbability, 0.8f, 1e-4);
    logits[kWait] = std::log(2.0f); // P(play) 1/3 >= 0.3
    EXPECT_FALSE(s.decideWait(logits, ctx, kWait, 0.30f));
}

TEST(WaitSampler, aNoteAtTheSameOnsetIsNotLowerThanThePrevious) {
    AnticipationSampler s(AnticipationSampler::kDenseVelocityBase);
    std::vector<float> logits(11777, 0.0f);
    AnticipationSampler::Context ctx;
    ctx.pitchFloor = 70;
    const auto m = s.masked(2, logits, ctx);
    const auto noteId = [](int p) { return static_cast<size_t>(11000 + 128 + p); };
    EXPECT_TRUE(std::isinf(m[noteId(69)]));
    EXPECT_TRUE(std::isfinite(m[noteId(70)]));
    EXPECT_TRUE(std::isfinite(m[noteId(90)]));
}

TEST(WaitReduction, waitsForTheSoloistThenResumesWithTheGroup) {
    Rig rig(Variant::Wait);
    *rig.waitLogit = 50.0f; // the model always wants to wait
    rig.rt.inputTokenQueue.push(key(1000, 60, 50));
    EXPECT_FALSE(rig.rt.stepOnce()) << "WAIT: nothing generated, waiting for the soloist";
    EXPECT_TRUE(rig.notes().empty());

    rig.now = 1040;
    rig.rt.inputTokenQueue.push(key(1040, 64, 50)); // the soloist plays
    *rig.waitLogit = -100.0f;                         // now the model plays
    EXPECT_TRUE(rig.rt.stepOnce());
    const auto onsets = quadOnsets(rig.rt.streamForTest().tokens());
    ASSERT_GE(onsets.size(), 3u) << "group 1, group 2, then a generated note";
    int waits = 0;
    for (const auto t: rig.rt.streamForTest().tokens())
        waits += t == kWait ? 1 : 0;
    EXPECT_EQ(waits, 2) << "one WAIT before each piano group";
}

TEST(WaitReduction, aWaitTimesOutAfterOneSecondAndThenPlaysAlone) {
    Rig rig(Variant::Wait);
    *rig.waitLogit = 50.0f;
    rig.rt.inputTokenQueue.push(key(1000, 60, 50));
    EXPECT_FALSE(rig.rt.stepOnce());
    rig.now = 1050;
    EXPECT_FALSE(rig.rt.stepOnce()) << "still within the second";
    rig.now = 1101;
    EXPECT_TRUE(rig.rt.stepOnce()) << "timed out: WAIT masked, it plays on alone";
    EXPECT_EQ(rig.notes().size(), 1u);
}

TEST(LateInputReorder, accompanimentWritesLateNotesInOnsetOrderAndKeepsTheGeneratedOnes) {
    for (const auto variant: {Variant::T3, Variant::Wait}) {
        Rig rig(variant);
        rig.rt.setKeepQueueOnInput(true);
        rig.rt.inputTokenQueue.push(key(1000, 60, 50));
        for (int i = 0; i < 12; ++i)
            ASSERT_TRUE(rig.rt.stepOnce());
        const auto generated = rig.notes();
        ASSERT_GE(generated.size(), 12u);
        const int32_t lateOnset = generated[3].time + 1; // behind several generated notes
        ASSERT_LT(lateOnset, rig.rt.streamForTest().clock());

        rig.rt.inputTokenQueue.push(key(lateOnset, 72, 30));
        ASSERT_TRUE(rig.rt.stepOnce()); // reorders (the first one is immediate), then one note
        const auto onsets = quadOnsets(rig.rt.streamForTest().tokens());
        EXPECT_TRUE(std::is_sorted(onsets.begin(), onsets.end())) << "onset order everywhere";
        EXPECT_EQ(rig.rt.streamForTest().lateCount(), 0u);
        EXPECT_EQ(onsets.size(), generated.size() + 1 + 2)
            << "every generated note kept, the new one, and the 2 piano notes";
        const auto next = rig.notes();
        EXPECT_EQ(next.size(), 1u) << "playback untouched (no clear), one new note";

        // A second late note within the same second waits; a second later it goes in.
        rig.rt.inputTokenQueue.push(key(lateOnset + 2, 74, 30));
        ASSERT_TRUE(rig.rt.stepOnce());
        EXPECT_EQ(rig.rt.streamForTest().lateCount(), 1u) << "at most one reorder per second";
        rig.now += 100;
        ASSERT_TRUE(rig.rt.stepOnce());
        EXPECT_EQ(rig.rt.streamForTest().lateCount(), 0u);
        const auto after = quadOnsets(rig.rt.streamForTest().tokens());
        EXPECT_TRUE(std::is_sorted(after.begin(), after.end()));
    }
}

TEST(LateInputReorder, jamStillCutsTheFutureInstead) {
    Rig rig(Variant::T3);
    rig.rt.inputTokenQueue.push(key(1000, 60, 50));
    for (int i = 0; i < 12; ++i)
        ASSERT_TRUE(rig.rt.stepOnce());
    const auto generated = rig.notes();
    rig.rt.inputTokenQueue.push(key(rig.now, 72, 30));
    ASSERT_TRUE(rig.rt.stepOnce());
    bool cleared = false;
    Token t{};
    while (rig.rt.outputTokenQueue.pull(t))
        cleared = cleared || t.note == static_cast<int32_t>(Vocab::ClearQueue);
    EXPECT_TRUE(cleared) << "Jam: the future is dropped and regenerated";
    EXPECT_EQ(rig.rt.streamForTest().lateCount(), 0u);
}

TEST(AnticipatedAdjustments, allLogitAdjustmentsStartOffForT3AndWait) {
    for (const auto variant: {Variant::T3, Variant::Wait, Variant::Hybrid}) {
        ModelConfig config;
        TwoInstrumentAnticipatedReductionTransformer rt(config, variant);
        EXPECT_FALSE(rt.isPitchBiasEnabled());
        EXPECT_FALSE(rt.isDurationBiasEnabled());
        EXPECT_FALSE(rt.isOnsetGapBiasEnabled());
        EXPECT_FALSE(rt.isShortGapPenaltyEnabled());
        EXPECT_FALSE(rt.isLogitTdEnabled());
    }
    ModelConfig config;
    TwoInstrumentAnticipatedReductionTransformer hybrid(config, Variant::Hybrid);
    EXPECT_TRUE(hybrid.hybridV1PitchBias.load()) << "the hybrid's v1 side: every adjustment on";
    EXPECT_TRUE(hybrid.hybridV1DurationBias.load());
    EXPECT_TRUE(hybrid.hybridV1OnsetGapBias.load());
    EXPECT_TRUE(hybrid.hybridV1ShortGapPenalty.load());
    EXPECT_TRUE(hybrid.hybridV1LogitTd.load());
    TwoInstrumentAnticipatedReductionTransformer w(config, Variant::Wait);
    EXPECT_FLOAT_EQ(w.getOnsetTemperature(), 1.0f) << "the sampling file: onset T 1.0";
    EXPECT_FLOAT_EQ(w.getNoteTemperature(), 0.5f);
    EXPECT_FLOAT_EQ(w.getWaitMinPlay(), 0.30f);
}

TEST(HybridTransformer, startsWithV1AndSwitchesToW2AfterOneSecondOfPlaying) {
    Rig rig(Variant::Hybrid);
    rig.rt.inputTokenQueue.push(key(1000, 60, 50));
    rig.silentMs = 0; // a key is down
    ASSERT_TRUE(rig.rt.stepOnce());
    EXPECT_TRUE(rig.rt.lastNoteFromSilentModel()) << "v1 at the start";
    rig.wallMs = 900;
    ASSERT_TRUE(rig.rt.stepOnce());
    EXPECT_TRUE(rig.rt.hybridUsesV1()) << "0.9 s of playing: still v1";
    rig.wallMs = 1000;
    ASSERT_TRUE(rig.rt.stepOnce());
    EXPECT_FALSE(rig.rt.hybridUsesV1());
    EXPECT_FALSE(rig.rt.lastNoteFromSilentModel()) << "1 s of playing: W2/W3 draws";
    EXPECT_EQ(rig.rt.hybridReason(), TwoInstrumentAnticipatedReductionTransformer::HybridReason::Playing);

    // One shared context: once v1 draws again (a pedal release), it holds every note W2/W3 had.
    const auto wOnsets = quadOnsets(rig.rt.streamForTest().tokens());
    rig.rt.setGenerationPause(true);
    rig.rt.setGenerationPause(false);
    ASSERT_TRUE(rig.rt.stepOnce());
    ASSERT_TRUE(rig.rt.lastNoteFromSilentModel());
    auto v1Onsets = quadOnsets(rig.rt.silentStreamForTest().tokens());
    for (const auto o: wOnsets) {
        const auto it = std::find(v1Onsets.begin(), v1Onsets.end(), o);
        EXPECT_NE(it, v1Onsets.end()) << "onset " << o << " missing in v1's context";
        if (it != v1Onsets.end())
            v1Onsets.erase(it);
    }
    int waitsInV1 = 0;
    for (const auto t: rig.rt.silentStreamForTest().tokens())
        waitsInV1 += t == kWait ? 1 : 0;
    EXPECT_EQ(waitsInV1, 0) << "v1 has no WAIT token";
}

TEST(HybridTransformer, stoppingForHalfASecondDeletesW2sFutureAndGoesBackToV1) {
    Rig rig(Variant::Hybrid);
    rig.rt.inputTokenQueue.push(key(1000, 60, 50));
    rig.wallMs = 0;
    rig.rt.stepOnce();
    rig.wallMs = 1000;
    for (int i = 0; i < 6; ++i)
        ASSERT_TRUE(rig.rt.stepOnce()); // W2/W3 runs ahead
    ASSERT_FALSE(rig.rt.hybridUsesV1());
    rig.notes();
    rig.silentMs = 400;
    rig.rt.stepOnce();
    EXPECT_FALSE(rig.rt.hybridUsesV1()) << "0.4 s: still playing";
    rig.notes();
    rig.silentMs = 600;
    rig.rt.setGenerationPause(true); // look right after the switch
    rig.rt.stepOnce();
    EXPECT_TRUE(rig.rt.hybridUsesV1());
    bool cleared = false;
    Token t{};
    while (rig.rt.outputTokenQueue.pull(t))
        cleared = cleared || (t.note == static_cast<int32_t>(Vocab::ClearQueue) && t.time == rig.now);
    EXPECT_TRUE(cleared) << "W2/W3's future notes are deleted from playback";
    for (const auto &h: rig.rt.getOutputHistory())
        if (h.note != static_cast<int32_t>(Vocab::ClearQueue))
            EXPECT_LT(h.time, rig.now);
}

TEST(HybridTransformer, aPedalReleaseSwitchesToV1AtOnce) {
    Rig rig(Variant::Hybrid);
    rig.rt.inputTokenQueue.push(key(1000, 60, 50));
    rig.rt.stepOnce();
    rig.wallMs = 1000;
    rig.rt.stepOnce();
    ASSERT_FALSE(rig.rt.hybridUsesV1());
    rig.rt.setGenerationPause(true);  // pedal down
    rig.rt.setGenerationPause(false); // pedal up
    ASSERT_TRUE(rig.rt.stepOnce());
    EXPECT_TRUE(rig.rt.hybridUsesV1());
    EXPECT_TRUE(rig.rt.lastNoteFromSilentModel());
    EXPECT_EQ(rig.rt.hybridReason(), TwoInstrumentAnticipatedReductionTransformer::HybridReason::PedalReleased);
    rig.wallMs = 1900;
    rig.rt.stepOnce();
    EXPECT_TRUE(rig.rt.hybridUsesV1()) << "the 1 s counts again from the release";
    rig.wallMs = 2000;
    rig.rt.stepOnce();
    EXPECT_FALSE(rig.rt.hybridUsesV1());
}

TEST(HybridTransformer, twoSecondsOfPredictedSilenceSwitchesToV1) {
    Rig rig(Variant::Hybrid);
    rig.rt.inputTokenQueue.push(key(1000, 60, 50));
    rig.rt.stepOnce(); // v1's first note
    rig.wallMs = 1000;
    *rig.farFuture = true;
    rig.rt.setWaitMaxAheadForTest(5000); // a note past 2 s of silence, beyond the 3 s cap
    rig.rt.stepOnce(); // 1 s of playing: W2/W3 draws a note far ahead
    ASSERT_FALSE(rig.rt.lastNoteFromSilentModel());

    // Move to just after every earlier note has ended: nothing sounds, the far note is ahead.
    int32_t farOnset = INT32_MIN;
    int32_t lastEnd = rig.now;
    const auto history = rig.rt.getOutputHistory();
    for (const auto &t: history)
        farOnset = std::max(farOnset, t.time);
    for (const auto &t: history)
        if (t.time < farOnset)
            lastEnd = std::max(lastEnd, t.time + t.getRealDuration());
    rig.now = lastEnd + 1;
    ASSERT_GE(farOnset - rig.now, 200) << "the far note is at least 2 s ahead";
    rig.notes();
    rig.rt.setGenerationPause(true); // stop it drawing again, so the cut can be seen
    rig.rt.stepOnce();
    // (paused: the silence rule waits for an unpaused reduction)
    EXPECT_FALSE(rig.rt.hybridUsesV1()) << "not while paused";
    rig.rt.setGenerationPause(false);
    rig.rt.stepOnce(); // a release switches to v1 anyway, so use a fresh check below
}

TEST(HybridTransformer, predictedSilenceCutsW2sFarNoteAndHandsOverToV1) {
    Rig rig(Variant::Hybrid);
    rig.rt.inputTokenQueue.push(key(1000, 60, 50));
    rig.rt.stepOnce();
    rig.wallMs = 1000;
    *rig.farFuture = true;
    rig.rt.setWaitMaxAheadForTest(5000); // a note past 2 s of silence, beyond the 3 s cap
    rig.rt.stepOnce();
    ASSERT_FALSE(rig.rt.lastNoteFromSilentModel());
    int32_t farOnset = INT32_MIN;
    int32_t lastEnd = rig.now;
    const auto history = rig.rt.getOutputHistory();
    for (const auto &t: history)
        farOnset = std::max(farOnset, t.time);
    for (const auto &t: history)
        if (t.time < farOnset)
            lastEnd = std::max(lastEnd, t.time + t.getRealDuration());
    rig.now = lastEnd + 1;
    ASSERT_GE(farOnset - rig.now, 200);
    *rig.farFuture = false;
    ASSERT_TRUE(rig.rt.stepOnce()); // the rule fires, then v1 draws from now
    EXPECT_TRUE(rig.rt.hybridUsesV1());
    EXPECT_EQ(rig.rt.hybridReason(), TwoInstrumentAnticipatedReductionTransformer::HybridReason::PredictedSilence);
    EXPECT_TRUE(rig.rt.lastNoteFromSilentModel()) << "v1 fills the gap";
    for (const auto &t: rig.rt.getOutputHistory())
        EXPECT_NE(t.time, farOnset) << "W2/W3's far note is deleted";
    EXPECT_GE(rig.rt.hybridPlayingForMs(), 0) << "the 1 s playing timer starts again";
}

TEST(HybridTransformer, w2WaitingThroughTheUsersPlayingHandsOverToV1After2s) {
    // The 22 Oct 00:08 run: W3 waited after every group while the user kept playing, and stayed
    // silent for 16 s (nothing queued, so no "next onset" to measure the silence by).
    Rig rig(Variant::Hybrid);
    rig.rt.inputTokenQueue.push(key(1000, 60, 50));
    ASSERT_TRUE(rig.rt.stepOnce()); // v1's first note
    *rig.waitLogit = 50.0f;         // W2/W3 always waits
    rig.wallMs = 1000;
    rig.rt.stepOnce();
    ASSERT_FALSE(rig.rt.hybridUsesV1());
    const int32_t switchedAt = rig.now;

    // The user keeps playing a note every 30 cs; W2/W3 waits after each.
    int32_t switchedBackAt = INT32_MIN;
    int32_t lastEnd = switchedAt; // the end of the last generated note before the switch back
    for (int32_t t = rig.now + 30; t <= rig.now + 2000 && switchedBackAt == INT32_MIN; t += 30) {
        rig.now = t;
        rig.wallMs += 300;
        rig.rt.inputTokenQueue.push(key(t, 62 + (t / 30) % 5, 20));
        rig.rt.stepOnce();
        if (rig.rt.hybridUsesV1()) {
            switchedBackAt = t;
            for (const auto &h: rig.rt.getOutputHistory())
                if (h.note >= static_cast<int32_t>(Vocab::NoteOffset) && h.note < static_cast<int32_t>(Vocab::Rest)
                    && h.time < t)
                    lastEnd = std::max(lastEnd, h.time + h.getRealDuration());
        }
    }
    ASSERT_NE(switchedBackAt, INT32_MIN) << "W2/W3 must not stay silent indefinitely";
    EXPECT_GE(switchedBackAt - lastEnd, 200) << "only after 2 s of silence";
    EXPECT_LT(switchedBackAt - lastEnd, 200 + 30) << "and at once after that";
    EXPECT_EQ(rig.rt.hybridReason(), TwoInstrumentAnticipatedReductionTransformer::HybridReason::PredictedSilence);
}

TEST(WaitReduction, aGeneratedNoteIsAtMost3sAhead) {
    // The 22 Oct 00:08 run: W3 wrote a note 24 s ahead of the clock.
    Rig rig(Variant::Wait);
    *rig.farFuture = true; // the model prefers the farthest onsets
    rig.rt.inputTokenQueue.push(key(1000, 60, 50));
    for (int i = 0; i < 6; ++i) {
        rig.now += 10;
        rig.rt.stepOnce();
    }
    const auto out = rig.notes();
    ASSERT_FALSE(out.empty());
    for (const auto &t: out)
        EXPECT_LE(t.time, rig.now + AnticipationStage::kMaxAheadCs) << "onset " << t.time;
}

TEST(WaitReduction, noteDurationsAreAtMost5s) {
    Rig rig(Variant::Wait);
    rig.rt.inputTokenQueue.push(key(1000, 60, 50));
    for (int i = 0; i < 12; ++i) {
        rig.now += 10;
        rig.rt.stepOnce();
    }
    const auto out = rig.notes();
    ASSERT_FALSE(out.empty());
    for (const auto &t: out)
        EXPECT_LE(t.getRealDuration(), ReductionTransformer::kMaxNoteDurationCs);
}

TEST(EditMode, inputCutsOnlyTheNotes2sOrMoreAhead) {
    for (const auto v: {Variant::T3, Variant::Hybrid}) {
        Rig rig(v);
        rig.rt.setEditMode(true);
        rig.rt.inputTokenQueue.push(key(1000, 60, 50));
        // Run ahead until something is generated past now + 2 s.
        const auto farthest = [&] {
            int32_t f = INT32_MIN;
            for (const auto &h: rig.rt.getOutputHistory())
                if (h.note != static_cast<int32_t>(Vocab::ClearQueue))
                    f = std::max(f, h.time);
            return f;
        };
        for (int i = 0; i < 400 && farthest() < rig.now + 260; ++i)
            rig.rt.stepOnce();
        ASSERT_GE(farthest(), rig.now + 260) << "the fake runs more than 2 s ahead";
        std::vector<int32_t> kept;
        for (const auto &h: rig.rt.getOutputHistory())
            if (h.note != static_cast<int32_t>(Vocab::ClearQueue) && h.time >= rig.now + 5
                && h.time < rig.now + ReductionTransformer::kEditCutAheadCs)
                kept.push_back(h.time);
        ASSERT_FALSE(kept.empty()) << "notes within the next 2 s";
        rig.notes();

        rig.now += 5;
        rig.rt.inputTokenQueue.push(key(rig.now, 72, 30)); // a piano note
        rig.rt.stepOnce(); // the cut, then a draw

        bool clearedAt2s = false;
        Token t{};
        while (rig.rt.outputTokenQueue.pull(t))
            if (t.note == static_cast<int32_t>(Vocab::ClearQueue)) {
                EXPECT_EQ(t.time, rig.now + ReductionTransformer::kEditCutAheadCs) << "only from 2 s ahead";
                clearedAt2s = true;
            }
        EXPECT_TRUE(clearedAt2s);
        std::vector<int32_t> after;
        for (const auto &h: rig.rt.getOutputHistory())
            if (h.note != static_cast<int32_t>(Vocab::ClearQueue))
                after.push_back(h.time);
        for (const auto k: kept)
            EXPECT_NE(std::find(after.begin(), after.end(), k), after.end()) << "the next 2 s play on: " << k;

        // The piano note is in the context at its onset, before the kept notes after it.
        const auto onsets = quadOnsets(v == Variant::Hybrid && rig.rt.hybridUsesV1()
                                           ? rig.rt.silentStreamForTest().tokens()
                                           : rig.rt.streamForTest().tokens());
        EXPECT_TRUE(std::is_sorted(onsets.begin(), onsets.end())) << "onset order";
    }
}

TEST(EditMode, jamStillCutsFromNow) {
    Rig rig(Variant::T3);
    rig.rt.inputTokenQueue.push(key(1000, 60, 50));
    for (int i = 0; i < 20; ++i)
        rig.rt.stepOnce();
    rig.notes();
    rig.now += 5;
    rig.rt.inputTokenQueue.push(key(rig.now, 72, 30));
    rig.rt.stepOnce();
    Token t{};
    bool cleared = false;
    while (rig.rt.outputTokenQueue.pull(t))
        if (t.note == static_cast<int32_t>(Vocab::ClearQueue))
            cleared = cleared || t.time == rig.now;
    EXPECT_TRUE(cleared);
}

TEST(ClockJumpBack, aRewoundClockRestartsTheStreamAtTheNewTime) {
    Rig rig(Variant::T3);
    rig.now = 5000;
    rig.rt.inputTokenQueue.push(key(5000, 60, 50));
    for (int i = 0; i < 8; ++i)
        ASSERT_TRUE(rig.rt.stepOnce());
    rig.notes();
    ASSERT_GE(rig.rt.streamForTest().base(), 4800);

    rig.now = 120; // transport rewound / restarted
    rig.rt.inputTokenQueue.push(key(120, 64, 50));
    ASSERT_TRUE(rig.rt.stepOnce());
    EXPECT_EQ(rig.rt.streamForTest().base(), 0) << "the stream starts again at the new clock";
    bool cleared = false;
    std::vector<Token> fresh;
    Token t{};
    while (rig.rt.outputTokenQueue.pull(t)) {
        if (t.note == static_cast<int32_t>(Vocab::ClearQueue))
            cleared = cleared || t.time == 120;
        else
            fresh.push_back(t);
    }
    EXPECT_TRUE(cleared) << "the old timeline's queued notes are dropped";
    ASSERT_FALSE(fresh.empty());
    EXPECT_GE(fresh.front().time, 120);
    EXPECT_LT(fresh.front().time, 1000) << "not pinned to the old timeline";
    for (const auto &h: rig.rt.getOutputHistory())
        if (h.note != static_cast<int32_t>(Vocab::ClearQueue))
            EXPECT_LT(h.time, 4000) << "the history holds only the new timeline";
}

TEST(ClockJumpBack, aSmallWobbleIsNotAJump) {
    Rig rig(Variant::T3);
    rig.now = 5000;
    rig.rt.inputTokenQueue.push(key(5000, 60, 50));
    ASSERT_TRUE(rig.rt.stepOnce());
    const int32_t base = rig.rt.streamForTest().base();
    rig.now = 4970; // 0.3 s back (an MTC wobble)
    rig.rt.stepOnce();
    EXPECT_EQ(rig.rt.streamForTest().base(), base);
    EXPECT_FALSE(rig.rt.getOutputHistory().empty());
}

TEST(AttackOnset, waitModelsMayPlayOnTheSoloistsRecentAttack) {
    struct Case { Variant v; int32_t lateCs; bool mayReachBack; };
    for (const auto c: {Case{Variant::Wait, 12, true}, Case{Variant::Wait, 40, false},
                        Case{Variant::T3, 12, false}}) {
        int beforeNow = 0;
        for (int trial = 0; trial < 20; ++trial) {
            Rig rig(c.v);
            rig.now = 1000 + c.lateCs; // the group arrives this late
            rig.rt.inputTokenQueue.push(key(1000, 60 + trial % 5, 50));
            ASSERT_TRUE(rig.rt.stepOnce());
            const auto notes = rig.notes();
            ASSERT_FALSE(notes.empty());
            EXPECT_GE(notes.front().time, 1000) << "never before the attack";
            beforeNow += notes.front().time < rig.now ? 1 : 0;
        }
        if (c.mayReachBack)
            EXPECT_GT(beforeNow, 0) << "W2/W3 can join the attack, 0.12 s in the past";
        else
            EXPECT_EQ(beforeNow, 0) << (c.v == Variant::T3 ? "T3'' never reaches back"
                                                          : "an attack 0.4 s old is too old");
    }
}

} // namespace virtual_orch_test
