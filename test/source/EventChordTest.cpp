#include <gtest/gtest.h>

#include <JuceHeader.h>

#include <algorithm>
#include <vector>

#include <VirtualOrch/reduction/EventEngine.h>
#include <VirtualOrch/reduction/HybridEventReductionTransformer.h>
#include <VirtualOrch/reduction/StreamingKvCache.h>
#include <VirtualOrch/reduction/VoicingBank.h>

/**
 * Hybrid v1.3, model 2 event (m2e, EventEngine::Settings::chordTokens; events2.py, spec section 4): [WAIT d][K k,+1]
 * where a time would be drawn, the fire (the onset's piano ONs, then [CHORD][BASS][SVEL] and the band notes), the
 * leftover ON, STACK_OFF, [LATE d] / [EXPIRED] at a close and the give-up, against a scripted model.
 */
namespace {

using Engine = EventEngine;
using namespace StreamingKv;
constexpr int32_t kOff0 = HybridEventReductionTransformer::kM2eOff0;
constexpr int32_t kStackOff = HybridEventReductionTransformer::kM2eStackOff;
constexpr int32_t kCMaj3 = Engine::kChord0 + (7 * 12 + 0) * 10 + 2; // maj, root C, 3 notes
constexpr int32_t kWait30 = Engine::kWait0 + 30 / 3 - 1;

class ChordEventScript : public Backend {
public:
    ChordEventScript() {
        d.layers = 1;
        d.heads = 1;
        d.headDim = 1;
        d.maxRows = 100000;
        d.sinkRows = 4;
    }
    [[nodiscard]] auto dims() const -> Dims override { return d; }

    bool waits = true;            // after an event: [WAIT 30] (else a time 30 cs later)
    int32_t kTok = Engine::kK0 + 4; // k = 3, no +1
    bool expired = false;         // at a close: EXPIRED (else LATE 9)
    bool expiredThenWait = false; // after EXPIRED: a re-wait (else CHORD: give up)
    bool stackOff = false;        // after a time: STACK_OFF (else ON 72)

    auto logitsAfter(const std::vector<int32_t> &seq) const -> std::vector<float> {
        std::vector<float> l(HybridEventReductionTransformer::kM2eVocab, 0.0f);
        const auto set = [&](int32_t id, float v) { l[static_cast<size_t>(id)] = v; };
        const int32_t last = seq.empty() ? -1 : seq.back();
        int32_t lastTime = 0;
        for (size_t i = 0; i + 1 < seq.size(); ++i)
            if (seq[i] <= Engine::kMaxTime && (seq[i + 1] >= Engine::kNote0 && seq[i + 1] < Engine::kVel0 || seq[i + 1] >= kOff0))
                lastTime = seq[i];
        if (last >= 0 && last <= Engine::kMaxTime) {
            set(stackOff ? kStackOff : Engine::kOrch0 + 72, 20.0f);
        } else if (last >= Engine::kWait0 && last < Engine::kK0) {
            set(kTok, 20.0f);
        } else if (last >= Engine::kChord0 && last < Engine::kChordEnd) {
            set(Engine::kOrch0 + 48, 20.0f); // BASS C3
        } else if (last >= Engine::kNote0 && last < Engine::kVel0) {
            set(Engine::kVel0 + 80, 20.0f);
        } else if (last == Engine::kExpired) {
            set(expiredThenWait ? kWait30 : kCMaj3, 20.0f);
        } else {
            // after an event / the block / K / LATE: the next step; the masks pick the field the engine wants
            set(waits ? kWait30 : std::min(lastTime + 30, Engine::kMaxTime), 20.0f);
            set(kCMaj3, 20.0f);
            set(expired ? Engine::kExpired : Engine::kLate0 + 2, 20.0f);
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
        std::vector<int32_t> seq;
        for (const float k: present.key)
            seq.push_back(static_cast<int32_t>(k));
        out = logitsAfter(seq);
        return true;
    }

private:
    Dims d;
};

struct ChordRig {
    ChordEventScript model;
    Engine engine;
    std::vector<Engine::Out> outs;
    ChordRig() {
        engine.setBackend(&model);
        auto s = engine.settings();
        s.off0 = kOff0;
        s.vocab = HybridEventReductionTransformer::kM2eVocab;
        s.chordTokens = true;
        s.stackOff = kStackOff;
        s.forceAfterCs = 1 << 20;
        s.expiryGraceCs = 0;
        engine.setSettings(s);
        engine.reset(0);
        engine.pianoOn(0, 36, 70);
    }
    auto run(int32_t from, int32_t to) -> void {
        for (int32_t now = from; now <= to; ++now) {
            const auto o = engine.step(now);
            outs.insert(outs.end(), o.begin(), o.end());
        }
    }
    [[nodiscard]] auto has(const std::vector<int32_t> &needle) const -> bool {
        const auto t = engine.tokens();
        return std::search(t.begin(), t.end(), needle.begin(), needle.end()) != t.end();
    }
};

} // namespace

TEST(EventChord, AFireWritesThePianoOnsThenTheBlockAndPlaysTheBand) {
    ChordRig rig;
    rig.run(0, 20);
    ASSERT_TRUE(rig.engine.hasPendingWait()); // [WAIT 30][K 3] from the piano ON at 0: target 30
    EXPECT_TRUE(rig.has({kWait30, Engine::kK0 + 4}));
    rig.engine.pianoOn(28, 60, 90); // within 6 cs before the target: fires
    rig.engine.pianoOn(31, 64, 90); // the same onset (within 5 cs): written at 28, before the block
    rig.run(21, 40);
    EXPECT_EQ(rig.engine.stats().fires, 1);
    EXPECT_TRUE(rig.has({28, Engine::kNote0 + 60, Engine::kVel0 + 90, 28, Engine::kNote0 + 64, Engine::kVel0 + 90, kCMaj3,
                         Engine::kOrch0 + 48, Engine::kVel0 + 80}));
    std::vector<int32_t> band;
    for (const auto &o: rig.outs)
        if (o.on && o.chord && o.timeCs == 28)
            band.push_back(o.pitch);
    ASSERT_FALSE(band.empty());
    EXPECT_EQ(band.front(), 48); // the bass
    for (const auto p: band)
        EXPECT_TRUE(p % 12 == 0 || p % 12 == 4 || p % 12 == 7) << p; // C major tones
}

TEST(EventChord, StackOffReleasesTheBandNotesTogether) {
    ChordRig rig;
    rig.model.stackOff = true; // after a time: STACK_OFF (allowed once a block sounds; orchestra OFFs pass a WAIT)
    rig.run(0, 20);
    rig.engine.pianoOn(28, 60, 90);
    rig.run(21, 120);
    EXPECT_GE(rig.engine.stats().stackOffs, 1);
    std::vector<int32_t> band;
    for (const auto &o: rig.outs)
        if (o.on && o.chord && o.timeCs == 28)
            band.push_back(o.pitch);
    ASSERT_FALSE(band.empty());
    std::vector<int32_t> released;
    int32_t at = -1;
    for (const auto &o: rig.outs)
        if (! o.on && o.chord && (at < 0 || o.timeCs == at)) {
            released.push_back(o.pitch);
            at = o.timeCs; // the first release: one event for the whole band
        }
    std::sort(band.begin(), band.end());
    std::sort(released.begin(), released.end());
    EXPECT_EQ(released, band);
    EXPECT_GT(at, 28);
    EXPECT_TRUE(rig.has({at, kStackOff}));
}

TEST(EventChord, AtACloseTheModelDrawsLateThenThePianoFires) {
    ChordRig rig;
    rig.run(0, 36); // close = 0 + 30 + 6 = 36
    ASSERT_TRUE(rig.engine.hasPendingWait());
    rig.run(37, 38); // the close: LATE 9 (model), the window now closes at 36 + 9 + 6
    EXPECT_EQ(rig.engine.stats().lates, 1);
    EXPECT_TRUE(rig.has({Engine::kLate0 + 2}));
    rig.engine.pianoOn(44, 60, 90);
    rig.run(39, 60);
    EXPECT_EQ(rig.engine.stats().fires, 1);
}

TEST(EventChord, ExpiredThenAChordGivesUpAndTheBlockPlaysAlone) {
    ChordRig rig;
    rig.model.expired = true;
    rig.run(0, 40);
    EXPECT_EQ(rig.engine.stats().expired, 1);
    EXPECT_EQ(rig.engine.stats().giveUps, 1);
    EXPECT_TRUE(rig.has({Engine::kExpired, kCMaj3, Engine::kOrch0 + 48}));
    ASSERT_FALSE(rig.outs.empty());
    EXPECT_TRUE(rig.outs.front().chord);
    EXPECT_EQ(rig.outs.front().timeCs, 36); // at the close
}

TEST(EventChord, ExpiredThenAWaitReWaitsWithTheSameK) {
    ChordRig rig;
    rig.model.expired = true;
    rig.model.expiredThenWait = true;
    rig.run(0, 40);
    EXPECT_EQ(rig.engine.stats().reWaits, 1);
    EXPECT_TRUE(rig.engine.hasPendingWait());
    EXPECT_TRUE(rig.has({Engine::kExpired, kWait30, Engine::kK0 + 4}));
}

TEST(EventChord, TheLeftoverOnFollowsTheBlockAtTheOnset) {
    ChordRig rig;
    rig.model.kTok = Engine::kK0 + 5; // k = 3, +1
    rig.run(0, 20);
    rig.engine.pianoOn(29, 60, 90);
    rig.run(21, 40);
    EXPECT_TRUE(rig.has({kCMaj3, Engine::kOrch0 + 48, Engine::kVel0 + 80, 29, Engine::kOrch0 + 72, Engine::kVel0 + 80}));
    EXPECT_TRUE(std::any_of(rig.outs.begin(), rig.outs.end(), [](const auto &o) { return o.on && o.pitch == 72 && o.timeCs == 29; }));
}

// Opt-in (IW_M2E=1): the real m2e export with the voicing bank on sample_01's piano (15 s, playing throughout);
// prints the WAIT family's counts and the orchestra's notes a second.
TEST(EventChord, RealM2eSmoke) {
    if (juce::SystemStats::getEnvironmentVariable("IW_M2E", "") != "1")
        GTEST_SKIP() << "set IW_M2E=1";
    const auto home = juce::File::getSpecialLocation(juce::File::userHomeDirectory);
    const auto models = home.getChildFile("Documents/virtual_orch_models");
    StreamingKv::OnnxBackend onnx;
    std::string err;
    ASSERT_TRUE(onnx.init(models.getChildFile("v1.3_simultaneous.onnx").getFullPathName().toRawUTF8(), err)) << err;
    auto bank = std::make_shared<VoicingBank>();
    ASSERT_TRUE(bank->loadJson(models.getChildFile("voicing_bank.json").loadFileAsString().toStdString(), err)) << err;
    juce::FileInputStream in(home.getChildFile("orchestration/samples/maestro_fixtures_10/sample_01/input.mid"));
    juce::MidiFile mf;
    ASSERT_TRUE(in.openedOk() && mf.readFrom(in));
    mf.convertTimestampTicksToSeconds();
    struct Ev { int32_t t; bool on; int32_t p, v; };
    std::vector<Ev> piano;
    for (int tr = 0; tr < mf.getNumTracks(); ++tr)
        for (const auto *e: *mf.getTrack(tr)) {
            const auto t = static_cast<int32_t>(std::lround(e->message.getTimeStamp() * 100));
            if (t >= 1500)
                continue;
            if (e->message.isNoteOn())
                piano.push_back({t, true, e->message.getNoteNumber(), e->message.getVelocity()});
            else if (e->message.isNoteOff())
                piano.push_back({t, false, e->message.getNoteNumber(), 0});
        }
    std::stable_sort(piano.begin(), piano.end(), [](const Ev &a, const Ev &b) { return a.t < b.t; });
    for (const uint64_t seed: {1ull, 2ull}) {
        Engine engine;
        engine.setBackend(&onnx);
        auto s = engine.settings();
        s.off0 = kOff0;
        s.vocab = HybridEventReductionTransformer::kM2eVocab;
        s.chordTokens = true;
        s.stackOff = kStackOff;
        s.voicings = bank;
        s.seed = seed;
        engine.setSettings(s);
        engine.reset(0);
        size_t next = 0;
        int ons = 0, chordOns = 0;
        for (int32_t now = 0; now < 1500; ++now) {
            while (next < piano.size() && piano[next].t <= now) {
                if (piano[next].on)
                    engine.pianoOn(now, piano[next].p, piano[next].v);
                else
                    engine.pianoOff(now, piano[next].p);
                ++next;
            }
            for (const auto &o: engine.step(now))
                if (o.on) {
                    ++ons;
                    chordOns += o.chord ? 1 : 0;
                }
        }
        const auto &stt = engine.stats();
        std::printf("[m2e] seed %llu: %d ONs (%d in chord blocks; %.1f/s), WAITs %d, fires %d, LATE %d (forced %d), "
                    "EXPIRED %d, re-waits %d, give-ups %d, STACK_OFF %d, voicing fallbacks %d, piano-meant %d, forced %d, "
                    "forced releases %d %s\n",
                    static_cast<unsigned long long>(seed), ons, chordOns, ons / 15.0, stt.waits, stt.fires, stt.lates,
                    stt.latesForced, stt.expired, stt.reWaits, stt.giveUps, stt.stackOffs, stt.voicingFallbacks,
                    stt.pianoMeant, stt.forced, stt.forcedReleases, engine.lastError.c_str());
        EXPECT_TRUE(engine.lastError.empty());
    }
}
