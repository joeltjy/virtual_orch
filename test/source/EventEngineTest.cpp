#include <gtest/gtest.h>

#include <JuceHeader.h>

#include <algorithm>
#include <vector>

#include <VirtualOrch/reduction/DenseTypes.h>
#include <VirtualOrch/reduction/EventEngine.h>
#include <VirtualOrch/reduction/HybridEventReductionTransformer.h>
#include <VirtualOrch/reduction/StreamingKvCache.h>

/**
 * Hybrid v1.3, model 1 event (m1e): ON / OFF events at their times, the section-8 rule (a piano token after a drawn
 * time is a no-op; the orchestra is forced after a long hold), piano input before a pending draw, the max hold and
 * the context rebuild, against a scripted backend that sees the whole token sequence.
 */
namespace {

using Engine = EventEngine;
using namespace StreamingKv;

class EventScript : public Backend {
public:
    EventScript() {
        d.layers = 1;
        d.heads = 1;
        d.headDim = 1;
        d.maxRows = 100000; // no eviction: the script reads the whole sequence back from the cache
        d.sinkRows = 4;
    }
    [[nodiscard]] auto dims() const -> Dims override { return d; }

    int32_t off0 = Engine::kM1eOff0;
    int32_t vocab = Engine::kM1eVocab;
    int32_t gap = 30;          // the next time = the last event's time + gap
    int32_t pitch = 72;        // the orchestra note it plays (and releases)
    bool pianoMeant = false;   // after a time: a piano ON
    bool noOff = false;        // never release

    auto logitsAfter(const std::vector<int32_t> &seq) const -> std::vector<float> {
        std::vector<float> l(static_cast<size_t>(vocab), 0.0f);
        const auto set = [&](int32_t id, float v) { l[static_cast<size_t>(id)] = v; };
        const int32_t last = seq.empty() ? -1 : seq.back();
        int32_t lastTime = 0;
        bool open = false;
        for (size_t i = 0; i < seq.size(); ++i) {
            if (seq[i] <= Engine::kMaxTime && i + 1 < seq.size()) {
                if (seq[i + 1] >= Engine::kNote0 && seq[i + 1] < Engine::kVel0)
                    lastTime = seq[i];
                else if (seq[i + 1] >= off0)
                    lastTime = seq[i];
            }
            if (seq[i] == Engine::kOrch0 + pitch)
                open = true;
            if (seq[i] == off0 + 128 + pitch)
                open = false;
        }
        if (last >= 0 && last <= Engine::kMaxTime) {
            if (pianoMeant)
                set(Engine::kNote0 + 60, 20.0f);
            else if (noOff)
                for (int32_t p = 21; p <= 108; ++p)
                    set(Engine::kOrch0 + p, 10.0f);
            else
                set(open ? off0 + 128 + pitch : Engine::kOrch0 + pitch, 20.0f);
        } else if (last >= Engine::kNote0 && last < Engine::kVel0) {
            set(Engine::kVel0 + 80, 20.0f);
        } else {
            set(std::min(lastTime + gap, Engine::kMaxTime), 20.0f);
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

struct Rig {
    EventScript model;
    Engine engine;
    std::vector<std::pair<int32_t, Engine::Out>> outs; // (step time, out)
    explicit Rig(int32_t forceAfterCs = 200, int32_t maxHoldCs = 500) {
        engine.setBackend(&model);
        auto s = engine.settings();
        s.forceAfterCs = forceAfterCs;
        s.maxHoldCs = maxHoldCs;
        engine.setSettings(s);
        engine.reset(0);
        engine.pianoOn(0, 48, 70); // a context to draw from (an empty one has no logits)
    }
    auto run(int32_t from, int32_t to) -> void {
        for (int32_t now = from; now <= to; ++now)
            for (const auto &o: engine.step(now))
                outs.emplace_back(now, o);
    }
};

} // namespace

TEST(EventEngine, PlaysOnsAndOffsAtTheirTimes) {
    Rig rig;
    rig.run(0, 200);
    ASSERT_EQ(rig.outs.size(), 6u);
    for (size_t i = 0; i < rig.outs.size(); ++i) {
        const auto &[at, o] = rig.outs[i];
        EXPECT_EQ(o.timeCs, 30 * static_cast<int32_t>(i + 1));
        EXPECT_EQ(at, o.timeCs); // produced when the clock reaches it
        EXPECT_EQ(o.on, i % 2 == 0);
        EXPECT_EQ(o.pitch, 72);
        if (o.on)
            EXPECT_EQ(o.velocity, 80);
    }
    // the context: ON = 3 tokens, OFF = 2 (OFF0 + 128 instr + pitch)
    const auto toks = rig.engine.tokens();
    ASSERT_GE(toks.size(), 8u);
    EXPECT_EQ(toks[0], 0); // the piano ON
    EXPECT_EQ(toks[1], Engine::kNote0 + 48);
    EXPECT_EQ(toks[3], 30);
    EXPECT_EQ(toks[4], Engine::kOrch0 + 72);
    EXPECT_EQ(toks[5], Engine::kVel0 + 80);
    EXPECT_EQ(toks[6], 60);
    EXPECT_EQ(toks[7], Engine::kM1eOff0 + 128 + 72);
}

TEST(EventEngine, APianoTokenAfterATimeIsANoOpUntilTheOrchestraIsForced) {
    Rig rig(200);
    rig.model.pianoMeant = true;
    rig.run(0, 400);
    EXPECT_GT(rig.engine.stats().pianoMeant, 0);
    ASSERT_FALSE(rig.outs.empty());
    EXPECT_GE(rig.outs.front().second.timeCs, 200); // nothing before the forced event
    EXPECT_GE(rig.engine.stats().forced, 1);
    for (const auto &e: rig.engine.events())
        if (e.timeCs > 0)
            EXPECT_NE(e.kind, Engine::Kind::PianoOn); // a piano-meant step writes nothing
}

TEST(EventEngine, PianoInputBeforeAPendingDrawThrowsItAway) {
    Rig rig;
    rig.run(0, 5); // ON 72 drawn for 30, pending
    ASSERT_TRUE(rig.engine.hasPending());
    rig.engine.pianoOn(10, 60, 90);
    EXPECT_EQ(rig.engine.stats().discarded, 1);
    rig.run(10, 60);
    ASSERT_FALSE(rig.outs.empty());
    EXPECT_EQ(rig.outs.front().second.timeCs, 40); // drawn again after the piano event (10 + 30)
    const auto &ev = rig.engine.events();
    ASSERT_GE(ev.size(), 3u);
    EXPECT_EQ(ev[1].kind, Engine::Kind::PianoOn);
    EXPECT_EQ(ev[1].timeCs, 10);
    // key-up: an OFF event at once
    rig.engine.pianoOff(70, 60);
    EXPECT_EQ(rig.engine.events().back().kind, Engine::Kind::PianoOff);
    EXPECT_EQ(rig.engine.tokens().back(), Engine::kM1eOff0 + 60);
}

TEST(EventEngine, ANoteNeverReleasedIsReleasedAfterTheMaxHold) {
    Rig rig(200, 50);
    rig.model.noOff = true;
    rig.run(0, 200);
    EXPECT_GE(rig.engine.stats().forcedReleases, 1);
    std::vector<std::pair<int32_t, int32_t>> ons;
    for (const auto &[at, o]: rig.outs)
        if (o.on)
            ons.emplace_back(o.timeCs, o.pitch);
    for (const auto &entry: rig.outs) {
        const auto &rel = entry.second;
        if (! rel.on && rel.forcedRelease) {
            const int32_t p = rel.pitch;
            const auto it = std::find_if(ons.begin(), ons.end(), [p](const auto &n) { return n.second == p; });
            ASSERT_NE(it, ons.end());
            EXPECT_EQ(rel.timeCs, it->first + 50);
            break;
        }
    }
}

TEST(EventEngine, TheContextIsRebuiltBeforeTimesPass9999) {
    Rig rig;
    rig.run(0, 12000);
    EXPECT_GE(rig.engine.stats().rebuilds, 1);
    for (const auto t: rig.engine.tokens())
        EXPECT_GE(t, 0);
    const auto &ev = rig.engine.events();
    ASSERT_FALSE(ev.empty());
    EXPECT_LE(ev.back().timeCs - ev.front().timeCs, Engine::kMaxTime);
    EXPECT_GT(rig.outs.size(), 300u); // it kept playing across the rebuild
}

// Opt-in (IW_M1E=1): the real m1e export (continuation_events_m1e_r1_kv.onnx) on sample_01's piano for 6 s, then
// silence for 9 s (the spec-8 test setup); prints notes/s and the rule's counts.
TEST(EventEngine, RealM1eSmoke) {
    if (juce::SystemStats::getEnvironmentVariable("IW_M1E", "") != "1")
        GTEST_SKIP() << "set IW_M1E=1";
    const auto home = juce::File::getSpecialLocation(juce::File::userHomeDirectory);
    StreamingKv::OnnxBackend onnx;
    std::string err;
    ASSERT_TRUE(onnx.init(home.getChildFile("Documents/virtual_orch_models/continuation_events_m1e_r1_kv.onnx")
                              .getFullPathName().toRawUTF8(), err)) << err;
    ASSERT_EQ(onnx.vocabSize(), static_cast<size_t>(Engine::kM1eVocab));
    juce::FileInputStream in(home.getChildFile("orchestration/samples/maestro_fixtures_10/sample_01/input.mid"));
    juce::MidiFile mf;
    ASSERT_TRUE(in.openedOk() && mf.readFrom(in));
    mf.convertTimestampTicksToSeconds();
    struct Ev { int32_t t; bool on; int32_t p, v; };
    std::vector<Ev> piano;
    for (int tr = 0; tr < mf.getNumTracks(); ++tr)
        for (const auto *e: *mf.getTrack(tr)) {
            const auto t = static_cast<int32_t>(std::lround(e->message.getTimeStamp() * 100));
            if (t >= 600)
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
        s.seed = seed;
        engine.setSettings(s);
        engine.reset(0);
        size_t next = 0;
        int ons = 0, onsSilent = 0;
        std::vector<int32_t> held;
        std::vector<int32_t> aheads;
        for (int32_t now = 0; now < 1500; ++now) {
            while (next < piano.size() && piano[next].t <= now) {
                if (piano[next].on) {
                    engine.pianoOn(now, piano[next].p, piano[next].v);
                    held.push_back(piano[next].p);
                } else {
                    engine.pianoOff(now, piano[next].p);
                    held.erase(std::remove(held.begin(), held.end(), piano[next].p), held.end());
                }
                ++next;
            }
            if (now == 600) { // the pianist stops: every key still down comes up
                for (const auto p: held)
                    engine.pianoOff(now, p);
                held.clear();
            }
            for (const auto &o: engine.step(now))
                if (o.on) {
                    ++ons;
                    onsSilent += now >= 600 ? 1 : 0;
                }
        }
        const auto &stt = engine.stats();
        std::printf("[m1e] seed %llu: %d ONs (%.1f/s while the piano is silent), OFFs %d, piano-meant %d, forced %d, "
                    "forced releases %d, discarded %d, rebuilds %d %s\n",
                    static_cast<unsigned long long>(seed), ons, onsSilent / 9.0, stt.offs, stt.pianoMeant, stt.forced,
                    stt.forcedReleases, stt.discarded, stt.rebuilds, engine.lastError.c_str());
        EXPECT_GT(ons, 0);
    }
}

// The v1.3 transformer: the piano's key-down / key-up reach both engines as ON / OFF events; model 1's ON goes out
// with the provisional length and its OFF sends the real length downstream (a TokenUpdate); after 1 s of playing
// model 2 takes over (after its start delay); silent 0.5 s: back to model 1.
TEST(HybridEvent, OnsGoOutAtOnceAndOffsSendTheRealLength) {
    ModelConfig config;
    HybridEventReductionTransformer hy(config);
    auto silent = std::make_unique<EventScript>();
    auto wait = std::make_unique<EventScript>();
    wait->off0 = HybridEventReductionTransformer::kM2eOff0;
    wait->vocab = HybridEventReductionTransformer::kM2eVocab;
    wait->pitch = 76;
    hy.initWithBackends(std::move(silent), std::move(wait));
    CircularFifo<TokenUpdate> updates;
    hy.orchestrationUpdatesIncoming = &updates;
    hy.setStartDelayMs(0);
    int32_t now = 0;
    int64_t wall = 0;
    int64_t silentFor = INT64_MAX;
    hy.nowCsProvider = [&] { return now; };
    hy.wallMsProvider = [&] { return wall; };
    hy.pianoSilentForMs = [&] { return silentFor; };
    const auto runFor = [&](int cs) {
        for (int i = 0; i < cs; ++i) {
            hy.stepOnce();
            now += 1;
            wall += 10;
        }
    };
    // a piano note (key-down and key-up known): ON now, OFF at 20
    Token t{0, static_cast<int32_t>(DenseVocab::DurOffset) + 20, static_cast<int32_t>(DenseVocab::NoteOffset) + 48};
    t.velocity = 70;
    hy.inputTokenQueue.push(t);
    runFor(70);
    const auto &ev = hy.silentEngineForTest().events();
    ASSERT_GE(ev.size(), 2u);
    EXPECT_EQ(ev[0].kind, Engine::Kind::PianoOn);
    EXPECT_TRUE(std::any_of(ev.begin(), ev.end(), [](const auto &e) { return e.kind == Engine::Kind::PianoOff && e.timeCs == 20; }));
    // model 1: ON 72 at 50 (20 + 30), OFF at 80
    auto out = hy.getOutputHistory();
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].time, 50);
    EXPECT_EQ(out[0].getPitch(), 72);
    EXPECT_EQ(out[0].getRealDuration(), ReductionTransformer::kMaxNoteDurationCs); // provisional
    runFor(20);
    TokenUpdate u{};
    ASSERT_TRUE(updates.pull(u));
    EXPECT_EQ(u.oldNote.time, 50);
    EXPECT_EQ(u.newNote.getRealDuration(), 30);
    out = hy.getOutputHistory();
    EXPECT_EQ(out[0].getRealDuration(), 30);
    // model 2's context has model 1's notes (mirrored)
    EXPECT_TRUE(std::any_of(hy.waitEngineForTest().events().begin(), hy.waitEngineForTest().events().end(),
                            [](const auto &e) { return e.kind == Engine::Kind::OrchOn && e.pitch == 72; }));
    // the pianist plays for 1.2 s: model 2 (its own pitch, 76) takes over
    silentFor = 0;
    runFor(120);
    EXPECT_FALSE(hy.usesSilentModel());
    runFor(100);
    out = hy.getOutputHistory();
    EXPECT_TRUE(std::any_of(out.begin(), out.end(), [](const Token &k) { return k.getPitch() == 76; }));
    // silent 0.6 s: back to model 1
    silentFor = 600;
    runFor(1);
    EXPECT_TRUE(hy.usesSilentModel());
    EXPECT_EQ(hy.reason(), HybridEventReductionTransformer::Reason::Stopped);
}

TEST(HybridEvent, ModelTwoDrawsNothingDuringTheStartDelay) {
    ModelConfig config;
    HybridEventReductionTransformer hy(config);
    auto wait = std::make_unique<EventScript>();
    wait->off0 = HybridEventReductionTransformer::kM2eOff0;
    wait->vocab = HybridEventReductionTransformer::kM2eVocab;
    hy.initWithBackends(std::make_unique<EventScript>(), std::move(wait));
    hy.setStartDelayMs(3000);
    int32_t now = 0;
    int64_t wall = 0;
    hy.nowCsProvider = [&] { return now; };
    hy.wallMsProvider = [&] { return wall; };
    hy.pianoSilentForMs = [] { return int64_t{0}; }; // playing all along
    for (int i = 0; i < 400; ++i) {
        if (i % 50 == 0) {
            Token t{now, static_cast<int32_t>(DenseVocab::DurOffset) + 20, static_cast<int32_t>(DenseVocab::NoteOffset) + 60};
            t.velocity = 80;
            hy.inputTokenQueue.push(t);
        }
        hy.stepOnce();
        if (now == 299)
            EXPECT_FALSE(hy.usesSilentModel());
        ++now;
        wall += 10;
    }
    for (const auto &k: hy.getOutputHistory())
        if (k.time >= 100) // model 1 played the first second; model 2 nothing before 300
            EXPECT_GE(k.time, 300);
}

TEST(HybridEvent, LoadsTheRealExports) {
    const auto dir = juce::File::getSpecialLocation(juce::File::userHomeDirectory).getChildFile("Documents/virtual_orch_models");
    const auto m1 = dir.getChildFile("continuation_events_m1e_r1_kv.onnx");
    const auto m2 = dir.getChildFile("simultaneous_events_chordtokens_m2e_r1_kv.onnx");
    if (! m1.existsAsFile() || ! m2.existsAsFile())
        GTEST_SKIP() << "v1.3 exports not in virtual_orch_models";
    ModelConfig config;
    HybridEventReductionTransformer hy(config);
    EXPECT_TRUE(hy.init(m1.getFullPathName().toRawUTF8(), m2.getFullPathName().toRawUTF8())) << hy.lastInitError;
    EXPECT_TRUE(hy.isModelLoaded());
    // swapped: the vocab check refuses them
    HybridEventReductionTransformer bad(config);
    EXPECT_FALSE(bad.init(m2.getFullPathName().toRawUTF8(), m1.getFullPathName().toRawUTF8()));
}
