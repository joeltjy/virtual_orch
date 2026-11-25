#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include <VirtualOrch/reduction/EventEngine.h>
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

    int32_t gap = 30;          // the next time = the last event's time + gap
    int32_t pitch = 72;        // the orchestra note it plays (and releases)
    bool pianoMeant = false;   // after a time: a piano ON
    bool noOff = false;        // never release

    auto logitsAfter(const std::vector<int32_t> &seq) const -> std::vector<float> {
        std::vector<float> l(Engine::kM1eVocab, 0.0f);
        const auto set = [&](int32_t id, float v) { l[static_cast<size_t>(id)] = v; };
        const int32_t last = seq.empty() ? -1 : seq.back();
        int32_t lastTime = 0;
        bool open = false;
        for (size_t i = 0; i < seq.size(); ++i) {
            if (seq[i] <= Engine::kMaxTime && i + 1 < seq.size()) {
                if (seq[i + 1] >= Engine::kNote0 && seq[i + 1] < Engine::kVel0)
                    lastTime = seq[i];
                else if (seq[i + 1] >= Engine::kM1eOff0)
                    lastTime = seq[i];
            }
            if (seq[i] == Engine::kOrch0 + pitch)
                open = true;
            if (seq[i] == Engine::kM1eOff0 + 128 + pitch)
                open = false;
        }
        if (last >= 0 && last <= Engine::kMaxTime) {
            if (pianoMeant)
                set(Engine::kNote0 + 60, 20.0f);
            else if (noOff)
                for (int32_t p = 21; p <= 108; ++p)
                    set(Engine::kOrch0 + p, 10.0f);
            else
                set(open ? Engine::kM1eOff0 + 128 + pitch : Engine::kOrch0 + pitch, 20.0f);
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
