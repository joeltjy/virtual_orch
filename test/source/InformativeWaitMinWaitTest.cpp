#include <gtest/gtest.h>

#include "VirtualOrch/reduction/InformativeWaitEngine.h"
#include "VirtualOrch/reduction/StreamingKvCache.h"

#include <vector>

namespace {

// A model that wants a short WAIT (150 ms) far more than anything else at the onset step.
class ShortWaitModel final : public StreamingKv::Backend {
public:
    [[nodiscard]] auto dims() const -> StreamingKv::Dims override {
        StreamingKv::Dims d;
        d.layers = 1;
        d.heads = 1;
        d.headDim = 1;
        return d;
    }
    auto run(const StreamingKv::State &past, const std::vector<int64_t> &ids, StreamingKv::State &present,
             std::vector<float> &logits) -> bool override {
        present.rows = past.rows + static_cast<int>(ids.size());
        present.key = past.key;
        present.key.resize(static_cast<size_t>(present.rows), 0.0f);
        present.value = present.key;
        logits.assign(InformativeWaitEngine::kVocab, 0.0f);
        for (int32_t t = InformativeWaitEngine::kWait0; t < InformativeWaitEngine::kWait0 + 100; ++t)
            logits[static_cast<size_t>(t)] = 4.0f;
        logits[InformativeWaitEngine::kWait0 + 4] = 20.0f; // [WAIT 15]: 150 ms, under the 250 ms minimum
        return true;
    }
};

struct Drawn {
    std::vector<int32_t> waits;
    int replaced = 0;
    int orchAfterShort = 0; // a replaced short WAIT followed by an orchestra note (the redraw)
    int waitAfterShort = 0; // ... by a WAIT (must not happen)
    int atLoggedOnset = 0;  // the orchestra note at the onset the "short" event gave
    int atPredicted = 0;    // ... which is the WAIT's predicted onset (the event before it + d)
};

auto drawnWaits(int32_t minWaitCs, bool redraw = false) -> Drawn {
    ShortWaitModel model;
    InformativeWaitEngine engine;
    engine.setBackend(&model);
    auto s = engine.settings();
    s.mode = InformativeWaitEngine::Mode::Unconstrained;
    s.precompute = false;
    s.noWaitAfterExpiry = false;
    s.minWaitCs = minWaitCs;
    s.shortWaitRedraw = redraw;
    s.seed = 3;
    engine.setSettings(s);
    Drawn drawn;
    bool afterShort = false;
    InformativeWaitEngine::Event shortEvent;
    engine.onEvent = [&](const InformativeWaitEngine::Event &e) {
        if (e.what == "short") {
            afterShort = true;
            shortEvent = e;
        } else if (e.what == "orch" && afterShort) {
            ++drawn.orchAfterShort;
            drawn.atLoggedOnset += e.atCs == shortEvent.b ? 1 : 0;
            drawn.atPredicted += e.atCs == shortEvent.atCs + shortEvent.a ? 1 : 0;
            afterShort = false;
        } else if (e.what == "wait") {
            drawn.waits.push_back(e.a);
            drawn.waitAfterShort += afterShort ? 1 : 0;
            afterShort = false;
        }
    };
    engine.reset(0);
    // a note every 50 cs; the engine steps every cs in between
    for (int32_t now = 0; now < 2000; ++now) {
        if (now % 50 == 10) {
            engine.noteOn(now, 60 + (now / 50) % 12, 64);
            engine.noteOff(now, 60 + (now / 50) % 12, 30);
        }
        (void) engine.step(now);
    }
    drawn.replaced = engine.stats().shortWaitsReplaced;
    return drawn;
}

} // namespace

TEST(InformativeWaitMinWait, aShortWaitIsReplacedByAnOrchestraNote) {
    const auto free = drawnWaits(0);
    ASSERT_FALSE(free.waits.empty());
    int shortOnes = 0;
    for (const int32_t d: free.waits)
        shortOnes += d < 25 ? 1 : 0;
    EXPECT_GT(shortOnes, 0) << "without the minimum this model draws short WAITs";
    EXPECT_EQ(free.replaced, 0);

    // with it: no short WAIT; each one drawn is replaced at once by an orchestra note, never by a WAIT
    const auto banned = drawnWaits(25);
    for (const int32_t d: banned.waits)
        EXPECT_GE(d, 25);
    EXPECT_GT(banned.replaced, 0);
    EXPECT_EQ(banned.orchAfterShort, banned.replaced);
    EXPECT_EQ(banned.waitAfterShort, 0);
    // ... at the onset the WAIT predicted (or the nearest allowed one)
    EXPECT_EQ(banned.atLoggedOnset, banned.replaced);
    EXPECT_GT(banned.atPredicted, 0);

    // the redraw variant: also an orchestra note, never a WAIT
    const auto redrawn = drawnWaits(25, true);
    EXPECT_GT(redrawn.replaced, 0);
    EXPECT_EQ(redrawn.orchAfterShort, redrawn.replaced);
    EXPECT_EQ(redrawn.waitAfterShort, 0);
}
