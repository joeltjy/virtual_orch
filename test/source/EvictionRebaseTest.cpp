#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include <VirtualOrch/reduction/AnticipatedStream.h>
#include <VirtualOrch/reduction/MelodyStream.h>
#include <VirtualOrch/reduction/StreamingKvCache.h>

namespace {

using namespace StreamingKv;

/** One layer / head / dim; a cache row is the token id; the logits are a function of all rows. */
class FakeBackend : public Backend {
public:
    explicit FakeBackend(int maxRows) {
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
        double h = 0;
        for (const float k: present.key)
            h = h * 31.0 + k + 1.0;
        last = {static_cast<float>(std::fmod(h, 1.0e6)), static_cast<float>(present.rows)};
        return true;
    }
    Dims d;
};

struct Quad {
    size_t tokenStart;
    int32_t onsetAbs;
};

/** The notes of a token list (TD ids >= 11768 are single tokens), with absolute onsets. */
auto decode(const std::vector<int32_t> &tokens, int32_t base) -> std::vector<Quad> {
    std::vector<Quad> out;
    for (size_t i = 0; i < tokens.size();) {
        if (tokens[i] >= 11768) {
            ++i;
            continue;
        }
        out.push_back({i, tokens[i] + base});
        i += 4;
    }
    return out;
}

struct Harness {
    AnticipatedStream stream;
    FakeBackend backend{24};
    CommittedScratchCache cache{backend, 64};
    int rebases = 0;

    Harness() {
        stream.tdTokenProvider = [] { return 11772; };
        stream.reset(0);
    }

    /** Append one event, run the cache on the stream, then apply the eviction rule. */
    auto step(int32_t onset, int32_t *shiftOut = nullptr) -> void {
        stream.appendEvent({onset, 50, 60 + (onset / 100) % 12, 64});
        (void) cache.logitsAfter(stream.tokens(), stream.committedLen());
        const int32_t k = stream.rebaseForEviction(cache.evictedTokens(), cache.sinkRows());
        if (shiftOut != nullptr)
            *shiftOut = k;
        if (k > 0) {
            ++rebases;
            cache.reset(); // what the transformers do; the prefix would mismatch anyway
        }
    }

    /** Largest absolute onset among the notes the cache has evicted (decoded independently). */
    auto maxEvictedOnset() const -> int32_t {
        const auto notes = decode(stream.tokens(), stream.base());
        const size_t evictedBegin = cache.sinkRows();
        const size_t evictedEnd = cache.sinkRows() + cache.evictedTokens();
        int32_t best = INT32_MIN;
        for (const auto &q: notes)
            if (q.tokenStart < evictedEnd && q.tokenStart + 4 > evictedBegin)
                best = std::max(best, q.onsetAbs);
        return best;
    }
};

} // namespace

namespace virtual_orch_test {

TEST(EvictionRebase, rebasesExactlyWhenANoteOlderThanThirtySecondsIsEvicted) {
    Harness h;
    bool triggered = false;
    for (int32_t onset = 0; onset <= 12000 && ! triggered; onset += 400) {
        const int32_t baseBefore = h.stream.base();
        const int32_t evictedOnsetBefore = h.maxEvictedOnset();
        int32_t k = 0;
        h.stream.appendEvent({onset, 50, 60 + (onset / 100) % 12, 64});
        (void) h.cache.logitsAfter(h.stream.tokens(), h.stream.committedLen());
        const int32_t evictedOnset = h.maxEvictedOnset();
        k = h.stream.rebaseForEviction(h.cache.evictedTokens(), h.cache.sinkRows());
        (void) evictedOnsetBefore;
        if (evictedOnset != INT32_MIN && evictedOnset - baseBefore > 3000) {
            EXPECT_GT(k, 0) << "a note evicted at onset " << evictedOnset << " must rebase";
            EXPECT_EQ(k % 3000, 0);
            triggered = true;
        } else {
            EXPECT_EQ(k, 0) << "no note older than 30 s was evicted (onset " << onset << ")";
        }
    }
    EXPECT_TRUE(triggered);
}

TEST(EvictionRebase, shiftIsTheLargestMultipleOfThirtySecondsKeepingRetainedNotesNonNegative) {
    Harness h;
    int32_t shift = 0;
    for (int32_t onset = 0; onset <= 12000 && shift == 0; onset += 400) {
        const int32_t baseBefore = h.stream.base();
        const auto before = decode(h.stream.tokens(), baseBefore);
        h.step(onset, &shift);
        if (shift == 0)
            continue;
        const auto after = decode(h.stream.tokens(), h.stream.base());
        ASSERT_FALSE(after.empty());
        const int32_t oldestRetained = after.front().onsetAbs;
        EXPECT_GE(oldestRetained - h.stream.base(), 0) << "no negative times";
        EXPECT_LT(oldestRetained - h.stream.base(), 3000) << "k is the LARGEST such multiple";
        EXPECT_EQ(h.stream.base(), baseBefore + shift);
        for (const auto id: h.stream.tokens())
            EXPECT_GE(id, 0);
        (void) before;
    }
    EXPECT_GT(shift, 0);
}

TEST(EvictionRebase, outputTimesStayContinuousAcrossARebase) {
    Harness h;
    std::vector<int32_t> absoluteOnsets;
    int32_t shift = 0;
    int32_t last = 0;
    for (int32_t onset = 0; onset <= 20000; onset += 400) {
        h.step(onset, &shift);
        absoluteOnsets.push_back(onset);
        // every note still in the stream keeps its absolute onset: time token + origin
        for (const auto &q: decode(h.stream.tokens(), h.stream.base())) {
            EXPECT_TRUE(std::find(absoluteOnsets.begin(), absoluteOnsets.end(), q.onsetAbs)
                        != absoluteOnsets.end())
                << "onset " << q.onsetAbs << " was never written";
            EXPECT_GE(q.onsetAbs, last - 20000);
        }
        EXPECT_EQ(h.stream.clock(), onset) << "the event clock is absolute";
        last = onset;
    }
    EXPECT_GE(h.rebases, 2) << "20 s of events at 0.4 s spacing and a 24-row cache rebases repeatedly";
    // the time tokens the model sees stay small even after minutes of absolute time
    for (const auto &q: decode(h.stream.tokens(), h.stream.base()))
        EXPECT_LT(q.onsetAbs - h.stream.base(), 9999);
}

TEST(EvictionRebase, retainedTokensAreTheOldOnesWithTheirTimesShiftedByK) {
    Harness h;
    int32_t shift = 0;
    std::vector<int32_t> before;
    size_t evictedEnd = 0;
    for (int32_t onset = 0; onset <= 12000 && shift == 0; onset += 400) {
        before = h.stream.tokens();
        h.stream.appendEvent({onset, 50, 60 + (onset / 100) % 12, 64});
        (void) h.cache.logitsAfter(h.stream.tokens(), h.stream.committedLen());
        before = h.stream.tokens();
        evictedEnd = h.cache.sinkRows() + h.cache.evictedTokens();
        shift = h.stream.rebaseForEviction(h.cache.evictedTokens(), h.cache.sinkRows());
    }
    ASSERT_GT(shift, 0);
    // expected window: skip the sinks and the evicted tokens (and leading TD tokens), shift times
    std::vector<int32_t> expected;
    size_t i = evictedEnd;
    while (i < before.size() && before[i] >= 11768)
        ++i;
    // the evicted range may end inside a quad: the retained window starts at the next quad
    // boundary of the original layout
    const auto notes = decode(before, 0);
    size_t start = before.size();
    for (const auto &q: notes)
        if (q.tokenStart >= evictedEnd) { start = q.tokenStart; break; }
    for (size_t t = start; t < before.size();) {
        if (before[t] >= 11768) { expected.push_back(before[t]); ++t; continue; }
        expected.push_back(before[t] - shift);
        expected.insert(expected.end(), before.begin() + static_cast<std::ptrdiff_t>(t) + 1,
                        before.begin() + static_cast<std::ptrdiff_t>(t) + 4);
        t += 4;
    }
    // leading TD tokens are dropped so the window starts on a quad
    while (! expected.empty() && expected.front() >= 11768)
        expected.erase(expected.begin());
    EXPECT_EQ(h.stream.tokens(), expected);
}

TEST(EvictionRebase, logitsAfterTheRebuildMatchAFreshPrefillOfTheShiftedWindow) {
    Harness h;
    int32_t shift = 0;
    for (int32_t onset = 0; onset <= 12000 && shift == 0; onset += 400)
        h.step(onset, &shift);
    ASSERT_GT(shift, 0);
    const auto window = h.stream.tokens();
    const auto &rebuilt = h.cache.logitsAfter(window, h.stream.committedLen());
    ASSERT_FALSE(rebuilt.empty());
    const auto rebuiltCopy = rebuilt;

    FakeBackend freshBackend(24);
    CommittedScratchCache fresh(freshBackend, 64);
    const auto &expected = fresh.logitsAfter(window, window.size());
    ASSERT_FALSE(expected.empty());
    EXPECT_EQ(rebuiltCopy, expected);
    EXPECT_EQ(h.cache.committedRows(), fresh.committedRows());
    EXPECT_EQ(h.cache.evictedTokens(), 0u) << "the rebuilt cache starts without evictions";
}

TEST(EvictionRebase, noRebaseWhileTheEvictedNotesAreYoungerThanThirtySeconds) {
    Harness h;
    for (int32_t onset = 0; onset <= 2400; onset += 200) {
        int32_t k = 0;
        h.step(onset, &k);
        EXPECT_EQ(k, 0);
    }
    EXPECT_EQ(h.rebases, 0);
}

TEST(EvictionRebase, pendingControlsOlderThanTheNewOriginAreDroppedFutureOnesKeepTheirAbsoluteTime) {
    Harness h;
    h.stream.addControl({100, 50, 70, 64}, false); // old: will fall before the new origin
    h.stream.addControl({99000, 50, 71, 64}, false); // far in the future
    int32_t shift = 0;
    for (int32_t onset = 0; onset <= 12000 && shift == 0; onset += 400)
        h.step(onset, &shift);
    ASSERT_GT(shift, 0);
    EXPECT_EQ(h.stream.pendingCount(), 1u);
    EXPECT_TRUE(h.stream.hasPendingDueBy(99000));
    EXPECT_FALSE(h.stream.hasPendingDueBy(98999));
}

// ---- T8: the stream cut is its eviction ----

TEST(EvictionRebaseT8, aCutThatDropsANoteOlderThanThirtySecondsMovesTheOrigin) {
    MelodyStream s;
    s.reset(0);
    // 80 events, 0.5 s apart: the cut at 256 tokens drops notes up to ~4 s... use 1 s spacing
    for (int i = 0; i < 80; ++i)
        s.appendEvent({i * 100, 50, 60 + i % 12, 64});
    const int32_t baseBefore = s.base();
    ASSERT_TRUE(s.trimIfNeeded());
    // dropped about 50 events (onsets up to ~5000): older than 30 s -> shifted by 3000 or 6000...
    EXPECT_GT(s.lastCutShift(), 0);
    EXPECT_EQ(s.lastCutShift() % 3000, 0);
    EXPECT_EQ(s.base(), baseBefore + s.lastCutShift());
    for (size_t i = 1; i + 3 < s.tokens().size(); i += 4) {
        EXPECT_GE(s.tokens()[i], 0);
        EXPECT_LE(s.tokens()[i], 9999);
    }
    // continuity: the newest event still has its absolute onset 7900
    const auto t = s.tokens();
    EXPECT_EQ(t[t.size() - 4] + s.base(), 7900);
    EXPECT_EQ(t[0], 55027) << "ANTICIPATE stays the first token";
    EXPECT_EQ(s.clock(), 7900);
}

TEST(EvictionRebaseT8, aCutOfYoungNotesDoesNotMoveTheOrigin) {
    MelodyStream s;
    s.reset(0);
    for (int i = 0; i < 80; ++i)
        s.appendEvent({i * 30, 50, 60, 64}); // 2.4 s of notes
    ASSERT_TRUE(s.trimIfNeeded());
    EXPECT_EQ(s.lastCutShift(), 0);
    EXPECT_EQ(s.base(), 0);
}

} // namespace virtual_orch_test
