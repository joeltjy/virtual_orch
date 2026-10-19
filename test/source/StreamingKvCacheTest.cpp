#include <gtest/gtest.h>

#include <VirtualOrch/reduction/StreamingKvCache.h>

namespace {

using namespace StreamingKv;

/**
 * Fake model. One layer/head/dim: a cache row is the token id as float. The "logits" of the last
 * token are a function of every row of the cache plus the new ids, so two caches that hold the
 * same tokens give the same logits.
 */
class FakeBackend : public Backend {
public:
    explicit FakeBackend(int maxRows = 16) { d.layers = 1; d.heads = 1; d.headDim = 1;
                                             d.maxRows = maxRows; d.sinkRows = 4; }
    [[nodiscard]] auto dims() const -> Dims override { return d; }
    auto run(const State &past, const std::vector<int64_t> &ids, State &present,
             std::vector<float> &last) -> bool override {
        ++calls;
        if (failNext) { failNext = false; return false; }
        EXPECT_LE(past.rows + static_cast<int>(ids.size()), d.maxRows) << "must evict before feeding";
        present = past;
        for (const auto id: ids) {
            present.key.push_back(static_cast<float>(id));
            present.value.push_back(static_cast<float>(id));
            ++present.rows;
        }
        last = {signature(present)};
        return true;
    }
    static auto signature(const State &s) -> float {
        double h = 0;
        for (const float k: s.key)
            h = h * 31.0 + k + 1.0;
        return static_cast<float>(std::fmod(h, 1.0e6));
    }
    Dims d;
    int calls = 0;
    bool failNext = false;
};

auto reference(const std::vector<int32_t> &tokens, int maxRows = 16) -> float {
    FakeBackend backend(maxRows);
    CommittedScratchCache cache(backend, 1);
    return cache.logitsAfter(tokens, tokens.size())[0];
}

} // namespace

namespace virtual_orch_test {

TEST(StreamingKvRows, cutAndEvictKeepSinksAndNewestRows) {
    Dims d; d.layers = 2; d.heads = 2; d.headDim = 1; d.maxRows = 8; d.sinkRows = 2;
    State s;
    s.rows = 8;
    for (int block = 0; block < 4; ++block)
        for (int r = 0; r < 8; ++r) {
            s.key.push_back(static_cast<float>(block * 100 + r));
            s.value.push_back(static_cast<float>(block * 100 + r + 50));
        }
    auto cut = s;
    cutRows(cut, d, 3);
    ASSERT_EQ(cut.rows, 3);
    EXPECT_EQ(cut.key, (std::vector<float>{0, 1, 2, 100, 101, 102, 200, 201, 202, 300, 301, 302}));

    evictForIncoming(s, d, 3); // 8 + 3 > 8: keep 2 sinks + (8 - 2 - 3) = 3 newest rows
    ASSERT_EQ(s.rows, 5);
    EXPECT_EQ(s.key[0], 0.0f);
    EXPECT_EQ(s.key[1], 1.0f);
    EXPECT_EQ(s.key[2], 5.0f);
    EXPECT_EQ(s.key[4], 7.0f);
    EXPECT_EQ(s.key[5], 100.0f) << "second block starts after 5 rows";
    EXPECT_EQ(s.value[2], 55.0f);

    State small; small.rows = 3; small.key = {1, 2, 3}; small.value = {1, 2, 3};
    d.layers = 1; d.heads = 1;
    evictForIncoming(small, d, 2);
    EXPECT_EQ(small.rows, 3) << "no eviction while it fits";
}

TEST(CommittedScratchCache, appendOnlyFeedsEachTokenOnce) {
    FakeBackend backend;
    CommittedScratchCache cache(backend, 4);
    std::vector<int32_t> tokens{1, 2, 3, 4, 5};
    EXPECT_FLOAT_EQ(cache.logitsAfter(tokens, tokens.size())[0], reference(tokens));
    tokens.push_back(6);
    tokens.push_back(7);
    EXPECT_FLOAT_EQ(cache.logitsAfter(tokens, tokens.size())[0], reference(tokens));
    EXPECT_EQ(cache.tokensFed(), 7u);
    EXPECT_EQ(cache.committedFed(), 7u);
    // asking again for the same list re-feeds nothing
    const int calls = backend.calls;
    EXPECT_FLOAT_EQ(cache.logitsAfter(tokens, tokens.size())[0], reference(tokens));
    EXPECT_EQ(backend.calls, calls);
}

TEST(CommittedScratchCache, scratchTailIsNotCommittedAndIsRebuiltWhenItChanges) {
    FakeBackend backend;
    CommittedScratchCache cache(backend, 64);
    std::vector<int32_t> tokens{1, 2, 3, 4, /*held control*/ 50, 51, 52, 53};
    const size_t committed = 4;
    EXPECT_FLOAT_EQ(cache.logitsAfter(tokens, committed)[0], reference(tokens));
    EXPECT_EQ(cache.committedFed(), 4u);
    EXPECT_EQ(cache.scratchFed(), 4u);

    // tokens generated after the held control: W grows, nothing re-fed
    tokens.push_back(60);
    tokens.push_back(61);
    const size_t fedBefore = cache.tokensFed();
    EXPECT_FLOAT_EQ(cache.logitsAfter(tokens, committed)[0], reference(tokens));
    EXPECT_EQ(cache.tokensFed() - fedBefore, 2u);
    EXPECT_EQ(cache.committedFed(), 4u) << "the held control is still not committed";

    // note-off: the control's duration token changes -> the tail is re-fed from the commit point
    tokens[6] = 99;
    EXPECT_FLOAT_EQ(cache.logitsAfter(tokens, committed)[0], reference(tokens));
    EXPECT_EQ(cache.committedFed(), 4u);
    EXPECT_EQ(cache.scratchFed(), 6u);

    // confirmed: everything is committed now
    EXPECT_FLOAT_EQ(cache.logitsAfter(tokens, tokens.size())[0], reference(tokens));
    EXPECT_EQ(cache.committedFed(), tokens.size());
    EXPECT_EQ(cache.scratchFed(), 0u);
}

TEST(CommittedScratchCache, committedCacheNeverHoldsScratchRows) {
    FakeBackend backend;
    CommittedScratchCache cache(backend, 64);
    std::vector<int32_t> tokens{1, 2, 3, 70, 71};
    (void) cache.logitsAfter(tokens, 3);
    EXPECT_EQ(cache.committedRows(), 3);
    tokens[3] = 80; // scratch changes, C untouched
    EXPECT_FLOAT_EQ(cache.logitsAfter(tokens, 3)[0], reference(tokens));
    EXPECT_EQ(cache.committedRows(), 3);
}

TEST(CommittedScratchCache, evictionKeepsSinksAndStaysInsideTheWindow) {
    FakeBackend backend(16);
    CommittedScratchCache cache(backend, 3);
    std::vector<int32_t> tokens;
    for (int i = 0; i < 40; ++i) {
        tokens.push_back(100 + i);
        const auto &logits = cache.logitsAfter(tokens, tokens.size());
        ASSERT_FALSE(logits.empty());
    }
    EXPECT_LE(cache.committedRows(), 16);
    EXPECT_FALSE(cache.lastFailed());
}

TEST(CommittedScratchCache, aChangedCommittedPrefixRebuildsTheCache) {
    FakeBackend backend;
    CommittedScratchCache cache(backend, 64);
    std::vector<int32_t> a{1, 2, 3, 4};
    (void) cache.logitsAfter(a, a.size());
    std::vector<int32_t> b{9, 8, 7, 6, 5}; // e.g. a time rebase rewrote the stream
    EXPECT_FLOAT_EQ(cache.logitsAfter(b, b.size())[0], reference(b));
    EXPECT_EQ(cache.committedFed(), 5u);
}

TEST(CommittedScratchCache, backendFailureReturnsNothingAndRecovers) {
    FakeBackend backend;
    CommittedScratchCache cache(backend, 64);
    std::vector<int32_t> t{1, 2, 3};
    backend.failNext = true;
    EXPECT_TRUE(cache.logitsAfter(t, t.size()).empty());
    EXPECT_TRUE(cache.lastFailed());
    EXPECT_FLOAT_EQ(cache.logitsAfter(t, t.size())[0], reference(t));
}

} // namespace virtual_orch_test
