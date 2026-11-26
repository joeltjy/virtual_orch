#include <gtest/gtest.h>

#include "VirtualOrch/ProjectPaths.h"
#include "VirtualOrch/reduction/StreamingKvCache.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using namespace StreamingKv;

auto maxDiff(const std::vector<float> &a, const float *b, size_t n) -> float {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i)
        m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

auto argmax(const float *x, size_t n) -> size_t { return static_cast<size_t>(std::max_element(x, x + n) - x); }

} // namespace

// The informative-WAIT precompute tiles the context into 5 branches and runs them in one call. With
// the KV kept in backend-owned tensors, every branch must give what a batch-1 run on the same
// state gives: after tiling, after a second batched step, and after evicting rows.
TEST(StreamingKvBatchedDevice, batchedBranchesEqualBatchOneRuns) {
    const auto model = modelsDir().getChildFile("rt_simultaneous_latest.onnx");
    if (! model.existsAsFile())
        GTEST_SKIP() << "no " << model.getFullPathName();
    OnnxBackend backend;
    std::string error;
    ASSERT_TRUE(backend.init(model.getFullPathName().toRawUTF8(), error)) << error;
    const auto dims = backend.dims();
    const size_t vocab = backend.vocabSize();
    constexpr int kBranches = 5;
    // The GPU gives logits up to ~0.02 apart for the same tokens fed in calls of other shapes (one call
    // vs one token at a time; measured below), so branches are compared to within that, and by argmax.
    constexpr float kTol = 0.05f;

    // a short context: three piano notes (onset, duration, piano pitch, velocity)
    const std::vector<int64_t> context{100, 10050, 11060, 11704, 130, 10040, 11064, 11700, 160, 10030, 11067, 11696};
    State base;
    std::vector<float> baseLogits;
    ASSERT_TRUE(backend.run({}, context, base, baseLogits)) << backend.lastError;
    if (base.device == nullptr)
        GTEST_SKIP() << "this backend copies KV into vectors";

    BatchState tiled;
    ASSERT_TRUE(tileState(base, dims, kBranches, tiled));
    ASSERT_NE(tiled.device, nullptr);
    EXPECT_EQ(tiled.rows, base.rows);

    // step 1: a different piano pitch per branch, then its velocity and an onset
    std::vector<int64_t> feed;
    for (int b = 0; b < kBranches; ++b)
        feed.insert(feed.end(), {11060 + b, 11700, 200});
    BatchState step1;
    std::vector<float> logits1;
    ASSERT_TRUE(backend.runBatched(tiled, feed, step1, logits1)) << backend.lastError;
    ASSERT_EQ(logits1.size(), vocab * kBranches);
    ASSERT_NE(step1.device, nullptr);
    EXPECT_EQ(step1.rows, base.rows + 3);

    // the GPU's own noise for comparison: the same 3 tokens in one call vs one at a time
    {
        State a, x, y, z;
        std::vector<float> la, lx, ly, lz;
        ASSERT_TRUE(backend.run(base, {11060, 11700, 200}, a, la));
        ASSERT_TRUE(backend.run(base, {11060}, x, lx));
        ASSERT_TRUE(backend.run(x, {11700}, y, ly));
        ASSERT_TRUE(backend.run(y, {200}, z, lz));
        State a2;
        std::vector<float> la2;
        ASSERT_TRUE(backend.run(base, {11060, 11700, 200}, a2, la2));
        std::printf("Backend numerical variation: one call vs one token at a time %.4g, the same call twice %.4g\n",
                    maxDiff(la, lz.data(), vocab), maxDiff(la, la2.data(), vocab));
        EXPECT_LT(maxDiff(la, lz.data(), vocab), kTol);
    }
    // the branches are not copies of one another: different pitches, different predictions
    EXPECT_GT(maxDiff(logits1, logits1.data() + vocab, vocab), 0.5f);

    std::vector<State> single(kBranches);
    for (int b = 0; b < kBranches; ++b) {
        std::vector<float> one;
        const std::vector<int64_t> ids(feed.begin() + 3 * b, feed.begin() + 3 * (b + 1));
        ASSERT_TRUE(backend.run(base, ids, single[static_cast<size_t>(b)], one)) << backend.lastError;
        EXPECT_LT(maxDiff(one, logits1.data() + static_cast<size_t>(b) * vocab, vocab), kTol) << "branch " << b;
        EXPECT_EQ(argmax(one.data(), vocab), argmax(logits1.data() + static_cast<size_t>(b) * vocab, vocab));
    }

    // step 2 on top of the batched present: the device state chains
    std::vector<int64_t> feed2;
    for (int b = 0; b < kBranches; ++b)
        feed2.push_back(10010 + 5 * b);
    BatchState step2;
    std::vector<float> logits2;
    ASSERT_TRUE(backend.runBatched(step1, feed2, step2, logits2)) << backend.lastError;
    for (int b = 0; b < kBranches; ++b) {
        State next;
        std::vector<float> one;
        ASSERT_TRUE(backend.run(single[static_cast<size_t>(b)], {feed2[static_cast<size_t>(b)]}, next, one));
        EXPECT_LT(maxDiff(one, logits2.data() + static_cast<size_t>(b) * vocab, vocab), kTol) << "branch " << b;
        single[static_cast<size_t>(b)] = next;
    }

    // timing (printed only): a 5-branch step vs a batch-1 step, one token each
    {
        const auto ms = [](auto f) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 10; ++i)
                f();
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 10;
        };
        BatchState out;
        std::vector<float> lg;
        const double batched = ms([&] { (void) backend.runBatched(step1, feed2, out, lg); });
        State one;
        const double single1 = ms([&] { (void) backend.run(single[0], {feed2[0]}, one, lg); });
        BatchState t;
        const double tiling = ms([&] { (void) tileState(base, dims, kBranches, t); });
        std::printf("Backend step: 5 branches %.1f ms, 1 branch %.1f ms; tiling %.1f ms\n", batched, single1, tiling);
    }

    // eviction on the device: the sink rows and the newest rows, as for a batch-1 state
    auto small = dims;
    small.maxRows = step2.rows + 1; // feeding 3 more rows evicts 2
    small.sinkRows = 4;
    evictBatchForIncoming(step2, small, 3);
    ASSERT_NE(step2.device, nullptr);
    std::vector<int64_t> feed3;
    for (int b = 0; b < kBranches; ++b)
        feed3.insert(feed3.end(), {11070 + b, 11700, 240});
    BatchState step3;
    std::vector<float> logits3;
    ASSERT_TRUE(backend.runBatched(step2, feed3, step3, logits3)) << backend.lastError;
    for (int b = 0; b < kBranches; ++b) {
        auto &one = single[static_cast<size_t>(b)];
        evictForIncoming(one, small, 3);
        State next;
        std::vector<float> lg;
        const std::vector<int64_t> ids(feed3.begin() + 3 * b, feed3.begin() + 3 * (b + 1));
        ASSERT_TRUE(backend.run(one, ids, next, lg));
        EXPECT_EQ(step3.rows, next.rows);
        EXPECT_LT(maxDiff(lg, logits3.data() + static_cast<size_t>(b) * vocab, vocab), kTol) << "branch " << b;
    }
}
