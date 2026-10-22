#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

#include <VirtualOrch/ProjectPaths.h>
#include <VirtualOrch/TdTokens.h>
#include <VirtualOrch/reduction/StreamingKvCache.h>

/**
 * Checks the app's KV cache (StreamingKv::CommittedScratchCache over OnnxBackend) against a full
 * no-cache recompute on the real T3'' (rope_cache_RT.onnx). Skipped when the model is not here.
 * Prints the numbers (max |logit diff|, argmax agreement, speed) so they can be read in the log.
 */
namespace virtual_orch_test {

namespace {

auto t3Onnx() -> juce::File {
    std::vector<juce::File> dirs{documentsModelsDir(), onnxExportDir(), onnxExportFallbackDir()};
    for (const auto &d: onnxExportDirs())
        dirs.push_back(d);
    for (const auto &d: dirs)
        if (const auto f = d.getChildFile("rope_cache_RT.onnx"); f.existsAsFile())
            return f;
    return {};
}

/**
 * A T3''-shaped dense stream: TD token every 2 s, then quads (onset, duration, note, velocity),
 * piano (instrument 0) and reduction (instrument 1) interleaved, onsets moving forward.
 */
auto t3Stream(size_t tokens) -> std::vector<int32_t> {
    std::vector<int32_t> out;
    int32_t onset = 0;
    int32_t nextTd = 0;
    int i = 0;
    while (out.size() < tokens) {
        if (onset >= nextTd) {
            out.push_back(TdTokens::denseIdForLevel(TdTokens::kDenseLevels[1 + (i / 7) % 6], 11776));
            nextTd += TdTokens::kBoundaryCs;
            continue;
        }
        const int32_t instrument = i % 3 == 2 ? 1 : 0;
        const int32_t pitch = 48 + (i * 7) % 30;
        out.push_back(onset);
        out.push_back(10000 + 20 + (i * 15) % 120);
        out.push_back(11000 + 128 * instrument + pitch);
        out.push_back(11640 + 32 + (i * 8) % 64);
        onset += (i % 4 == 0) ? 0 : 10 + (i * 13) % 40;
        ++i;
    }
    out.resize(tokens);
    return out;
}

/** Logits after `ids` with no cache: one call from an empty past. */
auto fullRecompute(StreamingKv::OnnxBackend &backend, const std::vector<int32_t> &ids)
    -> std::vector<float> {
    StreamingKv::State past;
    StreamingKv::State present;
    std::vector<float> logits;
    const std::vector<int64_t> ids64(ids.begin(), ids.end());
    EXPECT_TRUE(backend.run(past, ids64, present, logits)) << backend.lastError;
    return logits;
}

struct Diff {
    float maxAbs = 0.0f;
    bool sameArgmax = false;
};

auto compare(const std::vector<float> &a, const std::vector<float> &b) -> Diff {
    Diff d;
    if (a.size() != b.size() || a.empty())
        return d;
    for (size_t i = 0; i < a.size(); ++i)
        d.maxAbs = std::max(d.maxAbs, std::abs(a[i] - b[i]));
    d.sameArgmax = std::max_element(a.begin(), a.end()) - a.begin()
                   == std::max_element(b.begin(), b.end()) - b.begin();
    return d;
}

using Clock = std::chrono::steady_clock;
auto msSince(Clock::time_point t) -> double {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

struct T3KvCache : ::testing::Test {
    void SetUp() override {
        const auto onnx = t3Onnx();
        if (! onnx.existsAsFile())
            GTEST_SKIP() << "rope_cache_RT.onnx is not on this machine";
        std::string error;
        ASSERT_TRUE(backend.init(onnx.getFullPathName().toRawUTF8(), error)) << error;
    }
    StreamingKv::OnnxBackend backend;
};

constexpr float kTolerance = 2e-2f; // fp32, chunked vs one-shot attention (export log: ~1e-4)

} // namespace

TEST_F(T3KvCache, incrementalCacheMatchesFullRecomputeBelow512) {
    const auto stream = t3Stream(500);
    StreamingKv::CommittedScratchCache cache(backend);
    const std::vector<size_t> checkpoints{1, 5, 37, 64, 65, 128, 200, 333, 500};

    double cachedMs = 0.0;
    float worst = 0.0f;
    for (size_t n = 1; n <= stream.size(); ++n) {
        const std::vector<int32_t> prefix(stream.begin(), stream.begin() + static_cast<std::ptrdiff_t>(n));
        const auto t0 = Clock::now();
        const auto cached = cache.logitsAfter(prefix, n); // all committed, one new token per step
        cachedMs += msSince(t0);
        ASSERT_FALSE(cache.lastFailed()) << n;
        if (std::find(checkpoints.begin(), checkpoints.end(), n) == checkpoints.end())
            continue;
        const auto t1 = Clock::now();
        const auto full = fullRecompute(backend, prefix);
        const double fullMs = msSince(t1);
        const auto d = compare(cached, full);
        worst = std::max(worst, d.maxAbs);
        std::printf("[kv] n=%3zu  max|diff|=%.2e  argmax %s  full recompute %.0f ms\n", n, d.maxAbs,
                    d.sameArgmax ? "same" : "DIFFERENT", fullMs);
        EXPECT_LT(d.maxAbs, kTolerance) << n;
        EXPECT_TRUE(d.sameArgmax) << n;
    }
    std::printf("[kv] cached (KV on %s): %.1f ms per token over %zu tokens; worst max|diff| %.2e\n",
                backend.keepsKvOnDevice() ? "GPU" : "host",
                cachedMs / static_cast<double>(stream.size()), stream.size(), worst);

    // Each token went through the model exactly once, one call per step.
    EXPECT_EQ(cache.tokensFed(), stream.size());
    EXPECT_EQ(cache.backendCalls(), static_cast<int>(stream.size()));
    EXPECT_EQ(cache.committedRows(), static_cast<int>(stream.size()));
}

TEST_F(T3KvCache, scratchTailIsRecomputedWhenItChanges) {
    const auto stream = t3Stream(300);
    StreamingKv::CommittedScratchCache cache(backend);
    const size_t committed = 280;

    // committed 280 + scratch 20
    const auto a = cache.logitsAfter(stream, committed);
    const auto da = compare(a, fullRecompute(backend, stream));
    std::printf("[kv] scratch tail 20: max|diff|=%.2e argmax %s\n", da.maxAbs, da.sameArgmax ? "same" : "DIFFERENT");
    EXPECT_LT(da.maxAbs, kTolerance);
    const size_t fedAfterA = cache.tokensFed();

    // the tail changes (a held note's duration grows): only the tail is fed again
    auto changed = stream;
    changed[committed + 2] += 15;
    const auto b = cache.logitsAfter(changed, committed);
    const auto db = compare(b, fullRecompute(backend, changed));
    std::printf("[kv] changed tail: max|diff|=%.2e argmax %s, fed %zu more tokens\n", db.maxAbs,
                db.sameArgmax ? "same" : "DIFFERENT", cache.tokensFed() - fedAfterA);
    EXPECT_LT(db.maxAbs, kTolerance);
    EXPECT_EQ(cache.tokensFed() - fedAfterA, stream.size() - committed) << "the committed part is reused";
    EXPECT_GT(compare(a, b).maxAbs, 1e-3f) << "the change reaches the logits";

    // the same tail again: nothing is fed
    const size_t fedAfterB = cache.tokensFed();
    const auto c = cache.logitsAfter(changed, committed);
    EXPECT_EQ(cache.tokensFed(), fedAfterB);
    EXPECT_EQ(compare(b, c).maxAbs, 0.0f);

    // a changed COMMITTED prefix resets the cache and still matches
    auto rewritten = changed;
    rewritten[10] += 5;
    const auto r = cache.logitsAfter(rewritten, committed);
    EXPECT_LT(compare(r, fullRecompute(backend, rewritten)).maxAbs, kTolerance);
}

TEST_F(T3KvCache, evictionKeepsTheCacheAt512AndTheLogitsSane) {
    const auto stream = t3Stream(900);
    StreamingKv::CommittedScratchCache cache(backend);
    int sameAsWindow = 0;
    int compared = 0;
    for (size_t n = 1; n <= stream.size(); ++n) {
        const std::vector<int32_t> prefix(stream.begin(), stream.begin() + static_cast<std::ptrdiff_t>(n));
        const auto logits = cache.logitsAfter(prefix, n);
        ASSERT_FALSE(cache.lastFailed()) << n;
        ASSERT_LE(cache.committedRows(), 512) << n;
        ASSERT_TRUE(std::all_of(logits.begin(), logits.end(), [](float l) { return std::isfinite(l); })) << n;
        if (n > 512 && n % 97 == 0) {
            // informational: a fresh run on the 4 sinks + the last 507 tokens (not expected equal:
            // the cache keeps hidden states computed with the evicted context, as in training)
            std::vector<int32_t> window(stream.begin(), stream.begin() + 4);
            window.insert(window.end(), prefix.end() - 507, prefix.end());
            const auto d = compare(logits, fullRecompute(backend, window));
            ++compared;
            sameAsWindow += d.sameArgmax ? 1 : 0;
            std::printf("[kv] n=%3zu rows=%d  vs window recompute max|diff|=%.2f argmax %s\n", n,
                        cache.committedRows(), d.maxAbs, d.sameArgmax ? "same" : "different");
        }
    }
    EXPECT_EQ(cache.committedRows(), 512);
    EXPECT_EQ(cache.evictedTokens(), stream.size() - 512);
    std::printf("[kv] past 512: argmax same as a window recompute %d/%d\n", sameAsWindow, compared);
}

} // namespace virtual_orch_test
