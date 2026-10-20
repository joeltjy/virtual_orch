#include <gtest/gtest.h>

#include <cmath>
#include <map>

#include <VirtualOrch/reduction/AnticipationSampler.h>

namespace {

auto flat(size_t vocab) -> std::vector<float> { return std::vector<float>(vocab, 0.0f); }

} // namespace

namespace virtual_orch_test {

TEST(AnticipationSampler, masksLeaveOnlyLegalEventIdsForEachField) {
    for (const int32_t velocityBase: {AnticipationSampler::kDenseVelocityBase,
                                       AnticipationSampler::kAmtVelocityBase}) {
        AnticipationSampler s(velocityBase);
        AnticipationSampler::Context ctx;
        ctx.lowerOnsetModel = 120;
        const auto logits = flat(velocityBase == AnticipationSampler::kDenseVelocityBase ? 11776 : 55293);

        const auto onset = s.masked(0, logits, ctx);
        for (size_t i = 0; i < onset.size(); ++i)
            EXPECT_EQ(std::isfinite(onset[i]), i >= 120 && i <= 9999) << i;

        const auto dur = s.masked(1, logits, ctx);
        for (size_t i = 0; i < dur.size(); ++i) {
            const bool legal = i >= 10010 && i <= 10995 && (i - 10000) % 5 == 0;
            EXPECT_EQ(std::isfinite(dur[i]), legal) << i;
        }

        const auto note = s.masked(2, logits, ctx);
        for (size_t i = 0; i < note.size(); ++i) {
            const bool legal = i >= 11128 + 21 && i <= 11128 + 108;
            EXPECT_EQ(std::isfinite(note[i]), legal) << i;
        }

        const auto vel = s.masked(3, logits, ctx);
        for (size_t i = 0; i < vel.size(); ++i) {
            const auto rel = static_cast<int64_t>(i) - velocityBase;
            const bool legal = rel >= 8 && rel <= 120 && rel % 8 == 0;
            EXPECT_EQ(std::isfinite(vel[i]), legal) << i;
        }
    }
}

TEST(AnticipationSampler, neverReturnsAnIllegalIdOverManyDraws) {
    AnticipationSampler s(AnticipationSampler::kAmtVelocityBase);
    AnticipationSampler::Context ctx;
    ctx.lowerOnsetModel = 50;
    std::vector<float> logits(55293);
    for (size_t i = 0; i < logits.size(); ++i)
        logits[i] = static_cast<float>(std::sin(static_cast<double>(i) * 0.37));
    for (int n = 0; n < 200; ++n) {
        EXPECT_GE(s.sample(0, logits, ctx, 0.5f), 50);
        const auto d = s.sample(1, logits, ctx, 0.5f);
        EXPECT_TRUE(d >= 10010 && d <= 10995 && (d - 10000) % 5 == 0);
        const auto p = s.sample(2, logits, ctx, 0.5f);
        EXPECT_TRUE(p >= 11128 + 21 && p <= 11128 + 108);
        const auto v = s.sample(3, logits, ctx, 0.5f);
        EXPECT_TRUE(v >= 55028 + 8 && v <= 55028 + 120 && (v - 55028) % 8 == 0);
    }
}

// The tau scheduler ramps with every note it counts, and the app draws once per generated note, so
// each draw here gets a fresh sampler with the same history.
namespace {
auto drawsOf(int field, bool pitchBias, bool durationBias, int target, int draws = 3000) -> int {
    int count = 0;
    const auto logits = flat(11776);
    for (int n = 0; n < draws; ++n) {
        AnticipationSampler s(AnticipationSampler::kDenseVelocityBase);
        s.pitchBias = pitchBias;
        s.durationBias = durationBias;
        s.onsetGapBias = false;
        for (int i = 0; i < 12; ++i)
            s.recordGenerated(1000 + i * 40, 50, field == 2 ? 60 : 60 + i, 64);
        AnticipationSampler::Context ctx;
        ctx.onsetAbs = 1000 + 12 * 40;
        ctx.clockAbs = ctx.onsetAbs;
        count += s.sample(field, logits, ctx, 1.0f, 1.0f) == target ? 1 : 0;
    }
    return count;
}
} // namespace

TEST(AnticipationSampler, pitchBiasPushesAwayFromARepeatedPitch) {
    const int unbiased = drawsOf(2, false, false, 11128 + 60);
    const int biased = drawsOf(2, true, false, 11128 + 60);
    EXPECT_GT(unbiased, 15) << "uniform over 88 pitches: about 34 of 3000";
    EXPECT_LT(biased, unbiased) << "the repeated pitch is less likely with the bias";
}

TEST(AnticipationSampler, durationBiasPushesAwayFromARepeatedDuration) {
    const int unbiased = drawsOf(1, false, false, 10050);
    const int biased = drawsOf(1, false, true, 10050);
    EXPECT_GT(unbiased, 5) << "uniform over 197 durations: about 15 of 3000";
    EXPECT_LT(biased, unbiased) << "a duration the last notes all used is less likely";
}

TEST(AnticipationSampler, resetForgetsTheHistory) {
    AnticipationSampler s(AnticipationSampler::kDenseVelocityBase);
    for (int i = 0; i < 8; ++i)
        s.recordGenerated(100 + i * 10, 30, 60, 64);
    s.reset();
    AnticipationSampler::Context ctx;
    const auto logits = flat(11776);
    int count = 0;
    for (int n = 0; n < 2000; ++n)
        count += s.sample(2, logits, ctx, 1.0f, 1.0f) == 11128 + 60 ? 1 : 0;
    EXPECT_GT(count, 8) << "no bias left: about 23 of 2000";
}

} // namespace virtual_orch_test
