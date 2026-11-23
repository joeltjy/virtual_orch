#include <gtest/gtest.h>

#include <VirtualOrch/reduction/TdController.h>
#include <VirtualOrch/reduction/TdOnsetBias.h>

#include <cmath>
#include <vector>

namespace virtual_orch_test {

namespace {

/** A density that answers gamma like the model would in the small: base x 2^(s x gamma). */
struct Plant {
    double base;
    double s;
    double ceiling = 1e9;
    [[nodiscard]] auto density(double gamma) const -> double { return std::min(ceiling, base * std::pow(2.0, s * gamma)); }
};

auto run(TdController &c, const Plant &plant, int32_t target, double reportedS, int ticks,
         double *maxDensity = nullptr) -> void {
    for (int i = 0; i < ticks; ++i) {
        const double d = plant.density(c.current().gamma);
        if (maxDensity != nullptr)
            *maxDensity = std::max(*maxDensity, d);
        c.tick({target, d, reportedS, true, false});
    }
}

/** Mean gap of softmax((scores + gamma u) / T) over c > 0, gaps offset by `offset`. */
auto meanGap(const std::vector<float> &scores, int32_t lower, int32_t offset, int32_t td, float temp, double gamma)
    -> double {
    double z = 0.0, eg = 0.0;
    for (int32_t t = lower + 1; t < static_cast<int32_t>(scores.size()); ++t) {
        const double l = scores[static_cast<size_t>(t)] + gamma * TdOnsetBias::kernel(td, t - lower);
        const double p = std::exp(l / temp);
        z += p;
        eg += p * (t - lower + offset);
    }
    return eg / z;
}

} // namespace

TEST(TdController, sensitivityMatchesTheDifferenceQuotient) {
    // an onset distribution with a short and a long bump
    std::vector<float> scores(400, -std::numeric_limits<float>::infinity());
    const int32_t lower = 20;
    for (int32_t t = lower; t < 400; ++t) {
        const double c = t - lower;
        scores[static_cast<size_t>(t)] = static_cast<float>(std::log(std::exp(-std::pow((c - 25) / 8.0, 2))
                                                                     + 0.6 * std::exp(-std::pow((c - 120) / 30.0, 2)) + 1e-4));
    }
    for (const float temp: {1.0f, 0.5f}) {
        const auto s = TdOnsetBias::sensitivity(scores, lower, 10, 8, temp);
        ASSERT_TRUE(s.has_value());
        const double h = 1e-3;
        const double numeric = -(std::log2(meanGap(scores, lower, 10, 8, temp, h))
                                 - std::log2(meanGap(scores, lower, 10, 8, temp, 0.0))) / h;
        EXPECT_NEAR(*s, numeric, 0.02 * std::abs(numeric)) << "temperature " << temp;
        EXPECT_GT(*s, 0.0) << "more gamma, denser";
    }
}

TEST(TdController, reachesTheTargetWithoutOvershooting) {
    TdController c;
    const Plant plant{4.0, 0.5};
    double maxDensity = 0.0;
    run(c, plant, 16, 0.5, 120, &maxDensity);
    EXPECT_LT(std::abs(std::log2(16.0 / plant.density(c.current().gamma))), 0.25);
    EXPECT_LT(maxDensity, 16.0 * 1.2) << "no big overshoot";
    EXPECT_FALSE(c.current().frozen);
}

TEST(TdController, aWrongSensitivityIsCaughtByTheOvershootRule) {
    // the model is 128x more sensitive than the floor kMinSensitivity: even a capped step
    // (kMaxStep, 6 TD steps here) overshoots, so the overshoot rule must shrink m
    TdController c;
    const Plant plant{4.0, 25.6};
    run(c, plant, 16, 0.125, 300);
    EXPECT_LT(std::abs(std::log2(16.0 / plant.density(c.current().gamma))), 0.3);
    EXPECT_LT(c.current().m, 1.0) << "m was halved";
}

TEST(TdController, thinsOutABusyModelWithNegativeGamma) {
    TdController c;
    const Plant plant{12.0, 0.5}; // TD 8 needs gamma -1.17, inside the bound
    run(c, plant, 8, 0.5, 120);
    EXPECT_LT(c.current().gamma, 0.0);
    EXPECT_LT(std::abs(std::log2(8.0 / plant.density(c.current().gamma))), 0.3);
}

TEST(TdController, aVeryBusyModelStopsAtTheNegativeLimit) {
    // Nov 23 2026: a dense v1 burst at TD 4 drove gamma to -3 and v1 to onsets 40-80 s ahead.
    TdController c;
    const Plant plant{64.0, 0.5}; // TD 4 would need gamma -8
    run(c, plant, 4, 0.5, 400);
    EXPECT_DOUBLE_EQ(c.current().gamma, TdController::kMinGamma);
    EXPECT_GE(TdController::kMinGamma, -2.0);
    EXPECT_EQ(c.current().why, "at limit");
}

TEST(TdController, theNov23BurstMovesGammaAtMostOneStepPerUpdate) {
    // v1's TD_CTRL on Nov 23 2026 (TD 4): measured and S per update as logged. At 250 ms ticks
    // gamma went +1.49 -> -3.02 in four updates; now each update moves it by at most kMaxStep.
    TdController c;
    c.tick({4, 2.67, 0.199, true, false});
    const double start = c.current().gamma;
    const double measured[] = {6.67, 9.33, 11.33, 14.67};
    const double s[] = {0.103, 0.073, 0.081, 0.057};
    double prev = start;
    for (int i = 0; i < 4; ++i) {
        c.tick({4, measured[i], s[i], true, false});
        EXPECT_LE(std::abs(c.current().gamma - prev), TdController::kMaxStep + 1e-12);
        prev = c.current().gamma;
    }
    EXPECT_GE(c.current().gamma, start - 4 * TdController::kMaxStep);
    EXPECT_GT(c.current().gamma, -1.0) << "4 updates (8 s) cannot reach the old -3";
    EXPECT_DOUBLE_EQ(TdController::kTickMs, 2000.0);
}

TEST(TdController, resetGoesBackToGammaZeroAndKeepsTheTarget) {
    TdController c;
    run(c, Plant{64.0, 0.5}, 4, 0.5, 40);
    ASSERT_LT(c.current().gamma, -1.0);
    c.reset();
    EXPECT_DOUBLE_EQ(c.current().gamma, 0.0);
    EXPECT_DOUBLE_EQ(c.current().m, 1.0);
    EXPECT_EQ(c.current().target, 4);
    EXPECT_TRUE(c.current().frozen);
}

TEST(TdController, negativeGammaBonusStopsGrowingFarAhead) {
    const int32_t td = 4; // mean gap 50 cs, the bonus held from 200 cs on
    const float gamma = -2.0f;
    const float at2s = gamma * TdOnsetBias::kernelFor(gamma, td, 200);
    EXPECT_GT(at2s, gamma * TdOnsetBias::kernelFor(gamma, td, 100)) << "still favours longer gaps up to the cap";
    EXPECT_FLOAT_EQ(gamma * TdOnsetBias::kernelFor(gamma, td, 4000), at2s) << "40 s ahead gets no more than 2 s";
    // positive gamma keeps the full taper
    EXPECT_FLOAT_EQ(TdOnsetBias::kernelFor(1.0f, td, 4000), TdOnsetBias::kernel(td, 4000));
}

TEST(TdController, freezesAtAutoWhenStoppedWaitingOrNotDrawing) {
    TdController c;
    c.tick({16, 4.0, 0.5, true, false});
    const double g = c.current().gamma;
    EXPECT_GT(g, 0.0);
    EXPECT_EQ(c.tick({TemporalDensity::kAuto, 4.0, 0.5, true, false}).why, "TD AUTO");
    EXPECT_EQ(c.tick({16, 4.0, 0.5, false, false}).why, "clock stopped");
    EXPECT_EQ(c.tick({16, 4.0, 0.5, true, true}).why, "waiting");
    EXPECT_EQ(c.tick({16, 4.0, std::nullopt, true, false}).why, "not drawing");
    EXPECT_TRUE(c.current().frozen);
    EXPECT_DOUBLE_EQ(c.current().gamma, g) << "frozen keeps gamma";
}

TEST(TdController, smallErrorsAreIgnoredAtLowTd) {
    TdController c;
    // TD 2: N = 3, dead zone 0.58 TD steps; 1.5 is 0.42 below
    c.tick({2, 1.5, 0.5, true, false});
    EXPECT_EQ(c.current().why, "dead zone");
    EXPECT_DOUBLE_EQ(c.current().gamma, 0.0);
    // TD 64: N = 96, dead zone 0.10; the same 0.42 counts
    c.tick({64, 48.0, 0.5, true, false});
    EXPECT_GT(c.current().gamma, 0.0);
}

TEST(TdController, anUnreachableTargetStopsAtTheLimit) {
    TdController c;
    const Plant plant{4.0, 0.5, 6.0}; // never denser than 6
    run(c, plant, 64, 0.5, 400);
    EXPECT_DOUBLE_EQ(c.current().gamma, TdController::kMaxGamma);
    EXPECT_EQ(c.current().why, "at limit");
}

TEST(TdController, silenceIsAFiniteError) {
    TdController c;
    c.tick({16, std::nullopt, 0.5, true, false});
    EXPECT_DOUBLE_EQ(c.current().error, TdController::kMaxError);
    EXPECT_TRUE(std::isfinite(c.current().gamma));
}

} // namespace virtual_orch_test
