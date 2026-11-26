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
        // near the target at most kMaxStep; at least 2x over it (Nov 26 2026) at most kMaxStepFar
        const double limit = c.current().error <= TdController::kFarError ? TdController::kMaxStepFar : TdController::kMaxStep;
        EXPECT_LE(std::abs(c.current().gamma - prev), limit + 1e-12);
        prev = c.current().gamma;
    }
    EXPECT_GE(c.current().gamma, TdController::kMinGamma) << "never past the floor: the old -3 is out of reach";
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

TEST(TdController, farUnderTheTargetTremoloGoesFirst) {
    TdController c;
    c.tick({32, 4.0, 0.3, true, false}); // TD 32, measured 4: far under
    EXPECT_EQ(c.current().why, "tremolo first");
    EXPECT_DOUBLE_EQ(c.current().gamma, TdController::startGamma(32)) << "held for 2 s";
    c.tick({32, 4.0, 0.3, true, false});
    EXPECT_GT(c.current().gamma, TdController::startGamma(32)) << "then it moves";
    c.tick({32, 4.0, 0.3, true, false});
    EXPECT_NE(c.current().why, "tremolo first") << "once per episode, not every update";
    // at TD 4 and below there is no tremolo bonus: no hold
    TdController d;
    d.tick({4, 1.0, 0.3, true, false});
    EXPECT_NE(d.current().why, "tremolo first");
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


TEST(TdController, aNewLevelStartsFromItsWarmStartGamma) {
    EXPECT_DOUBLE_EQ(TdController::startGamma(4), 0.0);
    EXPECT_DOUBLE_EQ(TdController::startGamma(2), 0.0);
    EXPECT_DOUBLE_EQ(TdController::startGamma(16), 2.0);
    EXPECT_DOUBLE_EQ(TdController::startGamma(64), 4.0);
    TdController c;
    c.tick({16, 16.0, 0.5, true, false}); // on target: dead zone, gamma stays at the warm start
    EXPECT_DOUBLE_EQ(c.current().gamma, 2.0);
    c.tick({64, 64.0, 0.5, true, false});
    EXPECT_DOUBLE_EQ(c.current().gamma, 4.0);
    c.tick({TemporalDensity::kAuto, 64.0, 0.5, true, false});
    c.tick({64, 64.0, 0.5, true, false});
    EXPECT_DOUBLE_EQ(c.current().gamma, 4.0) << "AUTO and back to the same level keeps gamma";
}

TEST(TdController, aPedalPressKeepsAPositiveGammaAndDropsANegativeOne) {
    TdController c;
    run(c, Plant{1.0, 0.3}, 16, 0.3, 20); // a sparse model: gamma climbs past the warm start
    const double learnt = c.current().gamma;
    ASSERT_GT(learnt, TdController::startGamma(16));
    c.softReset();
    EXPECT_DOUBLE_EQ(c.current().gamma, learnt);
    EXPECT_DOUBLE_EQ(c.current().m, 1.0);

    TdController busy;
    run(busy, Plant{64.0, 0.5}, 4, 0.5, 40);
    ASSERT_LT(busy.current().gamma, -1.0);
    busy.softReset();
    EXPECT_DOUBLE_EQ(busy.current().gamma, 0.0) << "a negative gamma does not survive the pedal";
}

TEST(TdController, gammaRisesFasterThanItFalls) {
    TdController c;
    c.tick({16, 1.0, 0.05, true, false}); // far too sparse (error +2): first the 2 s tremolo-first hold
    ASSERT_EQ(c.current().why, "tremolo first");
    c.tick({16, 1.0, 0.05, true, false}); // then low sensitivity: K e = 0.86
    EXPECT_GT(c.current().gamma - TdController::startGamma(16), 3 * TdController::kMaxStep) << "more than the old 0.25";
    EXPECT_LE(c.current().gamma - TdController::startGamma(16), TdController::kMaxStepUp + 1e-12);
    TdController d;
    d.tick({4, 7.5, 0.05, true, false}); // too dense, but under 2x the target: K e = -0.27, capped
    EXPECT_NEAR(d.current().gamma, -TdController::kMaxStep, 1e-12);
}

TEST(TdController, farOverTheTargetGammaDropsFaster) {
    // the sample viewer's Solemn run (Nov 26 2026): TD 2, a repeated 13 cs chord measured 15-16
    TdController c;
    c.tick({2, 16.0, 0.3, true, false});
    EXPECT_NEAR(c.current().gamma, -TdController::kMaxStepFar, 1e-12) << "error -2: the full far step";
    c.tick({2, 16.0, 0.3, true, false});
    EXPECT_NEAR(c.current().gamma, TdController::kMinGamma, 1e-12) << "the floor in 2 updates (4 s), not 8";
    TdController d;
    d.tick({2, 4.0, 0.3, true, false}); // exactly 2x: error -1, at least kFarGain x 1
    EXPECT_NEAR(d.current().gamma, -TdController::kFarGain, 1e-12);
}

TEST(TdController, farUnderTheTargetGammaComesBackFaster) {
    // after a pull-back to the floor the model went near silent (measured 0.67 at TD 2)
    TdController c;
    c.tick({2, 16.0, 0.3, true, false});
    c.tick({2, 16.0, 0.3, true, false});
    ASSERT_NEAR(c.current().gamma, TdController::kMinGamma, 1e-12);
    c.tick({2, 0.67, 0.3, true, false}); // error log2(2 / 0.67) = 1.58
    EXPECT_GE(c.current().gamma - TdController::kMinGamma, TdController::kFarGain * 1.5) << "not ~0.03 an update";
}

TEST(TdController, reachesTd16ThoughThePedalIsPressedEveryFourSeconds) {
    // Nov 24 2026, 21:50: model 1 at ~4 onsets per 2 s, the pianist pedalling every 2-5 s. A model that needs
    // gamma 5 for TD 16 (base 4, 0.4 TD steps per unit): before, every press went back to gamma 0 and 0.25 per tick
    // never got there; with the warm start, the faster rise and the soft reset it does within ~10 s.
    const Plant plant{4.0, 0.4};
    TdController c;
    double best = 0.0;
    for (int i = 0; i < 30; ++i) {
        const double d = plant.density(c.current().gamma);
        c.tick({16, d, 0.4, true, false});
        if (i % 2 == 1)
            c.softReset(); // a pedal press every 2 ticks
        if (i >= 5)
            best = std::max(best, plant.density(c.current().gamma));
    }
    EXPECT_LT(std::abs(std::log2(16.0 / plant.density(c.current().gamma))), 0.35);

    // the old behaviour for comparison: a full reset to 0 at every press, at most 0.25 per tick
    double gamma = 0.0, oldBest = 0.0;
    for (int i = 0; i < 30; ++i) {
        gamma = std::min(gamma + 0.25, 20.0);
        oldBest = std::max(oldBest, plant.density(gamma));
        if (i % 2 == 1)
            gamma = 0.0;
    }
    EXPECT_LT(oldBest, 6.0) << "the old controller stays near TD 4";
}

} // namespace virtual_orch_test
