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
    // the model is 32x more sensitive than reported: every step overshoots, growing at first
    TdController c;
    const Plant plant{4.0, 4.0};
    run(c, plant, 16, 0.125, 300);
    EXPECT_LT(std::abs(std::log2(16.0 / plant.density(c.current().gamma))), 0.3);
    EXPECT_LT(c.current().m, 1.0) << "m was halved";
}

TEST(TdController, thinsOutABusyModelWithNegativeGamma) {
    TdController c;
    const Plant plant{32.0, 0.5};
    run(c, plant, 8, 0.5, 120);
    EXPECT_LT(c.current().gamma, 0.0);
    EXPECT_LT(std::abs(std::log2(8.0 / plant.density(c.current().gamma))), 0.3);
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
