#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <VirtualOrch/reduction/TdOnsetBias.h>
#include <VirtualOrch/widgets/OnsetGapsWidget.h>

namespace virtual_orch_test {

namespace {

/** Restores LogitTD's default (on) after a test. */
struct LogitTdOn {
    LogitTdOn() { TdOnsetBias::setEnabled(true); }
    ~LogitTdOn() { TdOnsetBias::setEnabled(true); }
};

} // namespace

TEST(TdOnsetBias, isZeroForAutoLevelOneAndNoGap) {
    EXPECT_EQ(TdOnsetBias::shift(TemporalDensity::kAuto, 100), 0.0f);
    EXPECT_EQ(TdOnsetBias::shift(1, 100), 0.0f);
    for (const int32_t k: TemporalDensity::kLevels)
        EXPECT_EQ(TdOnsetBias::shift(k, 0), 0.0f);
}

TEST(TdOnsetBias, isFlatBelowTheMeanGapAndTapersBeyond) {
    for (const int32_t k: {2, 8, 64}) {
        const int32_t meanGap = 200 / k;
        for (int32_t c = 0; c < 400; c += 5)
            EXPECT_GE(TdOnsetBias::shift(k, c), TdOnsetBias::shift(k, c + 5)) << k << " " << c;
        const float a = TdOnsetBias::strength() * std::log2(static_cast<float>(k)) / 6.0f;
        EXPECT_GT(TdOnsetBias::shift(k, meanGap / 2), -0.02f * a) << "nearly flat below the knee";
        EXPECT_LT(TdOnsetBias::shift(k, 4 * meanGap), -1.0f * a) << "tapered beyond it";
    }
}

TEST(TdOnsetBias, higherTdFavoursShortGapsMoreAgainstLongOnes) {
    // f(k, short) - f(k, long): the log-odds of a short gap over a long one, growing with k.
    float previous = -1.0f;
    for (const int32_t k: TemporalDensity::kLevels) {
        const float advantage = TdOnsetBias::shift(k, 10) - TdOnsetBias::shift(k, 200);
        EXPECT_GT(advantage, previous) << k;
        previous = advantage;
    }
    // c = 200 at TD 64 is 64 mean gaps: -(a / 4) ln(1 + 64^4), a = s
    const float tail = std::log1p(std::pow(64.0f, 4.0f)) / 4.0f;
    EXPECT_NEAR(TdOnsetBias::shift(64, 200), -TdOnsetBias::strength() * tail, 1e-3);
    TdOnsetBias::setStrength(3.0f);
    EXPECT_NEAR(TdOnsetBias::shift(64, 200), -3.0f * tail, 1e-3);
    TdOnsetBias::setStrength(100.0f);
    EXPECT_EQ(TdOnsetBias::strength(), TdOnsetBias::kMaxStrength) << "clamped";
    TdOnsetBias::setStrength(TdOnsetBias::kDefaultStrength);
}

namespace {

/** P(onset t) at temperature T over the finite logits. */
auto probabilities(const std::vector<float> &logits, float temperature) -> std::vector<double> {
    double peak = -1e30;
    for (const float l: logits)
        if (std::isfinite(l))
            peak = std::max(peak, static_cast<double>(l));
    std::vector<double> p(logits.size(), 0.0);
    double total = 0.0;
    for (size_t i = 0; i < logits.size(); ++i)
        if (std::isfinite(logits[i]))
            total += p[i] = std::exp((logits[i] - peak) / temperature);
    for (auto &x: p)
        x /= total;
    return p;
}

} // namespace

TEST(TdOnsetBias, shortGapsAreDiscouragedSmoothlyEvenWhenOff) {
    const LogitTdOn guard;
    TdOnsetBias::setEnabled(false);
    std::vector<float> logits(40, 0.0f);
    TdOnsetBias::apply(logits, TemporalDensity::kAuto, 10, 0, 40);
    EXPECT_EQ(logits[9], 0.0f) << "before the bound: untouched (masked elsewhere)";
    EXPECT_EQ(logits[10], 0.0f) << "c = 0 (a chord) untouched";
    EXPECT_NEAR(std::exp(logits[16]), 0.5f, 1e-5) << "half kept at 6 cs";
    EXPECT_NEAR(std::exp(logits[11]), 1.0f / 1297.0f, 1e-6) << "1 cs nearly gone";
    for (int c = 1; c < 29; ++c)
        EXPECT_LT(logits[static_cast<size_t>(10 + c)], logits[static_cast<size_t>(11 + c)]) << c;
    EXPECT_TRUE(std::isfinite(logits[11])) << "no hard ban";
}

TEST(TdOnsetBias, chordShiftPushesSmallChordsAndBrakesBigOnes) {
    EXPECT_EQ(TdOnsetBias::chordShift(64, 0), 0.0f) << "t past the last onset: not stacking";
    EXPECT_EQ(TdOnsetBias::chordShift(1, 1), 0.0f);
    EXPECT_EQ(TdOnsetBias::chordShift(TemporalDensity::kAuto, 1), 0.0f);
    EXPECT_GT(TdOnsetBias::chordShift(64, 1), TdOnsetBias::chordShift(64, 2));
    EXPECT_GT(TdOnsetBias::chordShift(64, 2), 0.0f);
    EXPECT_NEAR(TdOnsetBias::chordShift(64, 4), 0.0f, 1e-6) << "neutral at the target size";
    EXPECT_LT(TdOnsetBias::chordShift(64, 8), TdOnsetBias::chordShift(64, 5));
    EXPECT_LT(TdOnsetBias::chordShift(64, 5), 0.0f);
    EXPECT_GT(TdOnsetBias::chordShift(64, 1), TdOnsetBias::chordShift(8, 1)) << "grows with TD";
}

TEST(TdOnsetBias, keepsTheChordProbabilityAndReshapesForwardGaps) {
    const LogitTdOn guard;
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();
    for (const float temperature: {1.0f, 0.5f}) {
        // chordNotes 0: no chord shift, so P(c = 0) must stay the model's
        std::vector<float> logits(400, 0.0f);
        logits[3] = kNegInf; // masked before the bound
        logits[2] = 2.0f;    // c = 0 (bound 2): the model likes a chord
        logits[50] = kNegInf; // masked forward id stays masked
        auto banOnly = logits;
        TdOnsetBias::setEnabled(false);
        TdOnsetBias::apply(banOnly, 8, 2, 0, 400, temperature);
        TdOnsetBias::setEnabled(true);
        TdOnsetBias::apply(logits, 8, 2, 0, 400, temperature);

        const auto before = probabilities(banOnly, temperature);
        const auto after = probabilities(logits, temperature);
        EXPECT_NEAR(after[2], before[2], 1e-6) << "P(chord) unchanged at T " << temperature;
        EXPECT_TRUE(std::isinf(logits[50]));
        // forward gaps get exactly the shape f: log-ratio of 20 vs 200 cs moves by f(8,18) - f(8,198)
        const double logRatio = std::log(after[20] / after[200]) - std::log(before[20] / before[200]);
        EXPECT_NEAR(logRatio,
                    (TdOnsetBias::shift(8, 18) - TdOnsetBias::shift(8, 198)) / temperature, 1e-3);
        EXPECT_GT(after[12], before[12]) << "short forward gaps gain (from the renormalisation)";
        EXPECT_LT(after[300], before[300]) << "long ones lose";
    }
}

TEST(TdOnsetBias, offOrAutoOnlyAddsTheShortGapPenalty) {
    const LogitTdOn guard;
    std::vector<float> logits(20, 1.0f);
    TdOnsetBias::apply(logits, TemporalDensity::kAuto, 0, 0, 20, 1.0f, 2);
    for (size_t t = 0; t < logits.size(); ++t)
        EXPECT_FLOAT_EQ(logits[t], 1.0f + TdOnsetBias::shortGapPenalty(static_cast<int32_t>(t))) << t;
    std::vector<float> off(20, 1.0f);
    TdOnsetBias::setEnabled(false);
    TdOnsetBias::apply(off, 64, 0, 0, 20, 1.0f, 2);
    EXPECT_EQ(off, logits);
}

TEST(TdOnsetBias, chordShiftLandsOnCZeroOnly) {
    const LogitTdOn guard;
    std::vector<float> withChord(400, 0.0f);
    std::vector<float> noChord(400, 0.0f);
    TdOnsetBias::apply(withChord, 16, 10, 0, 400, 1.0f, 1);
    TdOnsetBias::apply(noChord, 16, 10, 0, 400, 1.0f, 0);
    EXPECT_NEAR(withChord[10] - noChord[10], TdOnsetBias::chordShift(16, 1), 1e-5);
    for (size_t t = 11; t < 400; ++t)
        ASSERT_FLOAT_EQ(withChord[t], noChord[t]) << t;
}

TEST(TdOnsetBias, chordsStopAtTenNotesEvenWhenOff) {
    const LogitTdOn guard;
    for (const bool on: {true, false}) {
        TdOnsetBias::setEnabled(on);
        std::vector<float> nine(40, 0.0f);
        TdOnsetBias::apply(nine, 64, 10, 0, 40, 1.0f, 9);
        EXPECT_TRUE(std::isfinite(nine[10])) << "a 10th note may join";
        std::vector<float> ten(40, 0.0f);
        TdOnsetBias::apply(ten, 64, 10, 0, 40, 1.0f, 10);
        EXPECT_TRUE(std::isinf(ten[10])) << "no 11th note";
        EXPECT_TRUE(std::isfinite(ten[20]));
    }
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();
    std::vector<float> forced(40, kNegInf);
    forced[10] = 0.0f; // the only legal onset
    TdOnsetBias::apply(forced, 64, 10, 0, 40, 1.0f, 12);
    EXPECT_TRUE(std::isfinite(forced[10])) << "a forced onset is kept";
}

TEST(TdOnsetBias, eachDrawLeavesASnapshotOfTheGapDistribution) {
    const LogitTdOn guard;
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();
    std::vector<float> logits(1000, 0.0f);
    for (int t = 0; t < 100; ++t)
        logits[static_cast<size_t>(t)] = kNegInf; // before the lower bound
    const auto previous = TdOnsetBias::latestSnapshot().sequence;

    TdOnsetBias::apply(logits, 64, 100, 0, 1000);
    const auto snap = TdOnsetBias::latestSnapshot();
    EXPECT_EQ(snap.sequence, previous + 1);
    EXPECT_TRUE(snap.shifted);
    EXPECT_EQ(snap.tdLevel, 64);
    // flat logits plus the short-gap penalty; gaps 0..399 shown and the rest in the tail
    EXPECT_NEAR(snap.before[6] / snap.before[0], 0.5f, 1e-4);
    EXPECT_LT(snap.before[3], 0.1f * snap.before[0]) << "short gap discouraged";
    EXPECT_GT(snap.beforeTail, 0.5f);
    EXPECT_NEAR(snap.after[0], snap.before[0], 1e-6) << "chord probability kept";
    EXPECT_GT(snap.after[6], snap.after[100]) << "the shift favours short gaps";
    float total = snap.afterTail;
    for (const float p: snap.after)
        total += p;
    EXPECT_NEAR(total, 1.0f, 1e-4);

    TdOnsetBias::apply(logits, TemporalDensity::kAuto, 100, 0, 1000);
    const auto idle = TdOnsetBias::latestSnapshot();
    EXPECT_FALSE(idle.shifted);
    EXPECT_EQ(idle.before, idle.after);
}

TEST(TdOnsetBias, snapshotFollowsTheTemperature) {
    const LogitTdOn guard;
    std::vector<float> logits(40, 0.0f);
    logits[20] = 1.0f;
    auto warmLogits = logits; // apply changes its input: each run gets a fresh copy
    TdOnsetBias::apply(warmLogits, TemporalDensity::kAuto, 0, 0, 40, 1.0f);
    const float warm = TdOnsetBias::latestSnapshot().before[20];
    auto coldLogits = logits;
    TdOnsetBias::apply(coldLogits, TemporalDensity::kAuto, 0, 0, 40, 0.5f);
    EXPECT_GT(TdOnsetBias::latestSnapshot().before[20], warm) << "colder = sharper";
}

TEST(OnsetGapsView, gapsAreBetweenSortedOnsetsOfTheLastNotes) {
    EXPECT_EQ(OnsetGapsView::gapsBetweenOnsets({300, 100, 100, 150}, 10),
              (std::vector<int32_t>{0, 50, 150}));
    EXPECT_EQ(OnsetGapsView::gapsBetweenOnsets({0, 10, 30, 60}, 3), (std::vector<int32_t>{20, 30}));
    EXPECT_TRUE(OnsetGapsView::gapsBetweenOnsets({5}, 10).empty());
}

} // namespace virtual_orch_test
