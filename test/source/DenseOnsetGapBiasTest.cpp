#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <VirtualOrch/reduction/DenseOnsetGapBias.h>
#include <VirtualOrch/reduction/DenseTypes.h>

namespace virtual_orch_test {
namespace {

auto denseNote(int32_t onset) -> std::vector<int32_t> {
    return {onset,
            static_cast<int32_t>(DenseVocab::DurOffset) + 50,
            static_cast<int32_t>(DenseVocab::NoteOffset) + 60,
            static_cast<int32_t>(DenseVocab::VelocityOffset) + 100};
}

auto appendNote(std::vector<int32_t> &data, int32_t onset) -> void {
    const auto n = denseNote(onset);
    data.insert(data.end(), n.begin(), n.end());
}

} // namespace

TEST(DenseOnsetGapBias, TracksInRangeGapsOnly) {
    std::vector<int32_t> data;
    appendNote(data, 100);
    appendNote(data, 100); // gap 0 — out of range
    appendNote(data, 105); // gap 5 — out of range
    appendNote(data, 125); // gap 20
    appendNote(data, 145); // gap 20

    const auto stats = DenseOnsetGapBias::collectOnsetGapWindowAt(data, 145);
    EXPECT_EQ(stats.noteCount, 5);
    EXPECT_EQ(stats.lastOnsetCs, 145);
    EXPECT_EQ(stats.inRangeGapPairs, 2);
    EXPECT_EQ(stats.uniqueGaps, (std::vector<int32_t>{20}));
    // Unseen in-range bins still get GapEpsilon (never zero).
    EXPECT_GT(stats.sGap[DenseOnsetGapBias::gapIndex(10)], 0.0f);
    EXPECT_GT(stats.sGap[DenseOnsetGapBias::gapIndex(20)],
              stats.sGap[DenseOnsetGapBias::gapIndex(10)]);
}

TEST(DenseOnsetGapBias, BiasPenalizesFrequentGap) {
    std::vector<int32_t> data;
    // Many gap-10 steps ending at 200.
    for (int i = 0; i < 10; ++i)
        appendNote(data, 100 + i * 10);
    // One gap-50 at the end.
    appendNote(data, 250);

    const auto stats = DenseOnsetGapBias::collectOnsetGapWindowAt(data, 250);
    ASSERT_EQ(stats.lastOnsetCs, 250);

    std::vector<float> logits(DenseVocab::VocabSize, 0.0f);
    // Relative times: offset 0 so absolute == token index.
    DenseOnsetGapBias::applyOnsetLogitsBias(logits, stats, 1.0f, 0);

    // Candidate gap 10 → onset 260; gap 50 → onset 300. Frequent gap 10 gets
    // a smaller Menon boost than rare gap 50.
    EXPECT_LT(logits[260], logits[300]);
}

TEST(DenseOnsetGapBias, OutOfRangeGapUnchanged) {
    std::vector<int32_t> data;
    appendNote(data, 100);
    appendNote(data, 120); // gap 20 in range

    const auto stats = DenseOnsetGapBias::collectOnsetGapWindowAt(data, 120);
    std::vector<float> logits(DenseVocab::VocabSize, 0.0f);
    DenseOnsetGapBias::applyOnsetLogitsBias(logits, stats, 1.0f, 0);

    // Gap 5 from lastOnset 120 → onset 125 is below GapLoCs.
    EXPECT_FLOAT_EQ(logits[125], 0.0f);
    // Gap 101 → onset 221 is above GapHiCs.
    EXPECT_FLOAT_EQ(logits[221], 0.0f);
    // Gap 20 → onset 140 is in range.
    EXPECT_NE(logits[140], 0.0f);
}

TEST(DenseOnsetGapBias, EpsilonOnEmptyWindowBins) {
    std::vector<int32_t> data;
    appendNote(data, 100); // single note → no gap pairs, but bins still smoothed
    const auto stats = DenseOnsetGapBias::collectOnsetGapWindowAt(data, 100);
    EXPECT_EQ(stats.inRangeGapPairs, 0);
    for (int g = DenseOnsetGapBias::GapLoCs; g <= DenseOnsetGapBias::GapHiCs; ++g) {
        EXPECT_GT(stats.sGap[DenseOnsetGapBias::gapIndex(g)], 0.0f);
    }
}

TEST(DenseOnsetGapBias, TauFixedAtBase) {
    DenseOnsetGapBias::OnsetGapTauScheduler sched;
    EXPECT_FLOAT_EQ(sched.evaluate(0, {}).tau, DenseOnsetGapBias::TauBase);
    EXPECT_FLOAT_EQ(sched.evaluate(500, {10, 20}).tau, DenseOnsetGapBias::TauBase);
}

} // namespace virtual_orch_test
