#include <gtest/gtest.h>

#include "VirtualOrch/reduction/DenseDurationBias.h"

#include <vector>

using namespace DenseDurationBias;

namespace {

// orchestra (instrument 1) notes as dense quads: (onset, duration token, note token, velocity token)
auto quads(const std::vector<std::pair<int32_t, int32_t>> &onsetAndDuration) -> std::vector<int32_t> {
    std::vector<int32_t> q;
    for (const auto &[onset, dur]: onsetAndDuration) {
        q.push_back(onset);
        q.push_back(static_cast<int32_t>(DenseVocab::DurOffset) + dur);
        q.push_back(static_cast<int32_t>(DenseVocab::NoteOffset) + DenseConfig::MaxPitch + 60);
        q.push_back(static_cast<int32_t>(DenseVocab::VelocityOffset) + 64);
    }
    return q;
}

auto logitOf(const std::vector<float> &l, int32_t cs) -> float {
    return l[DenseVocab::DurOffset + static_cast<size_t>(cs)];
}

} // namespace

TEST(CoarseDurationBias, BinsAreRoughlyLogSpaced) {
    EXPECT_EQ(coarseBin(10), 0);
    EXPECT_EQ(coarseBin(14), 0);
    EXPECT_EQ(coarseBin(15), 1);
    EXPECT_EQ(coarseBin(35), 3);  // 30..39
    EXPECT_EQ(coarseBin(45), 4);  // 40..59
    EXPECT_EQ(coarseBin(500), CoarseBins - 1);
}

TEST(CoarseDurationBias, AUsedBinIsHeldBackAsAWhole) {
    // four notes of 40 / 45 / 50 / 55 cs: all in the 40..59 bin
    const auto history = quads({{100, 40}, {130, 45}, {160, 50}, {190, 55}});
    const auto stats = collectCoarseDurationWindowAt(history, 200, 1);
    EXPECT_EQ(stats.noteCount, 4);
    EXPECT_EQ(stats.lastBin, coarseBin(40));
    std::vector<float> l(DenseVocab::VocabSize, 0.0f);
    applyCoarseDurationLogitsBias(l, stats, 0.5f, 0.0f);
    // every duration in the used bin gets the same logit, also the ones never played (e.g. 35 is another bin)
    EXPECT_FLOAT_EQ(logitOf(l, 40), logitOf(l, 45));
    EXPECT_FLOAT_EQ(logitOf(l, 40), logitOf(l, 55));
    // and less than a neighbouring unused bin by a clear margin (a 5 cs step off is no escape)
    EXPECT_GT(logitOf(l, 35) - logitOf(l, 40), 2.0f);
    EXPECT_GT(logitOf(l, 60) - logitOf(l, 55), 2.0f);
    // unused bins are all alike
    EXPECT_FLOAT_EQ(logitOf(l, 10), logitOf(l, 200));
}

TEST(CoarseDurationBias, TauPrimeHoldsBackARepeatedMove) {
    // 20 -> 30 -> 20 -> 30: moves +1, -1, +1 between bins (20..29 is bin 2, 30..39 bin 3), no stays
    const auto moving = quads({{100, 20}, {130, 30}, {160, 20}, {190, 30}});
    const auto s1 = collectCoarseDurationWindowAt(moving, 200, 1);
    std::vector<float> a(DenseVocab::VocabSize, 0.0f);
    applyCoarseDurationLogitsBias(a, s1, 0.0f, 0.5f);
    // from 30 (bin 3): back to 20 (move -1) was made twice already; staying (move 0) never
    EXPECT_LT(logitOf(a, 20), logitOf(a, 30));
    // one note per bin with no stays: staying put is as new as any unmade move
    EXPECT_FLOAT_EQ(logitOf(a, 30), logitOf(a, 100));

    // the same bin four times: staying (move 0) is held back
    const auto staying = quads({{100, 40}, {130, 45}, {160, 50}, {190, 55}});
    const auto s2 = collectCoarseDurationWindowAt(staying, 200, 1);
    std::vector<float> b(DenseVocab::VocabSize, 0.0f);
    applyCoarseDurationLogitsBias(b, s2, 0.0f, 0.5f);
    EXPECT_GT(logitOf(b, 100) - logitOf(b, 45), 2.0f);
}

TEST(CoarseDurationBias, OnlyTheLastFiveSecondsAndTheInstrumentCount) {
    auto history = quads({{100, 40}});
    // a piano note (instrument 0) and one out of the window
    history.insert(history.end(), {300, static_cast<int32_t>(DenseVocab::DurOffset) + 100,
                                   static_cast<int32_t>(DenseVocab::NoteOffset) + 60,
                                   static_cast<int32_t>(DenseVocab::VelocityOffset) + 64});
    const auto stats = collectCoarseDurationWindowAt(history, 650, 1);
    EXPECT_EQ(stats.noteCount, 0); // 100 is older than 650 - 500; the piano note is another instrument
    std::vector<float> l(DenseVocab::VocabSize, 0.0f);
    applyCoarseDurationLogitsBias(l, stats, 0.5f, 0.5f);
    EXPECT_FLOAT_EQ(logitOf(l, 40), 0.0f);
}
