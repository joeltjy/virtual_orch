#include <gtest/gtest.h>

#include <vector>

#include <VirtualOrch/reduction/ReductionTransformerV1.h>
#include <VirtualOrch/reduction/ReductionTransformerV2.h>

namespace virtual_orch_test {
namespace {

/** Build dense inputData (time, durToken, noteToken, velToken) from onset/duration pairs. */
auto denseHistory(const std::vector<std::pair<int32_t, int32_t>> &onsetAndDurationCs)
    -> std::vector<int32_t> {
    std::vector<int32_t> data;
    data.reserve(onsetAndDurationCs.size() * 4);
    for (const auto &[onset, duration]: onsetAndDurationCs) {
        data.push_back(onset);
        data.push_back(static_cast<int32_t>(DenseVocab::DurOffset) + duration);
        data.push_back(static_cast<int32_t>(DenseVocab::NoteOffset) + 60);
        data.push_back(static_cast<int32_t>(DenseVocab::VelocityOffset) + 100);
    }
    return data;
}

} // namespace

TEST(ReductionTransformerV1, FieldSamplingParameters) {
    ModelConfig config;
    ReductionTransformerV1 dense(config);

    // Onset / duration / note: top-p 0.98, T=0.5. Velocity: top-p 0.3, T=1.
    for (const int step: {0, 1, 2}) {
        const auto sampling = dense.samplingForStep(step);
        EXPECT_FLOAT_EQ(sampling.topP, DenseSampling::OnsetTopP) << "step " << step;
        EXPECT_FLOAT_EQ(sampling.temperature, DenseSampling::OnsetTemperature) << "step " << step;
    }
    {
        const auto sampling = dense.samplingForStep(3);
        EXPECT_FLOAT_EQ(sampling.topP, DenseSampling::VelocityTopP);
        EXPECT_FLOAT_EQ(sampling.temperature, DenseSampling::VelocityTemperature);
    }

    // Steps index into a repeating 4-wide event (step 5 → duration).
    EXPECT_FLOAT_EQ(dense.samplingForStep(5).topP, DenseSampling::DurationTopP);

    dense.setDurationTemperature(0.25f);
    EXPECT_FLOAT_EQ(dense.samplingForStep(1).temperature, 0.25f);
}

TEST(ReductionTransformerV1, DurationCeilingUsesFloorWhenHistoryEmpty) {
    // No notes: the percentile falls back to the 40 cs floor → 4 * 40 = 160.
    EXPECT_EQ(ReductionTransformerV1::durationCeilingCs({}), 160);
}

TEST(ReductionTransformerV1, DurationCeilingFloorAppliesToShortMaterial) {
    // p75 of {10, 10, 10} is 10, below the 40 cs floor.
    const auto data = denseHistory({{0, 10}, {5, 10}, {10, 10}});
    EXPECT_EQ(ReductionTransformerV1::durationCeilingCs(data), 160);
}

TEST(ReductionTransformerV1, DurationCeilingIgnoresOnsetSpacing) {
    // Identical durations, wildly different spacing → same ceiling (4 * 200).
    const auto dense = denseHistory({{0, 200}, {50, 200}, {100, 200}, {150, 200}});
    const auto sparse = denseHistory({{0, 200}, {900, 200}, {1800, 200}, {2700, 200}});
    EXPECT_EQ(ReductionTransformerV1::durationCeilingCs(dense), 800);
    EXPECT_EQ(ReductionTransformerV1::durationCeilingCs(sparse), 800);
}

TEST(ReductionTransformerV1, DurationCeilingUsesSeventyFifthPercentileNotMedian) {
    // Sorted: 10,10,10,10,10,10,10,10,10,10,10,300,300,300,300 (15 notes).
    // Median would be 10; nearest-rank p75 is the 12th smallest = 300 → 1200.
    std::vector<std::pair<int32_t, int32_t>> notes;
    for (int32_t i = 0; i < 11; ++i)
        notes.emplace_back(i * 50, 10);
    for (int32_t i = 0; i < 4; ++i)
        notes.emplace_back(600 + i * 50, 300);

    const auto data = denseHistory(notes);
    EXPECT_EQ(ReductionTransformerV1::durationCeilingCs(data), 1200);
}

TEST(ReductionTransformerV1, DurationCeilingIgnoresNotesBeyondFifteen) {
    // Five stale long notes then fifteen short ones: only the recent window counts.
    std::vector<std::pair<int32_t, int32_t>> notes;
    for (int32_t i = 0; i < 5; ++i)
        notes.emplace_back(i * 1000, 900); // would otherwise allow a huge ceiling
    for (int32_t i = 0; i < 15; ++i)
        notes.emplace_back(5000 + i * 50, 50);

    const auto data = denseHistory(notes);
    EXPECT_EQ(ReductionTransformerV1::durationCeilingCs(data), 200);
}

TEST(ReductionTransformerV1, DurationCeilingUnaffectedByChordedOnsets) {
    // Four notes sharing an onset no longer collapse the ceiling.
    const auto data = denseHistory({{100, 200}, {100, 200}, {100, 200}, {100, 200}});
    EXPECT_EQ(ReductionTransformerV1::durationCeilingCs(data), 800);
}

TEST(ReductionTransformerV2, FieldSamplingDefaultsMatchUniform) {
    ModelConfig config;
    ReductionTransformerV2 dense(config);

    for (const int step: {0, 1, 2}) {
        const auto sampling = dense.samplingForStep(step);
        EXPECT_FLOAT_EQ(sampling.topP, DenseSampling::OnsetTopP) << "step " << step;
        EXPECT_FLOAT_EQ(sampling.temperature, DenseSampling::OnsetTemperature) << "step " << step;
    }
    {
        const auto sampling = dense.samplingForStep(3);
        EXPECT_FLOAT_EQ(sampling.topP, DenseSampling::VelocityTopP);
        EXPECT_FLOAT_EQ(sampling.temperature, DenseSampling::VelocityTemperature);
    }
}

} // namespace virtual_orch_test
