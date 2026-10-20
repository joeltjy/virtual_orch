#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include <VirtualOrch/MusicToken.h>

namespace virtual_orch_test {

TEST(SampleTopP, NucleusUsesUntemperedMassThenTempers) {
    // Two logits far apart: untempered mass is almost all on token 0.
    // With small p the nucleus is {0} only, so every sample is 0 even at high T.
    std::vector<float> scores{10.0f, 0.0f, -std::numeric_limits<float>::infinity()};
    for (int i = 0; i < 20; ++i)
        EXPECT_EQ(sampleTopP(scores, 0.5f, 5.0f), 0);
}

TEST(SampleTopP, TemperatureZeroIsArgmaxOnNucleus) {
    std::vector<float> scores{1.0f, 3.0f, 2.0f};
    EXPECT_EQ(sampleTopP(scores, 1.0f, 0.0f), 1);
}

TEST(SampleTopP, FullPStillSamplesFromTemperedDistribution) {
    // Equal logits → either token is fine; just ensure it returns a finite id.
    std::vector<float> scores{0.0f, 0.0f};
    const int32_t token = sampleTopP(scores, 1.0f, 1.0f);
    EXPECT_TRUE(token == 0 || token == 1);
}

} // namespace virtual_orch_test
