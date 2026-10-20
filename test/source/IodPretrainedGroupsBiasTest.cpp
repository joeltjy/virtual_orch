#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <VirtualOrch/orchestration-models/IodPretrainedTypes.h>

namespace virtual_orch_test {
namespace {

TEST(IodPretrainedGroupsBias, PenalizesOverusedGroupsMask) {
    using namespace IodPretrainedTypes;

    std::vector<float> logits(static_cast<size_t>(comboVocabSize), 0.0f);
    // 30 notes all used strings-only mask 0b0001 == 1
    std::vector<int32_t> recent(30, 1);

    applyGroupsTokenLogitsBias(logits.data(), logits.size(), recent.data(), recent.size(),
                               /*tau*/ 1.0f, /*epsilon*/ groupsBiasEpsilon);

    const float logitStrings = logits[static_cast<size_t>(groupsTokenForMask(1))];
    const float logitWood = logits[static_cast<size_t>(groupsTokenForMask(0b0010))];
    EXPECT_LT(logitStrings, logitWood);
    EXPECT_TRUE(std::isfinite(logitStrings));
    EXPECT_TRUE(std::isfinite(logitWood));

    // Unseen mask still finite (ε smoothing): π = ε / (30 + 15ε)
    const float piUnseen = groupsBiasEpsilon / (30.0f + 15.0f * groupsBiasEpsilon);
    EXPECT_NEAR(logitWood, -std::log(piUnseen), 1e-5f);
}

TEST(IodPretrainedGroupsBias, EmptyHistoryNoOp) {
    using namespace IodPretrainedTypes;
    std::vector<float> logits(static_cast<size_t>(comboVocabSize), 1.5f);
    applyGroupsTokenLogitsBias(logits.data(), logits.size(), nullptr, 0);
    EXPECT_FLOAT_EQ(logits[static_cast<size_t>(groupsTokenForMask(1))], 1.5f);
}

} // namespace
} // namespace virtual_orch_test
