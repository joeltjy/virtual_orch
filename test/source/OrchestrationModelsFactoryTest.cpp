#include <gtest/gtest.h>

#include <algorithm>

#include <VirtualOrch/orchestration-models/OrchestrationModels.h>

namespace virtual_orch_test {
namespace {

TEST(OrchestrationModelsFactory, NamesIncludeIodPretrained) {
    const auto names = orchestrationModelNames();
    EXPECT_NE(std::find(names.begin(), names.end(), "IodPretrained"), names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "InstrumentCombinations"), names.end());
}

TEST(OrchestrationModelsFactory, CreateIodPretrainedLoadsOrNull) {
    auto model = createOrchestrationModel("IodPretrained");
    // Missing ONNX or CUDA-only graph → nullptr is acceptable; when present, name matches.
    if (model != nullptr) {
        EXPECT_EQ(model->getName(), "IodPretrained");
    }
}

} // namespace
} // namespace virtual_orch_test
