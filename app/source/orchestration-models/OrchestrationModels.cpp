#include "VirtualOrch/orchestration-models/OrchestrationModels.h"

#include "VirtualOrch/ProjectPaths.h"
#include "VirtualOrch/orchestration-models/InstrumentCombinations.h"
#include "VirtualOrch/orchestration-models/TestModel.h"

auto orchestrationModelNames() -> std::vector<std::string> {
    return {"TestModel", "InstrumentCombinations"};
}

auto createOrchestrationModel(const std::string &name) -> std::unique_ptr<OrchestrationModel> {
    const auto modelsDir = projectModelsDir();

    if (name == "TestModel") {
        auto model = std::make_unique<TestModel>();
        const auto modelPath = modelsDir.getChildFile("dumb_pitch_split_instrument.onnx");
        model->init(modelPath.getFullPathName().toRawUTF8());
        return model;
    }
    if (name == "InstrumentCombinations") {
        auto model = std::make_unique<InstrumentCombinations>();
        const auto modelPath = modelsDir.getChildFile("instrument_combinations.onnx");
        model->init(modelPath.getFullPathName().toRawUTF8());
        return model;
    }
    return nullptr;
}
