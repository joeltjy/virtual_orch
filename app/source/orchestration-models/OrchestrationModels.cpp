#include "VirtualOrch/orchestration-models/OrchestrationModels.h"

#include "VirtualOrch/ProjectPaths.h"
#include "VirtualOrch/orchestration-models/InstrumentCombinations.h"
#include "VirtualOrch/orchestration-models/TestModel.h"

auto orchestrationModelNames() -> std::vector<std::string> {
    return {"TestModel", "InstrumentCombinations"};
}

auto createOrchestrationModel(const std::string &name) -> std::unique_ptr<OrchestrationModel> {
    const auto modelsDir = projectModelsDir();

    try {
        if (name == "TestModel") {
            auto model = std::make_unique<TestModel>();
            const auto path = modelsDir.getChildFile("dumb_pitch_split_instrument.onnx").getFullPathName();
            model->init(path.toRawUTF8());
            return model;
        }
        if (name == "InstrumentCombinations") {
            auto model = std::make_unique<InstrumentCombinations>();
            const auto path = modelsDir.getChildFile("instrument_combinations.onnx").getFullPathName();
            if (! modelsDir.getChildFile("instrument_combinations.onnx").existsAsFile()) {
                DBG("InstrumentCombinations model missing at " + path);
                return nullptr;
            }
            model->init(path.toRawUTF8());
            return model;
        }
    } catch (const Ort::Exception &exception) {
        DBG("createOrchestrationModel failed: " + juce::String(exception.what()));
        return nullptr;
    }

    return nullptr;
}
