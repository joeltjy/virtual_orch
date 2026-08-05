#include "VirtualOrch/orchestration-models/OrchestrationModels.h"

#include "VirtualOrch/orchestration-models/TestModel.h"

auto orchestrationModelNames() -> std::vector<std::string> {
    return {"TestModel"};
}

auto createOrchestrationModel(const std::string &name) -> std::unique_ptr<OrchestrationModel> {
    if (name == "TestModel") {
        auto model = std::make_unique<TestModel>();
        const auto modelPath =
            juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
                .getChildFile("virtual-orch")
                .getChildFile("Models")
                .getChildFile("dumb_pitch_split_instrument.onnx");
        model->init(modelPath.getFullPathName().toRawUTF8());
        return model;
    }
    return nullptr;
}
