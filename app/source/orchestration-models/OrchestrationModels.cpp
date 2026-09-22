#include "VirtualOrch/orchestration-models/OrchestrationModels.h"

#include "VirtualOrch/ProjectPaths.h"
#include "VirtualOrch/orchestration-models/InstrumentCombinations.h"
#include "VirtualOrch/orchestration-models/IodPretrained.h"
#include "VirtualOrch/orchestration-models/TestModel.h"

auto orchestrationModelNames() -> std::vector<std::string> {
    return {"TestModel", "InstrumentCombinations", "IodPretrained"};
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
            const auto onnx =
                documentsModelsDir().getChildFile("ice20_causal_lora_s2_checkpoint-27000.onnx");
            const auto path = onnx.getFullPathName();
            if (! onnx.existsAsFile()) {
                DBG("InstrumentCombinations model missing at " + path);
                return nullptr;
            }
            model->init(path.toRawUTF8());
            return model;
        }
        if (name == "IodPretrained") {
            auto model = std::make_unique<IodPretrained>();
            auto onnx = onnxExportDir().getChildFile("instrument_octave_predictions.onnx");
            if (! onnx.existsAsFile())
                onnx = onnxExportFallbackDir().getChildFile("instrument_octave_predictions.onnx");
            const auto path = onnx.getFullPathName();
            if (! onnx.existsAsFile()) {
                DBG("IodPretrained model missing at " + path);
                return nullptr;
            }
            model->init(path.toRawUTF8());
            if (! model->isModelLoaded()) {
                DBG("IodPretrained init failed for " + path);
                return nullptr;
            }
            return model;
        }
    } catch (const Ort::Exception &exception) {
        DBG("createOrchestrationModel failed: " + juce::String(exception.what()));
        return nullptr;
    }

    return nullptr;
}
