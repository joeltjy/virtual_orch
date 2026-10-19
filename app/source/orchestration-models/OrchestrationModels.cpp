#include "VirtualOrch/orchestration-models/OrchestrationModels.h"

#include "VirtualOrch/ProjectPaths.h"
#include "VirtualOrch/orchestration-models/InstrumentCombinations.h"
#include "VirtualOrch/orchestration-models/IodPretrained.h"
#include "VirtualOrch/orchestration-models/IodPretrainedTD.h"
#include "VirtualOrch/orchestration-models/TestModel.h"

auto orchestrationModelNames() -> std::vector<std::string> {
    return {"TestModel", "InstrumentCombinations", "IodPretrained", "IodPretrainedTD"};
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
        if (name == "IodPretrainedTD") {
            auto model = std::make_unique<IodPretrainedTD>();
            const std::string file = "td_grpo_unk_T14_dummy_from_td_explicit_given_ckpt18000.onnx";
            const juce::File candidates[] = {
                onnxExportDir().getChildFile(file),
                onnxExportFallbackDir().getChildFile(file),
                juce::File::getSpecialLocation(juce::File::userDesktopDirectory)
                    .getChildFile("onnx_export_101826")
                    .getChildFile(file),
            };
            for (const auto &onnx: candidates) {
                if (! onnx.existsAsFile())
                    continue;
                model->init(onnx.getFullPathName().toRawUTF8());
                if (model->isModelLoaded())
                    return model;
                DBG("IodPretrainedTD init failed for " + onnx.getFullPathName());
            }
            DBG("IodPretrainedTD model not found");
            return nullptr;
        }
    } catch (const Ort::Exception &exception) {
        DBG("createOrchestrationModel failed: " + juce::String(exception.what()));
        return nullptr;
    }

    return nullptr;
}
