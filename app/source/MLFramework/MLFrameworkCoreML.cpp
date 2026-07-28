#ifdef ENABLE_COREML

#include <string>
#include <unordered_set>

#include "JordanAI/MLFramework/MLFrameworkCoreML.h"
#include "JordanAI/CoreMLWrapper.h"

MLFrameworkCoreML::MLFrameworkCoreML() : MLFramework() {}

MLFrameworkCoreML::~MLFrameworkCoreML() {
    model = nullptr;
}

auto MLFrameworkCoreML::getBuiltAccelerators () const -> std::unordered_set<MLFramework::ACCELERATOR> {
    return std::unordered_set<MLFramework::ACCELERATOR>{
        MLFramework::ACCELERATOR::COREML_COREML,
    };
}

auto MLFrameworkCoreML::getAvailableAccelerators () -> std::unordered_set<MLFramework::ACCELERATOR> {
    std::unordered_set<MLFramework::ACCELERATOR> availableAccels;
    for (const auto &accel : getBuiltAccelerators()) {
        switch (accel) {
            case MLFramework::ACCELERATOR::COREML_COREML:
                if (tryAcceleratorCoreML()) {
                    availableAccels.insert(MLFramework::ACCELERATOR::COREML_COREML);
                }
                break;
            default:
                throw std::runtime_error("Unexpected accelerator when testing available accelerators for CoreML.");
                break;
        }
    }
    #ifdef DEBUG
        for (const auto &name : getBuiltAccelerators()) {
            std::string accelName = MLFramework::ACCELERATOR_FRIENDLY_NAMES.at(name);
            DBG("CoreML built provider: " + juce::String(accelName));
        }
        for (const auto name : availableAccels) {
            std::string accelName = MLFramework::ACCELERATOR_FRIENDLY_NAMES.at(name);
            DBG("CoreML usable provider: " + juce::String(accelName));
        }
    #endif
    return availableAccels;
}

auto MLFrameworkCoreML::tryAcceleratorCoreML() const -> bool {
    // TODO: Check for CoreML usability
    return true;
}

void MLFrameworkCoreML::init(const std::string &modelPath, const ModelType &newModelType, MLFramework::ACCELERATOR accelerator) {
    switch (accelerator) {
        case MLFramework::ACCELERATOR::COREML_COREML:
            // Nothing to do
            break;
        default:
            throw std::runtime_error("Unsupported accelerator for CoreML.");
            break;
    }

    // Load the model
    model = std::shared_ptr<const void>(loadModel(modelPath.c_str()), closeModel);
}

std::vector<float> MLFrameworkCoreML::runModelAndGetLogits(std::vector<int32_t> &tokens, bool returnAllLogits) {
    if (returnAllLogits) {
        // TODO: Implement returnAllLogits for CoreML
        throw std::runtime_error("returnAllLogits not yet supported for CoreML.");
    }

    std::vector<float> logits(55028, 0.0F);
    std::vector<float> inputs = std::vector<float>(tokens.begin(), tokens.end());

    predictWith(model.get(), inputs, logits);

    return logits;
}

#endif
