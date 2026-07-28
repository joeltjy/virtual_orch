#pragma once
#ifdef ENABLE_COREML

#include <string>
#include <unordered_set>

#include <JuceHeader.h>

#include "JordanAI/MLFramework/MLFramework.h"

class MLFrameworkCoreML : public MLFramework {
public:
    MLFrameworkCoreML();

    ~MLFrameworkCoreML() override;

    std::vector<float> runModelAndGetLogits(std::vector<int32_t> &tokens, bool returnAllLogits = false) override;

    auto getBuiltAccelerators () const -> std::unordered_set<MLFramework::ACCELERATOR> override;
    auto getAvailableAccelerators () -> std::unordered_set<MLFramework::ACCELERATOR> override;

    void init (const std::string &modelPath, const ModelType &newModelType, MLFramework::ACCELERATOR accelerator) override;

private:
    auto tryAcceleratorCoreML() const -> bool;

    std::shared_ptr<const void> model;
};

#endif
