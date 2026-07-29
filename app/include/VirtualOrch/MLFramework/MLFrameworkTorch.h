#pragma once
#ifdef ENABLE_TORCH

#include <string>
#include <unordered_set>

#include <JuceHeader.h>

#include "VirtualOrch/MLFramework/MLFramework.h"

#include <torch/script.h>

class MLFrameworkTorch : public MLFramework {
public:
    MLFrameworkTorch();

    ~MLFrameworkTorch() override;

    std::vector<float> runModelAndGetLogits(std::vector<int32_t> &tokens, bool returnAllLogits = false) override;

    auto getBuiltAccelerators () const -> std::unordered_set<MLFramework::ACCELERATOR> override;
    auto getAvailableAccelerators () -> std::unordered_set<MLFramework::ACCELERATOR> override;

    void init (const std::string &modelPath, const ModelType &newModelType, MLFramework::ACCELERATOR accelerator) override;

private:
    auto tryAcceleratorCUDA() const -> bool;
    auto tryAcceleratorMPS() const -> bool;
    auto tryAcceleratorCPU() const -> bool;

    std::unique_ptr<torch::jit::script::Module> module;
};

#endif
