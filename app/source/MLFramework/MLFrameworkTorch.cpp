#ifdef ENABLE_TORCH

#include <string>
#include <unordered_set>

#include "JordanAI/MLFramework/MLFrameworkTorch.h"

MLFrameworkTorch::MLFrameworkTorch() : MLFramework() {}

MLFrameworkTorch::~MLFrameworkTorch() {
    module = nullptr;
}

auto MLFrameworkTorch::getBuiltAccelerators () const -> std::unordered_set<MLFramework::ACCELERATOR> {
    return std::unordered_set<MLFramework::ACCELERATOR>{
        #if defined ENABLE_TORCH_CUDA
            MLFramework::ACCELERATOR::TORCH_CUDA,
        #elif defined ENABLE_TORCH_MPS
            MLFramework::ACCELERATOR::TORCH_MPS,
        #elif defined ENABLE_TORCH_CPU
            MLFramework::ACCELERATOR::TORCH_CPU,
        #endif
    };
}

auto MLFrameworkTorch::getAvailableAccelerators () -> std::unordered_set<MLFramework::ACCELERATOR> {
    std::unordered_set<MLFramework::ACCELERATOR> availableAccels;
    for (const auto &accel : getBuiltAccelerators()) {
        switch (accel) {
            case MLFramework::ACCELERATOR::TORCH_CUDA:
                if (tryAcceleratorCUDA()) {
                    availableAccels.insert(MLFramework::ACCELERATOR::TORCH_CUDA);
                }
                break;
            case MLFramework::ACCELERATOR::TORCH_MPS:
                if (tryAcceleratorMPS()) {
                    availableAccels.insert(MLFramework::ACCELERATOR::TORCH_MPS);
                }
                break;
            case MLFramework::ACCELERATOR::TORCH_CPU:
                if (tryAcceleratorCPU()) {
                    availableAccels.insert(MLFramework::ACCELERATOR::TORCH_CPU);
                }
                break;
            default:
                throw std::runtime_error("Unexpected accelerator when testing available accelerators for Torch.");
                break;
        }
    }
    #ifdef DEBUG
        for (const auto &name : getBuiltAccelerators()) {
            std::string accelName = MLFramework::ACCELERATOR_FRIENDLY_NAMES.at(name);
            DBG("Torch built provider: " + juce::String(accelName));
        }
        for (const auto name : availableAccels) {
            std::string accelName = MLFramework::ACCELERATOR_FRIENDLY_NAMES.at(name);
            DBG("Torch usable provider: " + juce::String(accelName));
        }
    #endif
    return availableAccels;
}

auto MLFrameworkTorch::tryAcceleratorCUDA() const -> bool {
    // TODO: Check for CUDA usability
    return true;
}

auto MLFrameworkTorch::tryAcceleratorMPS() const -> bool {
    // TODO: Check for MPS usability
    return true;
}

auto MLFrameworkTorch::tryAcceleratorCPU() const -> bool {
    // TODO: Check for CPU usability
    return true;
}

void MLFrameworkTorch::init(const std::string &modelPath, const ModelType &newModelType, MLFramework::ACCELERATOR accelerator) {
    // Load the model
    switch (accelerator) {
        #if defined ENABLE_TORCH_CUDA
        case MLFramework::ACCELERATOR::TORCH_CUDA:
            module = std::make_unique<torch::jit::script::Module>(torch::jit::load(modelPath, torch::kCUDA));
            break;
        #elif defined ENABLE_TORCH_MPS
        case MLFramework::ACCELERATOR::TORCH_MPS:
            module = std::make_unique<torch::jit::script::Module>(torch::jit::load(modelPath, torch::kMPS));
            module->to(torch::kMPS); // Unsure why we need to move to MPS twice
            break;
        #elif defined ENABLE_TORCH_CPU
        case MLFramework::ACCELERATOR::TORCH_CPU:
            module = std::make_unique<torch::jit::script::Module>(torch::jit::load(modelPath));
            break;
        #endif
        default:
            throw std::runtime_error("Unsupported accelerator for Torch.");
            break;
    }
}

std::vector<float> MLFrameworkTorch::runModelAndGetLogits(std::vector<int32_t> &tokens, bool returnAllLogits) {
    try {
        auto input = torch::from_blob(tokens.data(), {1, static_cast<int64_t>(tokens.size())},
                                      torch::kInt32);
#if defined ENABLE_TORCH_CUDA
        input = input.to(torch::kCUDA);
#elif defined ENABLE_TORCH_MPS
        input = input.to(torch::kMPS);
#endif
        auto logits = module->forward({input}).toTuple()->elements()[0].toTensor().to(torch::kCPU);
        if (returnAllLogits) {
            std::vector<float> output(logits.data_ptr<float>(), logits.data_ptr<float>() + logits.numel());
            return output;
        }
        std::vector<float> output(logits.data_ptr<float>() + (Vocab::VocabSize * (tokens.size() - 1)),
                                  logits.data_ptr<float>() + logits.numel());
        return output;
    } catch (const c10::Error &e) {
        std::cerr << "Error running the model: " << e.what() << std::endl;
    }
    return {};
}

#endif
