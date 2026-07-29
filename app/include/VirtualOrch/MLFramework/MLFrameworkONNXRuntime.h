#pragma once
#ifdef ENABLE_ONNXRUNTIME

#include <string>
#include <unordered_set>

#include "VirtualOrch/MLFramework/MLFramework.h"
#include <onnxruntime_cxx_api.h>
#include <JuceHeader.h>

class MLFrameworkONNXRuntime : public MLFramework {
public:
    MLFrameworkONNXRuntime();

    ~MLFrameworkONNXRuntime() override;

    std::vector<float> runModelAndGetLogits(std::vector<int32_t> &tokens, bool returnAllLogits = false) override;

    auto getBuiltAccelerators () const -> std::unordered_set<MLFramework::ACCELERATOR> override;
    auto getAvailableAccelerators () -> std::unordered_set<MLFramework::ACCELERATOR> override;

    void init (const std::string &modelPath, const ModelType &newModelType, MLFramework::ACCELERATOR accelerator) override;

private:
    auto tryAcceleratorTensorRT() const -> bool;
    auto tryAcceleratorCUDA() const -> bool;
    auto tryAcceleratorCPU() const -> bool;

    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<std::string> allocatedInputNames;
    std::vector<std::string> allocatedOutputNames;

    std::unique_ptr<std::vector<int64_t> > pastShape;
    std::unique_ptr<std::vector<float> > emptyPast;
};
#endif
