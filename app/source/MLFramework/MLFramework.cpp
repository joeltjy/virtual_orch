#include <map>
#include <string>

#include "JordanAI/MLFramework/MLFramework.h"
#include "JordanAI/MLFramework/MLFrameworkCoreML.h"
#include "JordanAI/MLFramework/MLFrameworkGGML.h"
#include "JordanAI/MLFramework/MLFrameworkONNXRuntime.h"
#include "JordanAI/MLFramework/MLFrameworkTorch.h"

const std::map<MLFramework::ACCELERATOR, std::string> MLFramework::ACCELERATOR_FRIENDLY_NAMES = {
    {MLFramework::ACCELERATOR::COREML_COREML, "Apple CoreML"},
    {MLFramework::ACCELERATOR::GGML_CPU, "GGML CPU"},
    {MLFramework::ACCELERATOR::GGML_CUDA, "GGML NVIDIA CUDA"},
    {MLFramework::ACCELERATOR::GGML_METAL, "GGML Apple Metal"},
    {MLFramework::ACCELERATOR::GGML_SYCL, "GGML SYCL"},
    {MLFramework::ACCELERATOR::GGML_VULKAN, "GGML Vulkan"},
    {MLFramework::ACCELERATOR::ONNXRUNTIME_CPU, "ONNX Runtime CPU"},
    {MLFramework::ACCELERATOR::ONNXRUNTIME_TENSORRT, "ONNX Runtime NVIDIA TensorRT"},
    {MLFramework::ACCELERATOR::ONNXRUNTIME_CUDA, "ONNX Runtime NVIDIA CUDA"},
    {MLFramework::ACCELERATOR::TORCH_CUDA, "Torch NVIDIA CUDA"},
    {MLFramework::ACCELERATOR::TORCH_MPS, "Torch Apple MPS"},
    {MLFramework::ACCELERATOR::TORCH_CPU, "Torch CPU"}
};

const std::map<MLFramework::ACCELERATOR, std::string> MLFramework::ACCELERATOR_NAMES = {
    {MLFramework::ACCELERATOR::COREML_COREML, "COREML_COREML"},
    {MLFramework::ACCELERATOR::GGML_CPU, "GGML_CPU"},
    {MLFramework::ACCELERATOR::GGML_CUDA, "GGML_CUDA"},
    {MLFramework::ACCELERATOR::GGML_METAL, "GGML_METAL"},
    {MLFramework::ACCELERATOR::GGML_SYCL, "GGML_SYCL"},
    {MLFramework::ACCELERATOR::GGML_VULKAN, "GGML_VULKAN"},
    {MLFramework::ACCELERATOR::ONNXRUNTIME_CPU, "ONNXRUNTIME_CPU"},
    {MLFramework::ACCELERATOR::ONNXRUNTIME_TENSORRT, "ONNXRUNTIME_TENSORRT"},
    {MLFramework::ACCELERATOR::ONNXRUNTIME_CUDA, "ONNXRUNTIME_CUDA"},
    {MLFramework::ACCELERATOR::TORCH_CUDA, "TORCH_CUDA"},
    {MLFramework::ACCELERATOR::TORCH_MPS, "TORCH_MPS"},
    {MLFramework::ACCELERATOR::TORCH_CPU, "TORCH_CPU"}
};

const std::map<MLFramework::ACCELERATOR, std::string> MLFramework::ACCELERATOR_FILE_EXTENSIONS = {
    {MLFramework::ACCELERATOR::COREML_COREML, ".mlmodelc"},
    {MLFramework::ACCELERATOR::GGML_CPU, ".ggml"},
    {MLFramework::ACCELERATOR::GGML_CUDA, ".ggml"},
    {MLFramework::ACCELERATOR::GGML_METAL, ".ggml"},
    {MLFramework::ACCELERATOR::GGML_SYCL, ".ggml"},
    {MLFramework::ACCELERATOR::GGML_VULKAN, ".ggml"},
    {MLFramework::ACCELERATOR::ONNXRUNTIME_CPU, ".onnx"},
    {MLFramework::ACCELERATOR::ONNXRUNTIME_TENSORRT, ".onnx_trt"},
    {MLFramework::ACCELERATOR::ONNXRUNTIME_CUDA, ".onnx"},
    {MLFramework::ACCELERATOR::TORCH_CUDA, ".cuda_pt"},
    {MLFramework::ACCELERATOR::TORCH_MPS, ".mps_pt"},
    {MLFramework::ACCELERATOR::TORCH_CPU, ".pt"}
};

auto MLFramework::getAllBuiltAccelerators () -> std::unordered_set<MLFramework::ACCELERATOR> {
    std::unordered_set<MLFramework::ACCELERATOR> accelerators;
    #ifdef ENABLE_COREML
        MLFrameworkCoreML mlFrameworkCoreML;
        accelerators.merge(mlFrameworkCoreML.getBuiltAccelerators());
    #endif
    #ifdef ENABLE_GGML
        MLFrameworkGGML mlFrameworkGGML;
        accelerators.merge(mlFrameworkGGML.getBuiltAccelerators());
    #endif
    #ifdef ENABLE_ONNXRUNTIME
        MLFrameworkONNXRuntime mlFrameworkONNXRuntime;
        accelerators.merge(mlFrameworkONNXRuntime.getBuiltAccelerators());
    #endif
    #ifdef ENABLE_TORCH
        MLFrameworkTorch mlFrameworkTorch;
        accelerators.merge(mlFrameworkTorch.getBuiltAccelerators());
    #endif
    return accelerators;
}

auto MLFramework::getAllAvailableAccelerators () -> std::unordered_set<MLFramework::ACCELERATOR> {
    std::unordered_set<MLFramework::ACCELERATOR> availableAccels;
    #ifdef ENABLE_COREML
        MLFrameworkCoreML mlFrameworkCoreML;
        availableAccels.merge(mlFrameworkCoreML.getAvailableAccelerators());
    #endif
    #ifdef ENABLE_GGML
        MLFrameworkGGML mlFrameworkGGML;
        availableAccels.merge(mlFrameworkGGML.getAvailableAccelerators());
    #endif
    #ifdef ENABLE_ONNXRUNTIME
        MLFrameworkONNXRuntime mlFrameworkONNXRuntime;
        availableAccels.merge(mlFrameworkONNXRuntime.getAvailableAccelerators());
    #endif
    #ifdef ENABLE_TORCH
        MLFrameworkTorch mlFrameworkTorch;
        availableAccels.merge(mlFrameworkTorch.getAvailableAccelerators());
    #endif
    return availableAccels;
}
