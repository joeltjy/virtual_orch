#ifdef ENABLE_ONNXRUNTIME
#include <numeric>
#include <string>
#include <unordered_set>

#include "VirtualOrch/MLFramework/MLFrameworkONNXRuntime.h"

MLFrameworkONNXRuntime::MLFrameworkONNXRuntime() : MLFramework() {}

MLFrameworkONNXRuntime::~MLFrameworkONNXRuntime() {
}

auto MLFrameworkONNXRuntime::getBuiltAccelerators () const -> std::unordered_set<MLFramework::ACCELERATOR> {
    return std::unordered_set<MLFramework::ACCELERATOR>{
        #if defined ENABLE_ONNXRUNTIME_TENSORRT
            MLFramework::ACCELERATOR::ONNXRUNTIME_TENSORRT,
        #endif
        #if defined ENABLE_ONNXRUNTIME_CUDA
            MLFramework::ACCELERATOR::ONNXRUNTIME_CUDA,
        #endif
        #if defined ENABLE_ONNXRUNTIME_CPU
            MLFramework::ACCELERATOR::ONNXRUNTIME_CPU,
        #endif
    };
}

auto MLFrameworkONNXRuntime::getAvailableAccelerators () -> std::unordered_set<MLFramework::ACCELERATOR> {
    std::unordered_set<MLFramework::ACCELERATOR> availableAccels;
    for (const auto &accel : getBuiltAccelerators()) {
        switch (accel) {
            case MLFramework::ACCELERATOR::ONNXRUNTIME_TENSORRT:
                if (tryAcceleratorTensorRT()) {
                    availableAccels.insert(MLFramework::ACCELERATOR::ONNXRUNTIME_TENSORRT);
                }
                break;
            case MLFramework::ACCELERATOR::ONNXRUNTIME_CUDA:
                if (tryAcceleratorCUDA()) {
                    availableAccels.insert(MLFramework::ACCELERATOR::ONNXRUNTIME_CUDA);
                }
                break;
            case MLFramework::ACCELERATOR::ONNXRUNTIME_CPU:
                if (tryAcceleratorCPU()) {
                    availableAccels.insert(MLFramework::ACCELERATOR::ONNXRUNTIME_CPU);
                }
                break;
            default:
                throw std::runtime_error("Unexpected accelerator when testing available accelerators for ONNX Runtime.");
                break;
        }
    }
    #ifdef DEBUG
        for (const auto &name : getBuiltAccelerators()) {
            std::string accelName = MLFramework::ACCELERATOR_FRIENDLY_NAMES.at(name);
            DBG("ONNX Runtime built provider: " + juce::String(accelName));
        }
        for (const auto &name :  Ort::GetAvailableProviders()) {
            DBG("ONNX Runtime available provider: " + juce::String(name));
        }
        for (const auto name : availableAccels) {
            std::string accelName = MLFramework::ACCELERATOR_FRIENDLY_NAMES.at(name);
            DBG("ONNX Runtime usable provider: " + juce::String(accelName));
        }
    #endif
    return availableAccels;
}

auto MLFrameworkONNXRuntime::tryAcceleratorTensorRT() const -> bool {
    bool result = false;
    #if defined ENABLE_ONNXRUNTIME_TENSORRT
        std::vector<std::string> onnxAvailableNames = Ort::GetAvailableProviders();
        if (std::find(onnxAvailableNames.begin(), onnxAvailableNames.end(), "TensorrtExecutionProvider") != onnxAvailableNames.end()) {
            try {
                Ort::SessionOptions sessionOptions{};
                OrtTensorRTProviderOptionsV2 *tensorOptions = nullptr;
                Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&tensorOptions));
                // tensorOptions->device_id = 0;
                sessionOptions.AppendExecutionProvider_TensorRT_V2(*tensorOptions);
                Ort::GetApi().ReleaseTensorRTProviderOptions(tensorOptions);
                result = true;
            } catch (const Ort::Exception &exception) {
                DBG("ONNX Runtime TensorRT not available: " + juce::String(exception.what()));
            }
        }
    #endif
    return result;
}

auto MLFrameworkONNXRuntime::tryAcceleratorCUDA() const -> bool {
    bool result = false;
    #if defined ENABLE_ONNXRUNTIME_CUDA
        std::vector<std::string> onnxAvailableNames = Ort::GetAvailableProviders();
        if (std::find(onnxAvailableNames.begin(), onnxAvailableNames.end(), "CUDAExecutionProvider") != onnxAvailableNames.end()) {
            try {
                Ort::SessionOptions sessionOptions{};
                OrtCUDAProviderOptionsV2 *cudaOptions = nullptr;
                Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cudaOptions));
                sessionOptions.AppendExecutionProvider_CUDA_V2(*cudaOptions);
                Ort::GetApi().ReleaseCUDAProviderOptions(cudaOptions);
                result = true;
            } catch (const Ort::Exception &exception) {
                DBG("ONNX Runtime CUDA not available: " + juce::String(exception.what()));
            }
        }
    #endif
    return result;
}

auto MLFrameworkONNXRuntime::tryAcceleratorCPU() const -> bool {
    bool result = false;
    #if defined ENABLE_ONNXRUNTIME_CPU
        std::vector<std::string> onnxAvailableNames = Ort::GetAvailableProviders();
        result = std::find(onnxAvailableNames.begin(), onnxAvailableNames.end(), "CPUExecutionProvider") != onnxAvailableNames.end();
    #endif
    return result;
}

void MLFrameworkONNXRuntime::init(const std::string &modelPath, const ModelType &newModelType, MLFramework::ACCELERATOR accelerator) {
    Ort::SessionOptions sessionOptions;
    OrtTensorRTProviderOptionsV2 *tensorOptions = nullptr;
    OrtCUDAProviderOptionsV2 *cudaOptions = nullptr;

    switch (accelerator) {
        #if defined ENABLE_ONNXRUNTIME_CPU
        case MLFramework::ACCELERATOR::ONNXRUNTIME_CPU:
            // Use default CPU execution provider
            break;
        #endif
        #ifdef ENABLE_ONNXRUNTIME_TENSORRT
        case MLFramework::ACCELERATOR::ONNXRUNTIME_TENSORRT:
            Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&tensorOptions));
            // tensorOptions->device_id = 0;
            sessionOptions.AppendExecutionProvider_TensorRT_V2(*tensorOptions);
            Ort::GetApi().ReleaseTensorRTProviderOptions(tensorOptions);
            break;
        #endif
        #if defined ENABLE_ONNXRUNTIME_CUDA
        case MLFramework::ACCELERATOR::ONNXRUNTIME_CUDA:
            Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cudaOptions));
            sessionOptions.AppendExecutionProvider_CUDA_V2(*cudaOptions);
            Ort::GetApi().ReleaseCUDAProviderOptions(cudaOptions);
            break;
        #endif
        default:
            throw std::runtime_error("Unsupported accelerator for ONNX Runtime.");
            break;
    }

    // Create session and set model type
    session = std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOptions);
    modelType = std::make_unique<ModelType>(newModelType);

    // Set Input and Output names
    {
        allocatedInputNames.clear();
        allocatedOutputNames.clear();
        const Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < session->GetInputCount(); i++) {
            allocatedInputNames.emplace_back(session->GetInputNameAllocated(i, allocator).get());
        }
        for (size_t i = 0; i < session->GetOutputCount(); i++) {
            allocatedOutputNames.emplace_back(
                session->GetOutputNameAllocated(i, allocator).get());
        }
    }

    // Create empty Past tensors
    pastShape = std::make_unique<std::vector<int64_t> >(std::initializer_list<int64_t>{2, 1, getHiddenSize(), 0, 64});
    emptyPast = std::make_unique<std::vector<float> >(std::accumulate(pastShape->begin(), pastShape->end(),
                                                                      static_cast<int64_t>(1),
                                                                      std::multiplies<int64_t>()), 0.0f);
}

std::vector<float> MLFrameworkONNXRuntime::runModelAndGetLogits(std::vector<int32_t> &tokens, bool returnAllLogits) {
    // Prepare buffers for inputs and outputs
    const std::vector input_shape{1, static_cast<int64_t>(tokens.size())};

    // Attention mask and position_ids setup (we assume the same shape as input for simplicity)
    std::vector attention_mask(tokens.size(), 1);
    std::vector position_ids(tokens.size(), 0);
    std::iota(position_ids.begin(), position_ids.end(), 0);

    Ort::Value input_tensor = Ort::Value::CreateTensor<int32_t>(memoryInfo, tokens.data(), tokens.size(),
                                                                input_shape.data(), input_shape.size());
    Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int32_t>(
        memoryInfo, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size());
    Ort::Value position_ids_tensor = Ort::Value::CreateTensor<int32_t>(memoryInfo, position_ids.data(),
                                                                       position_ids.size(), input_shape.data(),
                                                                       input_shape.size());

    // Fill the input tensor vector without copying Ort::Value objects
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));
    input_tensors.push_back(std::move(position_ids_tensor));
    input_tensors.push_back(std::move(attention_mask_tensor));

    // Add past state tensor(s) to the input_tensors if needed
    for (size_t i = 0; i < getNHeads(); ++i) {
        Ort::Value pastTensor = Ort::Value::CreateTensor<float>(memoryInfo, emptyPast->data(), emptyPast->size(),
                                                                pastShape->data(), pastShape->size());
        input_tensors.push_back(std::move(pastTensor));
    }

    // convert from string to char*
    std::vector<const char *> inputNames;
    std::vector<const char *> outputNames;
    for (auto &allocatedInputName: allocatedInputNames) {
        inputNames.push_back(allocatedInputName.c_str());
    }
    for (auto &allocatedOutputName: allocatedOutputNames) {
        outputNames.push_back(allocatedOutputName.c_str());
    }

    // Run inference
    try {
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, inputNames.data(), input_tensors.data(),
                                           input_tensors.size(), outputNames.data(), outputNames.size());


        // Get logits output tensor
        Ort::Value &logits_tensor = output_tensors.front();

        if (returnAllLogits) {
            std::vector scores(logits_tensor.GetTensorMutableData<float>(),
                               logits_tensor.GetTensorMutableData<float>()
                               + logits_tensor.GetTensorTypeAndShapeInfo().GetElementCount());
            return scores;
        }

        // Sort the logits tensor
        std::vector scores(logits_tensor.GetTensorMutableData<float>() + (Vocab::VocabSize * (tokens.size() - 1)),
                           logits_tensor.GetTensorMutableData<float>()
                           + logits_tensor.GetTensorTypeAndShapeInfo().GetElementCount());

        return scores;
    } catch (const Ort::Exception &exception) {
        DBG("Error running model: " + juce::String(exception.what()));
    }
}
#endif
