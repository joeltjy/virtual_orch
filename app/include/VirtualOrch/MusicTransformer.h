#pragma once

#include <JuceHeader.h>
#include <onnxruntime_cxx_api.h>

#include <memory>
#include <vector>

#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/reduction/ReductionTransformer.h"

class MusicTransformer : public ReductionTransformer {
public:
    explicit MusicTransformer(ModelConfig &modelConfig);

    void run() override {
        threadInit();
        threadRun();
        threadStop();
    }

    void init(const char *modelPath, ModelType newModelType);

    [[nodiscard]] auto isModelLoaded() const -> bool override { return session != nullptr; }

    void threadInit();

    void threadRun();

    void threadStop();

private:
    auto applyQueuedInputToInputData() -> bool;

    /** Drain updatesFromFilter; patch matching triplets in inputData (search backwards). */
    auto applyUpdatesFromFilter() -> bool;

    void instrLogits(std::vector<float> &logits);

    void futureLogits(std::vector<float> &logits, int currentTime, int32_t forceAtTime);

    void durLogits(std::vector<float> &logits);

    std::vector<float> runModelAndGetLogits(std::vector<int32_t> &tokens);

    Token generateNewToken(int32_t forceAtTime);

    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<ModelType> modelType;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    int32_t finalTime = 10000; // 100 seconds // TODO Set this from somewhere else.

    auto getHiddenSize() const -> int {
        switch (*modelType) {
            case Medium:
                return MEDIUM_HIDDEN_SIZE;
            case Small:
            default:
                return SMALL_HIDDEN_SIZE;
        }
    }

    auto getNHeads() const -> int {
        switch (*modelType) {
            case Medium:
                return MEDIUM_N_HEADS;
            case Small:
            default:
                return SMALL_N_HEADS;
        }
    }

    std::vector<std::string> allocatedInputNames;
    std::vector<std::string> allocatedOutputNames;

    std::unique_ptr<std::vector<int64_t>> pastShape;
    std::unique_ptr<std::vector<float>> emptyPast;

    const int32_t maximumFuture = 200;
};
