#pragma once

#include <JuceHeader.h>
#include <onnxruntime_cxx_api.h>

#include <functional>
#include <vector>

#include "Fifo.h"
#include "VirtualOrch/MusicTransformer.h"
#include "VirtualOrch/ui/ModelConfigurationComponent.h"

/**
 * Dense AMT-style vocab (time, duration, note, velocity).
 * Parallel to Config/Vocab in MusicTransformer.h — do not mix offsets.
 *
 * note = NoteOffset + MaxPitch * instrument + pitch, instrument in [0, MaxInstr).
 * velocity token = VelocityOffset + v, v in [0, MaxVelocity).
 * Model vocab ends at VocabSize (11768); no control / anticipate block.
 */
namespace DenseConfig {
inline constexpr int32_t MaxTimeInSeconds = 100;
inline constexpr int32_t MaxDurationInSeconds = 10;
inline constexpr int32_t TimeResolution = 100;

inline constexpr int32_t MaxPitch = 128;
inline constexpr int32_t MaxInstr = 5;
inline constexpr int32_t MaxNote = MaxPitch * MaxInstr; // 640
inline constexpr int32_t MaxVelocity = 128;
inline constexpr int32_t DefaultVelocity = 100;

inline constexpr int32_t MaxTime = TimeResolution * MaxTimeInSeconds; // 10000
inline constexpr int32_t MaxDur = TimeResolution * MaxDurationInSeconds; // 1000
} // namespace DenseConfig

namespace DenseVocab {
inline constexpr size_t TimeOffset = 0;
inline constexpr size_t DurOffset = TimeOffset + DenseConfig::MaxTime;       // 10000
inline constexpr size_t NoteOffset = DurOffset + DenseConfig::MaxDur;        // 11000
inline constexpr size_t VelocityOffset = NoteOffset + DenseConfig::MaxNote; // 11640 (= NoteOffset + 5*128)

/** Exclusive end of model vocab: ids in [0, VocabSize). */
inline constexpr size_t VocabSize = VelocityOffset + DenseConfig::MaxVelocity; // 11768
} // namespace DenseVocab

enum class MusicModelArch : uint8_t {
    Amt,
    Dense
};

/**
 * Dense Music Transformer: 4-wide events (time, dur, note, velocity).
 * ONNX: input_ids -> logits (no past KV / anticipate).
 */
class DenseMusicTransformer : public juce::Thread {
public:
    explicit DenseMusicTransformer(ModelConfig &modelConfig);

    void run() override {
        threadInit();
        threadRun();
        threadStop();
    }

    void init(const char *modelPath);

    [[nodiscard]] auto isModelLoaded() const -> bool { return session != nullptr; }

    void threadInit();

    void threadRun();

    void threadStop();

    CircularFifo<Token> inputTokenQueue;
    CircularFifo<Token> inputConditioningQueue;
    CircularFifo<Token> outputTokenQueue;
    CircularFifo<TokenUpdate> updatesFromFilter;

    CircularFifo<Token> *orchestrationMidiIncoming = nullptr;
    CircularFifo<Token> *orchestrationConditioningIncoming = nullptr;
    CircularFifo<TokenUpdate> *orchestrationUpdatesIncoming = nullptr;

    auto getInputData() const -> std::vector<int32_t> { return inputData; }

    auto getCurrentTime() const -> int32_t { return currentTime; }

    std::function<void(std::vector<int32_t>)> onInputDataChanged;

    juce::Atomic<bool> directInputBlock = false;

private:
    void notifyInputDataChanged();

    auto applyQueuedInputToInputData() -> bool;

    auto applyUpdatesFromFilter() -> bool;

    void instrLogits(std::vector<float> &logits);

    void futureLogits(std::vector<float> &logits, int currentTime, int32_t forceAtTime);

    void durLogits(std::vector<float> &logits);

    void velocityLogits(std::vector<float> &logits);

    void safeLogits(std::vector<float> &logits, size_t stepIdx);

    std::vector<float> runModelAndGetLogits(std::vector<int32_t> &tokens);

    Token generateNewToken(int32_t forceAtTime);

    static auto denseMinTime(std::vector<int32_t> &tokens) -> int32_t;

    ModelConfig &modelConfig;

    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<int32_t> inputData;

    int32_t currentTime = 0;

    std::vector<std::string> allocatedInputNames;
    std::vector<std::string> allocatedOutputNames;

    auto clearInputTokenQueue() -> void {
        Token t{.time = -1, .duration = -1, .note = -1};
        while (inputTokenQueue.pull(t)) {
        }
    }

    auto clearInputConditioningQueue() -> void {
        Token t{.time = -1, .duration = -1, .note = -1};
        while (inputConditioningQueue.pull(t)) {
        }
    }

    auto clearOutputTokenQueue() -> void {
        Token t{.time = -1, .duration = -1, .note = -1};
        while (outputTokenQueue.pull(t)) {
        }
    }

    auto clearUpdatesFromFilter() -> void {
        TokenUpdate u{};
        while (updatesFromFilter.pull(u)) {
        }
    }

    const int32_t maximumFuture = 200;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DenseMusicTransformer)
};
