#pragma once

#include <JuceHeader.h>
#include <onnxruntime_cxx_api.h>

#include <memory>
#include <vector>

#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/ReductionTransformer.h"

/**
 * Dense AMT-style vocab (time, duration, note, velocity).
 * Parallel to Config/Vocab in MusicToken.h — do not mix offsets.
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

/**
 * Dense Music Transformer: 4-wide events (time, dur, note, velocity).
 * ONNX: input_ids -> logits (no past KV / anticipate).
 */
class DenseMusicTransformer : public ReductionTransformer {
public:
    explicit DenseMusicTransformer(ModelConfig &modelConfig);

    void run() override {
        threadInit();
        threadRun();
        threadStop();
    }

    void init(const char *modelPath);

    [[nodiscard]] auto isModelLoaded() const -> bool override { return session != nullptr; }

    void threadInit();

    void threadRun();

    void threadStop();

private:
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

    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<std::string> allocatedInputNames;
    std::vector<std::string> allocatedOutputNames;

    const int32_t maximumFuture = 200;
};
