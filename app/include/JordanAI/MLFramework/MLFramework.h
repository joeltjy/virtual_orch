#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace Config {
    constexpr int32_t MaxTimeInSeconds = 100;
    constexpr int32_t MaxDurationInSeconds = 10;
    constexpr int32_t TimeResolution = 100;

    constexpr int32_t MaxPitch = 128;
    constexpr int32_t MaxInstr = 129;
    constexpr int32_t MaxNote = MaxPitch * MaxInstr;

    constexpr int32_t MaxTime = TimeResolution * MaxTimeInSeconds;
    constexpr int32_t MaxDur = TimeResolution * MaxDurationInSeconds;

    constexpr int32_t AnticipationDelta = 0 * TimeResolution;
} // namespace Config

namespace Vocab {
    // Event Block
    constexpr size_t EventOffset = 0;
    constexpr size_t TimeOffset = EventOffset + 0;
    constexpr size_t DurOffset = TimeOffset + Config::MaxTime;
    constexpr size_t NoteOffset = DurOffset + Config::MaxDur;

    // SPECIAL ALIASES
    constexpr size_t BarClick = NoteOffset + Config::MaxPitch * 128 + 36;
    constexpr size_t BeatClick = NoteOffset + Config::MaxPitch * 128 + 42;

    constexpr size_t Rest = NoteOffset + Config::MaxNote;

    // Control Block
    constexpr size_t ControlOffset = NoteOffset + Config::MaxNote + 1;
    constexpr size_t AtimeOffset = ControlOffset + 0;
    constexpr size_t AdurOffset = AtimeOffset + Config::MaxTime;
    constexpr size_t AnoteOffset = AdurOffset + Config::MaxDur;

    // Special Block
    constexpr size_t SpecialOffset = AnoteOffset + Config::MaxNote;
    constexpr size_t Separator = SpecialOffset + 0;
    constexpr size_t AutoRegress = SpecialOffset + 1;
    constexpr size_t Anticipate = SpecialOffset + 2;
    constexpr size_t VocabSize = Anticipate + 1;

    // Added events
    constexpr size_t BarSeparator = VocabSize + 1;
    constexpr size_t ClearQueue = VocabSize + 2;
    constexpr size_t PauseQueue = VocabSize + 3;
    constexpr size_t UnpauseQueue = VocabSize + 4;
    constexpr size_t ReleaseNotes = VocabSize + 5;
} // namespace Vocab

enum ModelType : uint8_t {
    Small,
    Medium
};

constexpr int SMALL_HIDDEN_SIZE = 12;
constexpr int MEDIUM_HIDDEN_SIZE = 16;

constexpr int SMALL_N_HEADS = 12;
constexpr int MEDIUM_N_HEADS = 24;

class MLFramework {
public:
    virtual ~MLFramework() = default;

    auto getModelType() const -> ModelType {
        if (modelType) {
            return *modelType;
        }
        return Small;
    }

    virtual std::vector<float> runModelAndGetLogits(std::vector<int32_t> &tokens, bool returnAllLogits = false) = 0;

    enum class ACCELERATOR {
        COREML_COREML,
        GGML_CPU,
        GGML_CUDA,
        GGML_METAL,
        GGML_SYCL,
        GGML_VULKAN,
        ONNXRUNTIME_CPU,
        ONNXRUNTIME_TENSORRT,
        ONNXRUNTIME_CUDA,
        TORCH_CUDA,
        TORCH_MPS,
        TORCH_CPU
    };
    static const std::map<ACCELERATOR, std::string> ACCELERATOR_FRIENDLY_NAMES;
    static const std::map<ACCELERATOR, std::string> ACCELERATOR_NAMES;
    static const std::map<ACCELERATOR, std::string> ACCELERATOR_FILE_EXTENSIONS;

    static const size_t IDEAL_MIN_RESERVED_THREADS_NON_FRAMEWORK = 4; // MainComponent, MusicTransformer, MetricsComponent, MLFramework
    static const size_t UNIDEAL_MIN_RESERVED_THREADS_FRAMEWORK = 2;

    static auto getAllBuiltAccelerators () -> std::unordered_set<ACCELERATOR>;
    static auto getAllAvailableAccelerators () -> std::unordered_set<ACCELERATOR>;

    virtual void init (const std::string &modelPath, const ModelType &modelType, ACCELERATOR accelerator) = 0;

protected:
    MLFramework() {}

    virtual auto getBuiltAccelerators () const -> std::unordered_set<ACCELERATOR> = 0;
    virtual auto getAvailableAccelerators () -> std::unordered_set<ACCELERATOR> = 0;

    std::unique_ptr<ModelType> modelType;

    auto getHiddenSize() const -> int {
        switch (*modelType) {
            case Medium:
                return MEDIUM_HIDDEN_SIZE;
                break;
            case Small:
            default:
                return SMALL_HIDDEN_SIZE;
        }
    }

    auto getNHeads() const -> int {
        switch (*modelType) {
            case Medium:
                return MEDIUM_N_HEADS;
                break;
            case Small:
            default:
                return SMALL_N_HEADS;
        }
    }
};
