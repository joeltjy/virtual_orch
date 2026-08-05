#pragma once

#include <JuceHeader.h>
#include <onnxruntime_cxx_api.h>

#include <functional>
#include <random>
#include <limits>
#include <algorithm>
#include <vector>

#include "Fifo.h"
#include "VirtualOrch/ui/ModelConfigurationComponent.h"

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
} // namespace Vocab

template<typename T>
static void softmax(T &input, const float temperature) {
    float rowmax = *std::max_element(input.begin(), input.end());
    std::vector<float> y(input.size());
    float sum = 0.0F;
    for (size_t i = 0; i != input.size(); ++i) {
        sum += y[i] = std::exp((input[i] - rowmax) / temperature);
    }
    for (size_t i = 0; i != input.size(); ++i) {
        input[i] = y[i] / sum;
    }
}

template<typename T>
static int32_t sampleTopP(T &scores, const float p, const float temperature) {
    std::uniform_real_distribution<float> dis(0, p);
    std::random_device dev;
    std::mt19937 gen_(dev());
    gen_.seed(std::random_device()());
    softmax(scores, temperature);

    // Sort an array of indices into the scores
    std::vector<int32_t> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [scores = scores.data()](const int32_t i, const int32_t j) {
        return scores[i] > scores[j];
    });

    float threshold = dis(gen_);
    int32_t token = 0;
    // Find the first token where the cumulative probability exceeds the threshold
    for (size_t i = 0; i < scores.size(); i++) {
        threshold -= scores[indices[i]];
        if (threshold > 0) {
            continue;
        }
        token = indices[i];
        break;
    }

    return token;
}

static int32_t minTime(std::vector<int32_t> &tokens) {
    int32_t minTime = INT_MAX;
    for (size_t i = 0; i < tokens.size(); i += 3) {
        int32_t time = tokens[i];
        int32_t note = tokens[i + 2];

        // Stop calculating at sequence separator
        if (note == Vocab::Separator) {
            break;
        }

        if (note < Vocab::ControlOffset) {
            time -= Vocab::TimeOffset;
        } else {
            time -= Vocab::AtimeOffset;
        }

        minTime = std::min(minTime, time);
    }

    return minTime;
}

struct Token {
    int32_t time;
    int32_t duration;
    int32_t note;
    /** MIDI velocity 0–127. Default 100 matches AMT / orchestration when unset. */
    int32_t velocity = 100;

    auto getRealDuration() const -> int32_t {
        return duration - Vocab::DurOffset;
    }

    auto getInstrument() const -> int32_t {
        return (note - Vocab::NoteOffset) / Config::MaxPitch;
    }

    auto getPitch() const -> int32_t {
        return (note - Vocab::NoteOffset) % Config::MaxPitch;
    }

    std::string toString() const {
        return "(" + std::to_string(time) + ", " + std::to_string(duration) + ", "
               + std::to_string(note) + ", " + std::to_string(velocity) + ")";
    }

    std::string toUnderstandableString() const {
        // Check for special tokens
        if (note == Vocab::BarSeparator) {
            return "(" + std::to_string(time) + ", Bar Separator)";
        }
        if (note == Vocab::ClearQueue) {
            return "(" + std::to_string(time) + ", Clear Queue)";
        }
        return "(" + std::to_string(time) + ", "
               + std::to_string(duration - Vocab::DurOffset) + ", "
               + std::to_string(getInstrument()) + " - "
               + std::to_string(getPitch()) + ", vel "
               + std::to_string(velocity) + ")";
    }

    juce::MemoryBlock toMemoryBlock() const {
        juce::MemoryOutputStream stream;

        // Error Token
        if ((time > std::numeric_limits<uint32>::max())
            || (getRealDuration() > std::numeric_limits<uint16>::max())
            || (getInstrument() > std::numeric_limits<uint8>::max()
                && note != Vocab::BarSeparator && note != Vocab::ClearQueue)
            || (note >= Vocab::Rest
                && note != Vocab::BarSeparator && note != Vocab::ClearQueue)) {
            stream.writeInt(0);
            stream.writeShort(0);
            stream.writeByte(0);
            stream.writeByte(0);
        } else {
            stream.writeInt(time);
            stream.writeShort(getRealDuration());
            if (note == Vocab::BarSeparator) {
                stream.writeByte(0);
                stream.writeByte(129);
            } else if (note == Vocab::ClearQueue) {
                stream.writeByte(0);
                stream.writeByte(130);
            } else {
                stream.writeByte(getInstrument());
                stream.writeByte(getPitch());
            }
        }
        return stream.getMemoryBlock();
    }
};

inline bool operator<(const Token &lhs, const Token &rhs) {
    return lhs.time < rhs.time;
}

inline auto tokenEquals(const Token &lhs, const Token &rhs) -> bool {
    return lhs.time == rhs.time && lhs.duration == rhs.duration && lhs.note == rhs.note
           && lhs.velocity == rhs.velocity;
}

inline auto tokenOrderLess(const Token &lhs, const Token &rhs) -> bool {
    if (lhs.time != rhs.time)
        return lhs.time < rhs.time;
    if (lhs.duration != rhs.duration)
        return lhs.duration < rhs.duration;
    if (lhs.note != rhs.note)
        return lhs.note < rhs.note;
    return lhs.velocity < rhs.velocity;
}

/** Replace a note. For example, when noteOffs come in. */
struct TokenUpdate {
    Token oldNote;
    Token newNote;
};

/** Find oldNote scanning backwards; replace with newNote; reinsert if order key changed. */
inline auto applyTokenUpdateToHistory(std::vector<Token> &history, const TokenUpdate &update) -> bool {
    for (auto rit = history.rbegin(); rit != history.rend(); ++rit) {
        if (! tokenEquals(*rit, update.oldNote))
            continue;

        auto pos = std::prev(rit.base());

        const bool orderKeyChanged = update.newNote.time != update.oldNote.time
                                     || update.newNote.duration != update.oldNote.duration
                                     || update.newNote.note != update.oldNote.note
                                     || update.newNote.velocity != update.oldNote.velocity;

        if (! orderKeyChanged) {
            *pos = update.newNote;
            return true;
        }

        history.erase(pos);
        const auto insertAt = std::lower_bound(history.begin(), history.end(), update.newNote, tokenOrderLess);
        history.insert(insertAt, update.newNote);
        return true;
    }

    return false;
}

enum ModelType : uint8_t {
    Small,
    Medium
};

constexpr int SMALL_HIDDEN_SIZE = 12;
constexpr int MEDIUM_HIDDEN_SIZE = 16;

constexpr int SMALL_N_HEADS = 12;
constexpr int MEDIUM_N_HEADS = 24;

class MusicTransformer : public juce::Thread {
public:
    MusicTransformer(ModelConfig &modelConfig);

    void run() override {
        threadInit();
        threadRun();
        threadStop();
    }

    void init(const char *modelPath, ModelType newModelType);

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

    auto getInputData() const -> std::vector<int32_t> {
        return inputData;
    }

    auto getCurrentTime() const -> int32_t {
        return currentTime;
    }

    /** Called on the message thread whenever inputData changes. */
    std::function<void(std::vector<int32_t>)> onInputDataChanged;

    juce::Atomic<bool> directInputBlock = false;

private:
    void notifyInputDataChanged();

    // Decides how to apply the queued input from both queues into inputData. Returns true if any input was applied.
    auto applyQueuedInputToInputData() -> bool;

    /** Drain updatesFromFilter; patch matching triplets in inputData (search backwards). */
    auto applyUpdatesFromFilter() -> bool;

    ModelConfig &modelConfig;

    void instrLogits(std::vector<float> &logits);

    void futureLogits(std::vector<float> &logits, int currentTime, int32_t forceAtTime);

    void durLogits(std::vector<float> &logits);

    std::vector<float> runModelAndGetLogits(std::vector<int32_t> &tokens);

    Token generateNewToken(int32_t forceAtTime);

    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<ModelType> modelType;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<int32_t> inputData;

    int32_t currentTime = 0;
    int32_t finalTime = 10000; // 100 seconds // TODO Set this from somewhere else.

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

    std::vector<std::string> allocatedInputNames;
    std::vector<std::string> allocatedOutputNames;

    std::unique_ptr<std::vector<int64_t> > pastShape;
    std::unique_ptr<std::vector<float> > emptyPast;

    auto clearInputTokenQueue() -> void {
        Token t{.time=-1, .duration=-1, .note=-1};
        while (inputTokenQueue.pull(t)) {
        }
    }

    auto clearInputConditioningQueue() -> void {
        Token t{.time=-1, .duration=-1, .note=-1};
        while (inputConditioningQueue.pull(t)) {
        }
    }

    auto clearOutputTokenQueue() -> void {
        Token t{.time=-1, .duration=-1, .note=-1};
        while (outputTokenQueue.pull(t)) {
        }
    }

    auto clearUpdatesFromFilter() -> void {
        TokenUpdate u{};
        while (updatesFromFilter.pull(u)) {
        }
    }

    // CONFIG
    const int32_t maximumFuture = 200;
};
