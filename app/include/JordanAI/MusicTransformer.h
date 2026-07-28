#pragma once

#include <JuceHeader.h>

#include "MLFramework/MLFramework.h"

#ifdef ENABLE_COREML
#include "MLFramework/MLFrameworkCoreML.h"
#endif

#ifdef ENABLE_GGML
#include "MLFramework/MLFrameworkGGML.h"
#endif

#ifdef ENABLE_ONNXRUNTIME
#include "MLFramework/MLFrameworkONNXRuntime.h"
#endif

#ifdef ENABLE_TORCH
#include "MLFramework/MLFrameworkTorch.h"
#endif

#include <random>
#include <limits>

#include "Fifo.h"
#include "MetricsComponent.h"
#include "ModelConfigurationComponent.h"
#include "JordanAI/MusicModelList.h"

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
        return "(" + std::to_string(time) + ", " + std::to_string(duration) + ", " + std::to_string(note) + ")";
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
               + std::to_string(getPitch()) + ")";
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

class MusicTransformer : public juce::Thread {
public:
    MusicTransformer(ModelConfig &modelConfig);

    void run() override;

    void init(MusicModel newMusicModel, ModelType &modelType);

    auto getCurrentMusicModel() const -> MusicModel;

    CircularFifo<Token> inputTokenQueue;
    CircularFifo<Token> outputTokenQueue;

    auto getInputData() const -> std::vector<int32_t> {
        return inputData;
    }

    auto getCurrentTime() const -> int32_t {
        return currentTime;
    }

    juce::Atomic<int32_t> tradingOffset;

    juce::Atomic<bool> directInputBlock = false;


    /* Whether the generation is currently paused */
    juce::Atomic<bool> paused = false;

    juce::Component::SafePointer<MetricsComponent> metricsWindow;

private:
    MusicModel musicModel;

    ModelConfig &modelConfig;

    void instrLogits(std::vector<float> &logits);

    void futureLogits(std::vector<float> &logits, int currentTime, int32_t forceAtTime);

    void durLogits(std::vector<float> &logits);

    auto generateNewToken(int32_t forceAtTime) -> Token;

    auto shouldTradeNow() const -> bool;

    auto nextTradingStop() const -> int32_t;

    auto tradingStart() const -> int32_t;

    std::unique_ptr<MLFramework> mlFramework;

    std::vector<int32_t> inputData;

    int32_t currentTime = 0;
    int32_t finalTime = 10000; // 100 seconds // TODO Set this from somewhere else.

    int32_t currentBar = 0;

    /* Used to avoid looking at the minimum time interval on the first token */
    bool firstToken = true;

    /* Used to keep track of how many notes were generated with the same onset time */
    int32_t currentChordSize = 0;

    /* Keep track of the last generated token time for outputPauseAfterTimeInterval */
    // TODO: This kinda clashes with the lastTokenTime in the MusicTransformer.cpp code, which should be removed at some
    // TODO: point since it's used for the deprecated send click functionality.
    int32_t lastGeneratedTokenTime = -1;

    auto clearInputTokenQueue() -> void {
        Token t{-1, -1, -1};
        while (inputTokenQueue.pull(t)) {
        }
    }

    auto clearOutputTokenQueue() -> void {
        Token t{-1, -1, -1};
        while (outputTokenQueue.pull(t)) {
        }
    }

    // CONFIG
    const int32_t maximumFuture = 200;
};
