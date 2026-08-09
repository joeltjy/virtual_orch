#include "VirtualOrch/DenseMusicTransformer.h"
#include "VirtualOrch/OrtEnv.h"

#include <algorithm>
#include <limits>
#include <numeric>

namespace {

constexpr int denseEventWidth = 4;
constexpr int denseLookbackEvents = 40;
constexpr int denseLookbackInts = denseLookbackEvents * denseEventWidth;

} // namespace

DenseMusicTransformer::DenseMusicTransformer(ModelConfig &modelConfigIn)
    : ReductionTransformer("Dense Music Transformer", modelConfigIn) {
}

void DenseMusicTransformer::init(const char *modelPath) {
    Ort::SessionOptions sessionOptions;

#ifdef __linux__
    OrtCUDAProviderOptions cudaOptions{};
    sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
#endif

    session = std::make_unique<Ort::Session>(sharedOrtEnv(), modelPath, sessionOptions);

    allocatedInputNames.clear();
    allocatedOutputNames.clear();
    const Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session->GetInputCount(); ++i)
        allocatedInputNames.emplace_back(session->GetInputNameAllocated(i, allocator).get());
    for (size_t i = 0; i < session->GetOutputCount(); ++i)
        allocatedOutputNames.emplace_back(session->GetOutputNameAllocated(i, allocator).get());
}

void DenseMusicTransformer::threadInit() {
}

void DenseMusicTransformer::threadRun() {
    if (session == nullptr)
        return;

    inputData.clear();
    clearInputTokenQueue();
    clearInputConditioningQueue();
    clearOutputTokenQueue();
    clearUpdatesFromFilter();
    clearOutputHistory();

    currentTime = modelConfig.outputStartTime;

    int32_t forceAtTime = -1;
    if (modelConfig.outputForceStartTime)
        forceAtTime = modelConfig.outputStartTime;

    if (modelConfig.inputInitialData.length() > 0) {
        juce::StringArray parts;
        parts.addTokens(modelConfig.inputInitialData, " ", "");
        for (int i = 0; i + 3 < parts.size(); i += denseEventWidth) {
            inputData.push_back(parts[i].getIntValue());
            inputData.push_back(parts[i + 1].getIntValue());
            inputData.push_back(parts[i + 2].getIntValue());
            inputData.push_back(parts[i + 3].getIntValue());
        }
    }
    notifyInputDataChanged();

    bool inputApplied = false;
    aheadThrottleUntilMs = 0;

    while (! threadShouldExit()) {
        finishAheadThrottleIfDue();

        if (modelConfig.inputMode == InputMode::Direct && modelConfig.directInputStartOnInput
            && directInputBlock.value) {
            wait(5);
            continue;
        }

        inputApplied = applyQueuedInputToInputData();
        applyUpdatesFromFilter();

        if (paused.get()) {
            if (inputApplied) {
                const Token clearToken{static_cast<int32_t>(DenseVocab::TimeOffset),
                                       static_cast<int32_t>(DenseVocab::DurOffset),
                                       static_cast<int32_t>(Vocab::ClearQueue)};
                pushOutputToken(clearToken);
                inputApplied = false;
            } else {
                wait(5);
            }
            continue;
        }

        Token newToken = generateNewToken(forceAtTime);
        forceAtTime = -1;

        if (newToken.time < 0) {
            wait(5);
            continue;
        }

        inputData.push_back(newToken.time);
        inputData.push_back(newToken.duration);
        inputData.push_back(newToken.note);
        inputData.push_back(static_cast<int32_t>(DenseVocab::VelocityOffset + newToken.velocity));
        notifyInputDataChanged();

        currentTime = newToken.time;

        if (inputApplied) {
            // Shared app marker so OutputPlayback recognizes clears when wired.
            const Token clearToken{static_cast<int32_t>(DenseVocab::TimeOffset),
                                   static_cast<int32_t>(DenseVocab::DurOffset),
                                   static_cast<int32_t>(Vocab::ClearQueue)};
            pushOutputToken(clearToken);
            inputApplied = false;
        }

        pushOutputToken(newToken);

        if (isGeneratedTooFarAhead(newToken.time))
            startAheadThrottle();
    }
}

auto DenseMusicTransformer::applyQueuedInputToInputData() -> bool {
    Token discarded{-1, -1, -1};
    while (inputConditioningQueue.pull(discarded)) {
        if (orchestrationConditioningIncoming != nullptr)
            orchestrationConditioningIncoming->push(discarded);
    }

    bool inputApplied = false;
    Token inputToken{-1, -1, -1};

    while (inputTokenQueue.pull(inputToken)) {
        if (orchestrationMidiIncoming != nullptr)
            orchestrationMidiIncoming->push(inputToken);

        if (! inputApplied) {
            for (size_t i = 0; i + 3 < inputData.size(); i += denseEventWidth) {
                if (inputData[i] > inputToken.time) {
                    inputData.erase(inputData.begin() + static_cast<std::ptrdiff_t>(i), inputData.end());
                    break;
                }
            }
        }

        const int32_t velocity =
            juce::jlimit(0, DenseConfig::MaxVelocity - 1,
                         inputToken.velocity > 0 ? inputToken.velocity : DenseConfig::DefaultVelocity);

        if (modelConfig.inputMode != InputMode::Buffer) {
            inputData.push_back(inputToken.time);
            inputData.push_back(inputToken.duration);
            inputData.push_back(inputToken.note);
            inputData.push_back(static_cast<int32_t>(DenseVocab::VelocityOffset + velocity));
        } else {
            const int32_t pitchClass = inputToken.getPitch() % 12;
            const int32_t instrBase =
                static_cast<int32_t>(DenseVocab::NoteOffset
                                    + DenseConfig::MaxPitch * modelConfig.inputInstrument);
            if (! inputApplied) {
                inputData.push_back(inputToken.time);
                inputData.push_back(inputToken.duration);
                inputData.push_back(instrBase + 36 + pitchClass);
                inputData.push_back(static_cast<int32_t>(DenseVocab::VelocityOffset + velocity));
                inputData.push_back(inputToken.time);
                inputData.push_back(inputToken.duration);
                inputData.push_back(instrBase + 48 + pitchClass);
                inputData.push_back(static_cast<int32_t>(DenseVocab::VelocityOffset + velocity));
            } else {
                inputData.push_back(inputToken.time);
                inputData.push_back(inputToken.duration);
                inputData.push_back(instrBase + 60 + pitchClass);
                inputData.push_back(static_cast<int32_t>(DenseVocab::VelocityOffset + velocity));
            }
        }

        currentTime = inputToken.time;
        inputApplied = true;
    }

    if (inputApplied)
        notifyInputDataChanged();

    return inputApplied;
}

auto DenseMusicTransformer::applyUpdatesFromFilter() -> bool {
    TokenUpdate update{};
    if (! updatesFromFilter.pull(update))
        return false;

    std::vector<Token> history;
    history.reserve(inputData.size() / denseEventWidth);
    for (size_t i = 0; i + 3 < inputData.size(); i += denseEventWidth) {
        const int32_t velToken = inputData[i + 3];
        const int32_t velocity =
            juce::jlimit(0, DenseConfig::MaxVelocity - 1,
                         velToken - static_cast<int32_t>(DenseVocab::VelocityOffset));
        history.push_back(Token{inputData[i], inputData[i + 1], inputData[i + 2], velocity});
    }

    bool changed = false;
    do {
        if (orchestrationUpdatesIncoming != nullptr)
            orchestrationUpdatesIncoming->push(update);
        if (applyTokenUpdateToHistory(history, update))
            changed = true;
    } while (updatesFromFilter.pull(update));

    if (! changed)
        return false;

    inputData.clear();
    inputData.reserve(history.size() * denseEventWidth);
    for (const auto &t : history) {
        inputData.push_back(t.time);
        inputData.push_back(t.duration);
        inputData.push_back(t.note);
        inputData.push_back(static_cast<int32_t>(
            DenseVocab::VelocityOffset
            + juce::jlimit(0, DenseConfig::MaxVelocity - 1, t.velocity)));
    }
    notifyInputDataChanged();
    return true;
}

void DenseMusicTransformer::threadStop() {
}

void DenseMusicTransformer::instrLogits(std::vector<float> &logits) {
    auto it = modelConfig.sortedActiveOutputInstruments.begin();
    auto end = modelConfig.sortedActiveOutputInstruments.end();

    if (it == end) {
        // No active instruments configured: allow all dense instruments 0..MaxInstr-1.
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::VelocityOffset),
                  logits.end(),
                  -std::numeric_limits<float>::infinity());
        return;
    }

    // Clamp to dense instrument range.
    while (it != end && *it >= DenseConfig::MaxInstr)
        ++it;
    if (it == end) {
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::NoteOffset),
                  logits.end(),
                  -std::numeric_limits<float>::infinity());
        return;
    }

    std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::NoteOffset),
              logits.begin()
                  + static_cast<std::ptrdiff_t>(DenseVocab::NoteOffset
                                                + DenseConfig::MaxPitch * (*it)
                                                + modelConfig.outputInstruments[*it].low),
              -std::numeric_limits<float>::infinity());

    auto prev_it = it;
    ++it;
    for (; it != end; ++it, ++prev_it) {
        if (*it >= DenseConfig::MaxInstr)
            break;
        std::fill(
            logits.begin()
                + static_cast<std::ptrdiff_t>(DenseVocab::NoteOffset
                                              + DenseConfig::MaxPitch * (*prev_it)
                                              + modelConfig.outputInstruments[*prev_it].high),
            logits.begin()
                + static_cast<std::ptrdiff_t>(DenseVocab::NoteOffset
                                              + DenseConfig::MaxPitch * (*it)
                                              + modelConfig.outputInstruments[*it].low),
            -std::numeric_limits<float>::infinity());
    }

    std::fill(
        logits.begin()
            + static_cast<std::ptrdiff_t>(DenseVocab::NoteOffset
                                          + DenseConfig::MaxPitch * (*prev_it)
                                          + modelConfig.outputInstruments[*prev_it].high),
        logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::VelocityOffset),
        -std::numeric_limits<float>::infinity());

    std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::VelocityOffset),
              logits.end(),
              -std::numeric_limits<float>::infinity());
}

void DenseMusicTransformer::futureLogits(std::vector<float> &logits, const int curTime,
                                         int32_t forceAtTime) {
    if (forceAtTime != -1 && forceAtTime < DenseConfig::MaxTime) {
        std::fill_n(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::TimeOffset), forceAtTime,
                    -std::numeric_limits<float>::infinity());
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::TimeOffset + forceAtTime + 1),
                  logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::DurOffset),
                  -std::numeric_limits<float>::infinity());
        return;
    }

    if (curTime > 0) {
        std::fill_n(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::TimeOffset), curTime,
                    -std::numeric_limits<float>::infinity());
    }

    if (curTime < (DenseConfig::MaxTime - maximumFuture)) {
        std::fill(
            logits.begin()
                + static_cast<std::ptrdiff_t>(DenseVocab::TimeOffset + curTime + maximumFuture),
            logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::DurOffset),
            -std::numeric_limits<float>::infinity());
    }
}

void DenseMusicTransformer::durLogits(std::vector<float> &logits) {
    std::fill_n(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::DurOffset),
                modelConfig.outputMinimumDuration,
                -std::numeric_limits<float>::infinity());
    std::fill(logits.begin()
                  + static_cast<std::ptrdiff_t>(DenseVocab::DurOffset
                                                + modelConfig.outputMaximumDuration + 1),
              logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::NoteOffset),
              -std::numeric_limits<float>::infinity());
}

void DenseMusicTransformer::velocityLogits(std::vector<float> &logits) {
    // Allow full velocity band; mask everything else via safeLogits.
    juce::ignoreUnused(logits);
}

void DenseMusicTransformer::safeLogits(std::vector<float> &logits, size_t stepIdx) {
    if (logits.size() < DenseVocab::VocabSize) {
        DBG("DenseMusicTransformer::safeLogits: logits size "
            + juce::String(static_cast<int>(logits.size())) + " < VocabSize "
            + juce::String(static_cast<int>(DenseVocab::VocabSize)));
        return;
    }

    // Mask anything past model vocab (if ORT returns a larger last dim).
    if (logits.size() > DenseVocab::VocabSize) {
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::VocabSize), logits.end(),
                  -std::numeric_limits<float>::infinity());
    }

    const auto maskTime = [&] {
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::TimeOffset),
                  logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::DurOffset),
                  -std::numeric_limits<float>::infinity());
    };
    const auto maskDur = [&] {
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::DurOffset),
                  logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::NoteOffset),
                  -std::numeric_limits<float>::infinity());
    };
    const auto maskNote = [&] {
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::NoteOffset),
                  logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::VelocityOffset),
                  -std::numeric_limits<float>::infinity());
    };
    const auto maskVel = [&] {
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::VelocityOffset),
                  logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::VocabSize),
                  -std::numeric_limits<float>::infinity());
    };

    switch (stepIdx % denseEventWidth) {
        case 0: // time
            maskDur();
            maskNote();
            maskVel();
            break;
        case 1: // duration
            maskTime();
            maskNote();
            maskVel();
            break;
        case 2: // note
            maskTime();
            maskDur();
            maskVel();
            break;
        case 3: // velocity
            maskTime();
            maskDur();
            maskNote();
            break;
        default:
            break;
    }
}

std::vector<float> DenseMusicTransformer::runModelAndGetLogits(std::vector<int32_t> &tokens) {
    if (session == nullptr || tokens.empty())
        return {};

    const std::vector input_shape{1, static_cast<int64_t>(tokens.size())};

    // Prefer int64 input_ids (common for exported dense stubs); fall back to int32.
    std::vector<int64_t> tokenIds(tokens.begin(), tokens.end());
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        memoryInfo, tokenIds.data(), tokenIds.size(), input_shape.data(), input_shape.size());

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));

    std::vector<const char *> inputNames;
    std::vector<const char *> outputNames;
    for (auto &name : allocatedInputNames)
        inputNames.push_back(name.c_str());
    for (auto &name : allocatedOutputNames)
        outputNames.push_back(name.c_str());

    try {
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, inputNames.data(),
                                           input_tensors.data(), input_tensors.size(),
                                           outputNames.data(), outputNames.size());

        Ort::Value &logits_tensor = output_tensors.front();
        const auto shape = logits_tensor.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.empty())
            return {};

        const auto vocab = static_cast<size_t>(shape.back());
        if (vocab == 0)
            return {};

        const size_t seq = tokens.size();
        const auto elementCount = logits_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
        if (seq == 0 || vocab * seq > elementCount)
            return {};

        const float *data = logits_tensor.GetTensorMutableData<float>();
        const float *last = data + vocab * (seq - 1);
        return {last, last + vocab};
    } catch (const Ort::Exception &exception) {
        DBG("DenseMusicTransformer ORT error: " + juce::String(exception.what()));
    }

    return {};
}

auto DenseMusicTransformer::denseMinTime(std::vector<int32_t> &tokens) -> int32_t {
    int32_t minT = INT_MAX;
    for (size_t i = 0; i + 3 < tokens.size(); i += denseEventWidth) {
        minT = std::min(minT, tokens[i] - static_cast<int32_t>(DenseVocab::TimeOffset));
    }
    return minT == INT_MAX ? 0 : minT;
}

Token DenseMusicTransformer::generateNewToken(int32_t forceAtTime) {
    if (inputData.size() % denseEventWidth != 0)
        throw std::runtime_error("DenseMusicTransformer inputData must be a multiple of 4");

    if (inputData.empty())
        return {-1, -1, -1, DenseConfig::DefaultVelocity};

    const int lookback = std::max(static_cast<int>(inputData.size()) - denseLookbackInts, 0);
    std::vector history(inputData.begin() + lookback, inputData.end());
    int32_t offset = 0;
    if (! history.empty())
        offset = denseMinTime(history);

    for (size_t i = 0; i < history.size(); i += denseEventWidth)
        history[i] -= offset;

    Token newToken{-1, -1, -1, DenseConfig::DefaultVelocity};

    for (int i = 0; i < denseEventWidth; ++i) {
        std::vector<float> scores = runModelAndGetLogits(history);
        if (scores.size() < DenseVocab::VocabSize) {
            DBG("DenseMusicTransformer::generateNewToken: unexpected logits size "
                + juce::String(static_cast<int>(scores.size()))
                + " (historyInts=" + juce::String(static_cast<int>(history.size())) + ")");
            return {-1, -1, -1, DenseConfig::DefaultVelocity};
        }

        safeLogits(scores, static_cast<size_t>(i));
        if (i == 0)
            futureLogits(scores, currentTime - offset,
                         forceAtTime != -1 ? forceAtTime - offset : -1);
        else if (i == 1)
            durLogits(scores);
        else if (i == 2)
            instrLogits(scores);
        else
            velocityLogits(scores);

        const float temperature =
            i < 3 ? static_cast<float>(modelConfig.outputTemperatures[static_cast<size_t>(i)])
                  : static_cast<float>(modelConfig.outputTemperatures[2]);
        const int32_t token = sampleTopP(scores, 0.9f, temperature);
        history.push_back(token);

        if (i == 0)
            newToken.time = token + offset;
        else if (i == 1)
            newToken.duration = token;
        else if (i == 2)
            newToken.note = token;
        else
            newToken.velocity =
                juce::jlimit(0, DenseConfig::MaxVelocity - 1,
                             token - static_cast<int32_t>(DenseVocab::VelocityOffset));
    }

    return newToken;
}
