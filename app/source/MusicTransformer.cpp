#include "VirtualOrch/MusicTransformer.h"
#include "VirtualOrch/OrtEnv.h"
#include <onnxruntime_cxx_api.h>

MusicTransformer::MusicTransformer(ModelConfig &modelConfig): Thread("Music Transformer"), modelConfig(modelConfig) {
}

void MusicTransformer::notifyInputDataChanged() {
    auto callback = onInputDataChanged;
    if (!callback) {
        return;
    }
    auto snapshot = inputData;
    juce::MessageManager::callAsync([callback = std::move(callback), snapshot = std::move(snapshot)]() mutable {
        callback(std::move(snapshot));
    });
}

void MusicTransformer::init(const char *modelPath, ModelType newModelType) {
    Ort::SessionOptions sessionOptions;

    // If on Linux, use CUDA
#ifdef __linux__
    OrtCUDAProviderOptions cudaOptions{};
    sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
#endif

    // Create session and set model typez
    session = std::make_unique<Ort::Session>(sharedOrtEnv(), modelPath, sessionOptions);
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

void MusicTransformer::threadInit() {
    // TODO Maybe transfer memory_info initialization and stuff here.
}

void MusicTransformer::threadRun() {
    // If no session has been started, stop
    if (session == nullptr) {
        return;
    }

    inputData.clear();
    clearInputTokenQueue();
    clearInputConditioningQueue();
    clearOutputTokenQueue();
    clearUpdatesFromFilter();

    currentTime = modelConfig.outputStartTime;

    // Whether we want to generate a token at a precise time
    int32_t forceAtTime = -1;

    // If we force the start time, set the forceAtTime to the start time
    if (modelConfig.outputForceStartTime) {
        forceAtTime = modelConfig.outputStartTime;
    }

    // If we have initial input data, add it to the input data
    if (modelConfig.inputInitialData.length() > 0) {
        // Split the control sequence by space
        juce::StringArray inputInitialData;
        inputInitialData.addTokens(modelConfig.inputInitialData, " ", "");
        for (int i = 0; i < inputInitialData.size(); i += 3) {
            inputData.push_back(inputInitialData[i].getIntValue());
            inputData.push_back(inputInitialData[i + 1].getIntValue());
            inputData.push_back(inputInitialData[i + 2].getIntValue());
        }
    }
    notifyInputDataChanged();

    // Whether this iteration applied live token-queue input (triggers ClearQueue on output).
    bool inputApplied = false;

    // Until thread is not stopped
    while (!threadShouldExit()) {
        // DIRECT INPUT: WAIT FOR INPUT BLOCK
        if (modelConfig.inputMode == InputMode::Direct && modelConfig.directInputStartOnInput
            && directInputBlock.value) {
            continue;
        }

        inputApplied = applyQueuedInputToInputData();
        applyUpdatesFromFilter();

        // GENERATE NEW TOKEN (OR REST)
        Token newToken = {-1, -1, -1};
        // Generate new token
        newToken = generateNewToken(forceAtTime);

        // We set forceAtTime back to -1
        forceAtTime = -1;

        // Add the token to inputData
        inputData.push_back(newToken.time);
        inputData.push_back(newToken.duration);
        inputData.push_back(newToken.note);
        notifyInputDataChanged();

        // Update current time
        currentTime = newToken.time;

        // If we need to clear the queue, send a clear queue token
        // We do it here to ensure we can push the new token right after, and not have a moment without any token
        if (inputApplied) {
            Token clearToken = {Vocab::TimeOffset, Vocab::DurOffset, Vocab::ClearQueue};
            outputTokenQueue.push(clearToken);
        }

        if (inputApplied) { inputApplied = false; }

        // Push new token to output queue
        outputTokenQueue.push(newToken);
    }
}

auto MusicTransformer::applyQueuedInputToInputData() -> bool {
    Token discarded{-1, -1, -1};
    while (inputConditioningQueue.pull(discarded)) {
        if (orchestrationConditioningIncoming != nullptr)
            orchestrationConditioningIncoming->push(discarded);
    }

    bool inputApplied = false;
    Token inputToken = {-1, -1, -1};

    while (inputTokenQueue.pull(inputToken)) {
        if (orchestrationMidiIncoming != nullptr)
            orchestrationMidiIncoming->push(inputToken);

        DBG("input: " + inputToken.toUnderstandableString());
        // CLEARING FUTURE INPUT DATA IF FIRST TOKEN
        if (!inputApplied) {
            for (size_t i = 0; i < inputData.size(); i += 3) {
                if (inputData[i] > inputToken.time) {
                    inputData.erase(inputData.begin() + i, inputData.end());
                    break;
                }
            }
        }

        // ADD TOKEN TO INPUT DATA (IF BUFFER DO SOME PROCESSING)
        if (modelConfig.inputMode != InputMode::Buffer) {
            inputData.push_back(inputToken.time);
            inputData.push_back(inputToken.duration);
            inputData.push_back(inputToken.note);
        } else {
            // NOTE: Should the output ranges be a parameter?
            // If bass, add an octave to range 36-48 and 48-60
            // Otherwise, put in range 60-72
            if (!inputApplied) {
                inputData.push_back(inputToken.time);
                inputData.push_back(inputToken.duration);
                inputData.push_back(Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument
                                    + 36 + (inputToken.getPitch() % 12));
                inputData.push_back(inputToken.time);
                inputData.push_back(inputToken.duration);
                inputData.push_back(Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument
                                    + 48 + (inputToken.getPitch() % 12));
            } else {
                inputData.push_back(inputToken.time);
                inputData.push_back(inputToken.duration);
                inputData.push_back(Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument
                                    + 60 + (inputToken.getPitch() % 12));
            }
        }

        DBG("In prompt: " + inputToken.toUnderstandableString());

        // Set current time to last token time
        currentTime = inputToken.time;

        inputApplied = true;
    }

    if (inputApplied) {
        notifyInputDataChanged();
    }

    return inputApplied;
}

auto MusicTransformer::applyUpdatesFromFilter() -> bool {
    TokenUpdate update{};
    if (! updatesFromFilter.pull(update))
        return false;

    std::vector<Token> history;
    history.reserve(inputData.size() / 3);
    for (size_t i = 0; i + 2 < inputData.size(); i += 3)
        history.push_back({inputData[i], inputData[i + 1], inputData[i + 2]});

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
    inputData.reserve(history.size() * 3);
    for (const auto &t : history) {
        inputData.push_back(t.time);
        inputData.push_back(t.duration);
        inputData.push_back(t.note);
    }
    notifyInputDataChanged();
    return true;
}

void MusicTransformer::threadStop() {
    // TODO Maybe transfer memory_info destruction and stuff here.
}

void MusicTransformer::instrLogits(std::vector<float> &logits) {
    auto it = modelConfig.sortedActiveOutputInstruments.begin();
    auto end = modelConfig.sortedActiveOutputInstruments.end();

    if (it == end) return; // Early exit if the set is empty

    // Hide all instruments up to the lowest note of the first
    std::fill(logits.begin() + Vocab::NoteOffset,
              logits.begin() + Vocab::NoteOffset + Config::MaxPitch * (*it)
              + modelConfig.outputInstruments[*it].low,
              -std::numeric_limits<float>::infinity());

    auto prev_it = it;
    ++it;

    // Hide all instruments between the active instruments
    for (; it != end; ++it, ++prev_it) {
        std::fill(
            logits.begin() + Vocab::NoteOffset + Config::MaxPitch * (*prev_it)
            + modelConfig.outputInstruments[*prev_it].high,
            logits.begin() + Vocab::NoteOffset + Config::MaxPitch * (*it)
            + modelConfig.outputInstruments[*it].low,
            -std::numeric_limits<float>::infinity());
    }

    // Hide all instruments after the highest note of the last
    std::fill(
        logits.begin() + Vocab::NoteOffset + Config::MaxPitch * (*prev_it)
        + modelConfig.outputInstruments[*prev_it].high,
        logits.begin() + Vocab::Rest,
        -std::numeric_limits<float>::infinity());
}

/**
 * - Don't sample events in the past
 * - Don't sample events too far in the future
*/
void MusicTransformer::futureLogits(std::vector<float> &logits, const int currentTime, int32_t forceAtTime) {
    // If we need to force generation at currentTime, leave only this slot valid
    if (forceAtTime != -1 && forceAtTime < Config::MaxTime) {
        std::fill_n(logits.begin() + Vocab::TimeOffset, forceAtTime, -std::numeric_limits<float>::infinity());
        std::fill(logits.begin() + Vocab::TimeOffset + forceAtTime + 1, logits.begin() + Vocab::DurOffset,
                  -std::numeric_limits<float>::infinity());
        return;
    }

    // Force generation in the future
    if (currentTime > 0) {
        std::fill_n(logits.begin() + Vocab::TimeOffset, currentTime,
                    -std::numeric_limits<float>::infinity());
    }

    // Do not generate too far in the future (TODO: manually set to 200, but could be part of config)
    if (currentTime < (Config::MaxTime - maximumFuture)) {
        std::fill(
            logits.begin() + Vocab::TimeOffset + currentTime + maximumFuture,
            logits.begin() + Vocab::DurOffset,
            -std::numeric_limits<float>::infinity());
    }
}

void MusicTransformer::durLogits(std::vector<float> &logits) {
    std::fill_n(logits.begin() + Vocab::DurOffset, modelConfig.outputMinimumDuration,
                -std::numeric_limits<float>::infinity());
    std::fill(logits.begin() + Vocab::DurOffset + modelConfig.outputMaximumDuration + 1,
              logits.begin() + Vocab::NoteOffset,
              -std::numeric_limits<float>::infinity());
}

void safeLogits(std::vector<float> &logits, const size_t idx) {
    // Model vocab must match the Anticipatory Music Transformer layout.
    if (logits.size() < Vocab::VocabSize) {
        DBG("safeLogits: logits size " + juce::String(static_cast<int>(logits.size()))
            + " < VocabSize " + juce::String(static_cast<int>(Vocab::VocabSize)));
        return;
    }

    std::fill(logits.begin() + static_cast<std::ptrdiff_t>(Vocab::ControlOffset),
              logits.begin() + static_cast<std::ptrdiff_t>(Vocab::SpecialOffset),
              -std::numeric_limits<float>::infinity());
    std::fill(logits.begin() + static_cast<std::ptrdiff_t>(Vocab::SpecialOffset), logits.end(),
              -std::numeric_limits<float>::infinity());

    if (idx % 3 == 0) {
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(Vocab::DurOffset),
                  logits.begin() + static_cast<std::ptrdiff_t>(Vocab::NoteOffset),
                  -std::numeric_limits<float>::infinity());
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(Vocab::NoteOffset),
                  logits.begin() + static_cast<std::ptrdiff_t>(Vocab::ControlOffset),
                  -std::numeric_limits<float>::infinity());
    } else if (idx % 3 == 1) {
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(Vocab::TimeOffset),
                  logits.begin() + static_cast<std::ptrdiff_t>(Vocab::DurOffset),
                  -std::numeric_limits<float>::infinity());
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(Vocab::NoteOffset),
                  logits.begin() + static_cast<std::ptrdiff_t>(Vocab::ControlOffset),
                  -std::numeric_limits<float>::infinity());
    } else if (idx % 3 == 2) {
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(Vocab::TimeOffset),
                  logits.begin() + static_cast<std::ptrdiff_t>(Vocab::DurOffset),
                  -std::numeric_limits<float>::infinity());
        std::fill(logits.begin() + static_cast<std::ptrdiff_t>(Vocab::DurOffset),
                  logits.begin() + static_cast<std::ptrdiff_t>(Vocab::NoteOffset),
                  -std::numeric_limits<float>::infinity());
    }
}

std::vector<float> MusicTransformer::runModelAndGetLogits(std::vector<int32_t> &tokens) {
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
        DBG("Error running model: " + juce::String(exception.what()));
    }

    return {};
}

Token MusicTransformer::generateNewToken(int32_t forceAtTime) {
    if (inputData.size() % 3 != 0) {
        throw std::runtime_error("inputData must be a multiple of 3");
    }

    const int lookback = std::max(static_cast<int>(inputData.size() - 120), 0);
    std::vector history(inputData.begin() + lookback, inputData.end());
    int32_t offset = 0;
    if (!history.empty()) {
        offset = minTime(history);
    }

    // Relativize time in the history buffer
    for (size_t i = 0; i < history.size(); i++) {
        if (i % 3 == 0) {
            history[i] -= offset;
        }
    }

    history.insert(history.begin(), Vocab::Anticipate);

    Token newToken{-1, -1, -1};

    for (int i = 0; i < 3; i++) {
        std::vector<float> scores = runModelAndGetLogits(history);
        if (scores.size() < Vocab::VocabSize) {
            DBG("generateNewToken: unexpected logits size "
                + juce::String(static_cast<int>(scores.size())));
            return {-1, -1, -1};
        }
        safeLogits(scores, i % 3);
        if (i == 0) {
            // If forceAtTime is not -1, then pass it by removing the offset
            futureLogits(scores, currentTime - offset, forceAtTime != -1 ? forceAtTime - offset : -1);
        } else if (i == 1) {
            durLogits(scores);
        } else if (i == 2) {
            instrLogits(scores);
        }
        int32_t token = sampleTopP(scores, 0.9, modelConfig.outputTemperatures[i]);

        history.push_back(token);

        if (i == 0) {
            newToken.time = token + offset;
        } else if (i == 1) {
            newToken.duration = token;
        } else if (i == 2) {
            newToken.note = token;
        }
    }

    return newToken;
}
