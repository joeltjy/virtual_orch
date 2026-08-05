#include "VirtualOrch/orchestration-models/TestModel.h"
#include "VirtualOrch/OrtEnv.h"

#include <numeric>

TestModel::TestModel() = default;

auto TestModel::init(const char *modelPath) -> void {
    Ort::SessionOptions sessionOptions;
    session = std::make_unique<Ort::Session>(sharedOrtEnv(), modelPath, sessionOptions);

    allocatedInputNames.clear();
    allocatedOutputNames.clear();
    const Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session->GetInputCount(); ++i)
        allocatedInputNames.emplace_back(session->GetInputNameAllocated(i, allocator).get());
    for (size_t i = 0; i < session->GetOutputCount(); ++i)
        allocatedOutputNames.emplace_back(session->GetOutputNameAllocated(i, allocator).get());
}

auto TestModel::getName() const -> std::string {
    return "TestModel";
}

auto TestModel::tokenHistory() const -> const std::vector<OrchestrationNote> & {
    return history;
}

auto TestModel::tokensToInput(const std::vector<Token> &incomingTokens) const -> std::vector<int32_t> {
    std::vector<int32_t> newTokens;
    newTokens.reserve(incomingTokens.size() * 4);
    for (const auto &token: incomingTokens) {
        newTokens.push_back(token.time);
        newTokens.push_back(durOffset + token.getRealDuration());
        newTokens.push_back(noteOffset + token.getPitch());
        newTokens.push_back(velocityOffset
                            + (token.velocity > 0 ? token.velocity : defaultVelocity));
    }
    return newTokens;
}

auto TestModel::prependHistoryToInput(const std::vector<int32_t> &newTokens) const
    -> std::vector<int32_t> {
    static constexpr size_t tokensPerNote = 4;
    std::vector<int32_t> context;

    if (static_cast<int32_t>(newTokens.size()) < maxContextLength) {
        const size_t need = static_cast<size_t>(maxContextLength) - newTokens.size();
        if (tokenHistorySequence.size() >= need) {
            context.insert(context.end(),
                           tokenHistorySequence.end() - static_cast<std::ptrdiff_t>(need),
                           tokenHistorySequence.end());
        } else {
            context.insert(context.end(), tokenHistorySequence.begin(), tokenHistorySequence.end());
        }
        const size_t rem = context.size() % tokensPerNote;
        if (rem != 0)
            context.erase(context.begin(), context.begin() + static_cast<std::ptrdiff_t>(rem));
    }

    context.insert(context.end(), newTokens.begin(), newTokens.end());
    if (context.size() > static_cast<size_t>(maxContextLength)) {
        context.erase(context.begin(),
                      context.end() - static_cast<std::ptrdiff_t>(maxContextLength));
        const size_t rem = context.size() % tokensPerNote;
        if (rem != 0)
            context.erase(context.begin(), context.begin() + static_cast<std::ptrdiff_t>(rem));
    }

    return context;
}

auto TestModel::runModelAndGetLogits(std::vector<int32_t> &tokens) -> std::vector<float> {
    const std::vector input_shape{1, static_cast<int64_t>(tokens.size())};

    std::vector<int64_t> token_ids(tokens.begin(), tokens.end());
    std::vector<int64_t> position_ids(tokens.size(), 0);
    std::iota(position_ids.begin(), position_ids.end(), 0);
    std::vector<float> attention_mask(tokens.size(), 1.0f);

    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        memoryInfo, token_ids.data(), token_ids.size(), input_shape.data(), input_shape.size());
    Ort::Value position_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        memoryInfo, position_ids.data(), position_ids.size(), input_shape.data(), input_shape.size());
    Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<float>(
        memoryInfo, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size());

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));
    input_tensors.push_back(std::move(position_ids_tensor));
    input_tensors.push_back(std::move(attention_mask_tensor));

    std::vector<const char *> inputNames;
    std::vector<const char *> outputNames;
    for (auto &allocatedInputName: allocatedInputNames)
        inputNames.push_back(allocatedInputName.c_str());
    for (auto &allocatedOutputName: allocatedOutputNames)
        outputNames.push_back(allocatedOutputName.c_str());

    try {
        auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                           inputNames.data(),
                                           input_tensors.data(),
                                           input_tensors.size(),
                                           outputNames.data(),
                                           outputNames.size());

        Ort::Value &logits_tensor = output_tensors.front();
        const auto count = logits_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
        const float *data = logits_tensor.GetTensorMutableData<float>();
        return {data, data + count};
    } catch (const Ort::Exception &exception) {
        DBG("Error running model: " + juce::String(exception.what()));
    }

    return {};
}

auto TestModel::maskLogits(std::vector<float> &logits, size_t numNotes, size_t numInstruments) const
    -> void {
    juce::ignoreUnused(logits, numNotes, numInstruments);
}

auto TestModel::sample(const std::vector<Token> &incomingTokens,
                       std::vector<float> &newNoteLogits,
                       size_t numInstruments) const -> std::vector<OrchestrationNote> {
    const size_t n = incomingTokens.size();
    jassert(newNoteLogits.size() == n * numInstruments);

    std::vector<OrchestrationNote> result;
    result.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        std::vector<float> noteLogits(
            newNoteLogits.begin() + static_cast<std::ptrdiff_t>(i * numInstruments),
            newNoteLogits.begin() + static_cast<std::ptrdiff_t>((i + 1) * numInstruments));

        OrchestrationNote assigned;
        assigned.token = incomingTokens[i];
        assigned.velocity = incomingTokens[i].velocity > 0 ? incomingTokens[i].velocity
                                                           : defaultVelocity;
        assigned.localInstrumentId = sampleTopP(noteLogits, 1.0f, sampleTemperature);
        result.push_back(assigned);
    }
    return result;
}

auto TestModel::updateHistories(const std::vector<OrchestrationNote> &notes,
                                const std::vector<int32_t> &newTokens) -> void {
    history.insert(history.end(), notes.begin(), notes.end());
    tokenHistorySequence.insert(tokenHistorySequence.end(), newTokens.begin(), newTokens.end());
}

auto TestModel::getOutput(const std::vector<Token> &incomingTokens,
                          const std::vector<int32_t> &instruments,
                          const ConditioningSignal &conditioningSignal)
    -> std::vector<OrchestrationNote> {
    juce::ignoreUnused(instruments, conditioningSignal);

    if (incomingTokens.empty() || session == nullptr)
        return {};

    const auto newTokens = tokensToInput(incomingTokens);
    auto context = prependHistoryToInput(newTokens);

    auto logitsFlat = runModelAndGetLogits(context);
    if (logitsFlat.empty())
        return {};

    const size_t n = incomingTokens.size();
    const size_t numNotesInContext = context.size() / 4;
    if (numNotesInContext == 0 || logitsFlat.size() % numNotesInContext != 0 || numNotesInContext < n)
        return {};

    const size_t numInstruments = logitsFlat.size() / numNotesInContext;
    const size_t firstNewNote = numNotesInContext - n;
    std::vector<float> newNoteLogits(
        logitsFlat.begin() + static_cast<std::ptrdiff_t>(firstNewNote * numInstruments),
        logitsFlat.end());

    maskLogits(newNoteLogits, n, numInstruments);
    const auto result = sample(incomingTokens, newNoteLogits, numInstruments);
    updateHistories(result, newTokens);
    return result;
}
