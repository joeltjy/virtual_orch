#include "VirtualOrch/orchestration-models/InstrumentCombinations.h"
#include "VirtualOrch/OrtEnv.h"

#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_set>

namespace {

constexpr float kMaskedLogit = -std::numeric_limits<float>::infinity();

auto familyAllowedBits(const std::vector<int32_t> &instruments,
                       int32_t familyIdOffset,
                       int32_t familySize) -> int32_t {
    int32_t bits = 0;
    for (const int32_t id: instruments) {
        const int32_t rel = id - familyIdOffset;
        if (rel >= 0 && rel < familySize)
            bits |= (1 << rel);
    }
    return bits;
}

auto combosFromAllowedBits(int32_t allowedBits, int32_t familySize) -> std::vector<int32_t> {
    std::vector<int32_t> combos;
    const int32_t limit = 1 << familySize;
    // Skip combo 0 (empty set) for groups and within-family bitmasks.
    for (int32_t combo = 1; combo < limit; ++combo) {
        if ((combo & ~allowedBits) == 0)
            combos.push_back(combo);
    }
    return combos;
}

auto maskRowToAllowed(std::vector<float> &logits,
                      size_t row,
                      size_t vocab,
                      const std::vector<int32_t> &allowedTokens) -> void {
    if (row * vocab + vocab > logits.size())
        return;

    std::unordered_set<int32_t> allowed(allowedTokens.begin(), allowedTokens.end());
    float *rowData = logits.data() + row * vocab;
    for (size_t v = 0; v < vocab; ++v) {
        if (! allowed.contains(static_cast<int32_t>(v)))
            rowData[v] = kMaskedLogit;
    }
}

} // namespace

InstrumentCombinations::InstrumentCombinations() = default;

auto InstrumentCombinations::init(const char *modelPath) -> void {
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

auto InstrumentCombinations::getName() const -> std::string {
    return "InstrumentCombinations";
}

auto InstrumentCombinations::tokenHistory() const -> const std::vector<OrchestrationNote> & {
    return history;
}

auto InstrumentCombinations::tokensToInput(const std::vector<Token> &incomingTokens) const
    -> std::vector<int32_t> {
    std::vector<int32_t> newTokens;
    newTokens.reserve(incomingTokens.size() * static_cast<size_t>(tokensPerNote));
    for (const auto &token: incomingTokens) {
        newTokens.push_back(token.time);
        newTokens.push_back(durOffset + token.getRealDuration());
        newTokens.push_back(noteOffset + token.getPitch());
        newTokens.push_back(velocityOffset
                            + (token.velocity > 0 ? token.velocity : defaultVelocity));
        for (int i = 0; i < 5; ++i)
            newTokens.push_back(mask);
    }
    return newTokens;
}

auto InstrumentCombinations::prependHistoryToInput(const std::vector<int32_t> &newTokens) const
    -> std::vector<int32_t> {
    const size_t perNote = static_cast<size_t>(tokensPerNote);
    const size_t maxAligned =
        (static_cast<size_t>(maxContextLength) / perNote) * perNote;
    jassert(newTokens.size() % perNote == 0);

    std::vector<int32_t> context;
    if (newTokens.size() < maxAligned) {
        const size_t room = maxAligned - newTokens.size();
        const size_t historyTokens = (room / perNote) * perNote;
        if (historyTokens > 0) {
            if (tokenHistorySequence.size() >= historyTokens) {
                context.insert(context.end(),
                               tokenHistorySequence.end()
                                   - static_cast<std::ptrdiff_t>(historyTokens),
                               tokenHistorySequence.end());
            } else {
                context.insert(context.end(),
                               tokenHistorySequence.begin(),
                               tokenHistorySequence.end());
                const size_t rem = context.size() % perNote;
                if (rem != 0)
                    context.erase(context.begin(),
                                  context.begin() + static_cast<std::ptrdiff_t>(rem));
            }
        }

        for (const int32_t token: context)
            jassert(token != mask);
    }

    context.insert(context.end(), newTokens.begin(), newTokens.end());
    // Invariant: full new batch is always retained at the end.
    jassert(context.size() >= newTokens.size());
    jassert(context.size() % perNote == 0);
    return context;
}

auto InstrumentCombinations::runModelAndGetLogits(std::vector<int32_t> &tokens)
    -> std::pair<std::vector<float>, size_t> {
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
        return {{data, data + vocab * seq}, vocab};
    } catch (const Ort::Exception &exception) {
        DBG("Error running model: " + juce::String(exception.what()));
    }

    return {};
}

auto InstrumentCombinations::allowedTokensForFamily(size_t familyIndex,
                                                    const std::vector<int32_t> &instruments) const
    -> std::vector<int32_t> {
    static constexpr std::array<int32_t, 4> familySizes = {numStrings, numWoodwinds, numBrass, numOther};
    static constexpr std::array<int32_t, 4> familyTokenOffsets = {
        stringsOffset, woodwindOffset, brassOffset, otherOffset};
    static constexpr std::array<int32_t, 4> familyIdOffsets = {
        0,
        numStrings,
        numStrings + numWoodwinds,
        numStrings + numWoodwinds + numBrass,
    };

    jassert(familyIndex < 4);
    const int32_t familySize = familySizes[familyIndex];
    const int32_t tokenOffset = familyTokenOffsets[familyIndex];
    const int32_t idOffset = familyIdOffsets[familyIndex];
    const int32_t allowedBits = familyAllowedBits(instruments, idOffset, familySize);

    if (allowedBits == 0)
        return {endNote};

    std::vector<int32_t> tokens;
    for (const int32_t combo: combosFromAllowedBits(allowedBits, familySize))
        tokens.push_back(tokenOffset + combo);
    return tokens;
}

auto InstrumentCombinations::allowedTokensForGroups(const std::vector<int32_t> &instruments) const
    -> std::vector<int32_t> {
    static constexpr int32_t numFamilies = 4;
    static constexpr std::array<int32_t, 4> familySizes = {numStrings, numWoodwinds, numBrass, numOther};
    static constexpr std::array<int32_t, 4> familyIdOffsets = {
        0,
        numStrings,
        numStrings + numWoodwinds,
        numStrings + numWoodwinds + numBrass,
    };

    int32_t activeFamilies = 0;
    for (size_t family = 0; family < numFamilies; ++family) {
        if (familyAllowedBits(instruments, familyIdOffsets[family], familySizes[family]) != 0)
            activeFamilies |= (1 << static_cast<int32_t>(family));
    }

    std::vector<int32_t> tokens;
    for (const int32_t combo: combosFromAllowedBits(activeFamilies, numFamilies))
        tokens.push_back(groupsOffset + combo);
    return tokens;
}

auto InstrumentCombinations::maskLogits(std::vector<float> &logits,
                                        size_t numTokens,
                                        size_t vocab,
                                        const std::vector<int32_t> &instruments) const -> void {
    if (numTokens == 0 || vocab == 0 || logits.size() != numTokens * vocab)
        return;

    const auto groupsAllowed = allowedTokensForGroups(instruments);
    std::array<std::vector<int32_t>, 4> familyAllowed{};
    for (size_t family = 0; family < 4; ++family)
        familyAllowed[family] = allowedTokensForFamily(family, instruments);

    const size_t numNotes = numTokens / static_cast<size_t>(tokensPerNote);
    for (size_t note = 0; note < numNotes; ++note) {
        const size_t base = note * static_cast<size_t>(tokensPerNote);
        maskRowToAllowed(logits, base + 4, vocab, groupsAllowed);
        for (size_t family = 0; family < 4; ++family)
            maskRowToAllowed(logits, base + 5 + family, vocab, familyAllowed[family]);
    }
}

auto InstrumentCombinations::sample(const std::vector<int32_t> &maskedTokens,
                                    std::vector<float> &logits,
                                    size_t vocab) const -> std::vector<int32_t> {
    const size_t numTokens = maskedTokens.size();
    jassert(numTokens % static_cast<size_t>(tokensPerNote) == 0);
    jassert(logits.size() == numTokens * vocab);

    std::vector<int32_t> sampled;
    sampled.reserve(numTokens);

    for (size_t t = 0; t < numTokens; ++t) {
        const size_t posInNote = t % static_cast<size_t>(tokensPerNote);
        if (posInNote < 4) {
            sampled.push_back(maskedTokens[t]);
            continue;
        }

        std::vector<float> row(
            logits.begin() + static_cast<std::ptrdiff_t>(t * vocab),
            logits.begin() + static_cast<std::ptrdiff_t>((t + 1) * vocab));
        sampled.push_back(sampleTopP(row, 1.0f, sampleTemperature));
    }

    return sampled;
}

auto InstrumentCombinations::encodedCombinationToInstruments(const std::vector<int32_t> &sampledTokens) const
    -> std::vector<OrchestrationNote> {
    jassert(sampledTokens.size() % static_cast<size_t>(tokensPerNote) == 0);

    const size_t numNotes = sampledTokens.size() / static_cast<size_t>(tokensPerNote);
    std::vector<OrchestrationNote> notes;
    notes.reserve(numNotes); // might grow when a combo expands to multiple instruments

    const std::array<int32_t, 4> familyOffsets = {stringsOffset, woodwindOffset, brassOffset, otherOffset};
    const std::array<int32_t, 4> familySizes = {numStrings, numWoodwinds, numBrass, numOther};
    const std::array<int32_t, 4> instrumentIdOffsets = {
        0,
        numStrings,
        numStrings + numWoodwinds,
        numStrings + numWoodwinds + numBrass,
    };

    for (size_t noteIndex = 0; noteIndex < numNotes; ++noteIndex) {
        const size_t base = noteIndex * static_cast<size_t>(tokensPerNote);
        const int32_t onset = sampledTokens[base];
        const int32_t durToken = sampledTokens[base + 1];
        const int32_t noteToken = sampledTokens[base + 2];
        const int32_t velToken = sampledTokens[base + 3];
        // sampledTokens[base + 4]: groups token (unused for instrument expansion)

        const int32_t pitch = noteToken >= noteOffset ? noteToken - noteOffset : noteToken;
        const int32_t velocity = velToken >= velocityOffset ? velToken - velocityOffset : velToken;

        std::vector<int32_t> allInstruments;
        for (size_t family = 0; family < 4; ++family) {
            const int32_t familyToken = sampledTokens[base + 5 + family];
            if (familyToken == endNote)
                continue;

            const int32_t groupCombination = familyToken - familyOffsets[family];
            for (int32_t bit = 0; bit < familySizes[family]; ++bit) {
                if ((groupCombination & (1 << bit)) != 0)
                    allInstruments.push_back(instrumentIdOffsets[family] + bit);
            }
        }

        for (const int32_t instrument: allInstruments) {
            OrchestrationNote assigned;
            assigned.token.time = onset;
            assigned.token.duration = durToken;
            assigned.token.note = static_cast<int32_t>(Vocab::NoteOffset) + pitch;
            assigned.velocity = velocity;
            assigned.token.velocity = assigned.velocity;
            assigned.localInstrumentId = instrument;
            notes.push_back(assigned);
        }
    }

    return notes;
}

auto InstrumentCombinations::updateHistories(const std::vector<OrchestrationNote> &notes,
                                             const std::vector<int32_t> &sampledTokens) -> void {
    history.insert(history.end(), notes.begin(), notes.end());
    tokenHistorySequence.insert(tokenHistorySequence.end(),
                                sampledTokens.begin(),
                                sampledTokens.end());
}

auto InstrumentCombinations::getOutput(const std::vector<Token> &incomingTokens,
                                       const std::vector<int32_t> &instruments,
                                       const ConditioningSignal &conditioningSignal)
    -> std::vector<OrchestrationNote> {
    juce::ignoreUnused(conditioningSignal);

    if (incomingTokens.empty() || session == nullptr)
        return {};

    auto newTokens = tokensToInput(incomingTokens);
    const size_t perNote = static_cast<size_t>(tokensPerNote);
    const size_t maxAligned =
        (static_cast<size_t>(maxContextLength) / perNote) * perNote;
    if (newTokens.size() > maxAligned) {
        newTokens.erase(newTokens.begin(),
                        newTokens.end() - static_cast<std::ptrdiff_t>(maxAligned));
    }

    auto context = prependHistoryToInput(newTokens);

    auto [logitsAll, vocab] = runModelAndGetLogits(context);
    if (logitsAll.empty() || vocab == 0 || logitsAll.size() != context.size() * vocab)
        return {};

    const size_t newTokenCount = newTokens.size();
    jassert(context.size() >= newTokenCount);
    const size_t historyTokenCount = context.size() - newTokenCount;
    std::vector<float> logits(
        logitsAll.begin() + static_cast<std::ptrdiff_t>(historyTokenCount * vocab),
        logitsAll.end());
    if (logits.size() != newTokenCount * vocab)
        return {};

    maskLogits(logits, newTokenCount, vocab, instruments);
    const auto sampled = sample(newTokens, logits, vocab);
    const auto result = encodedCombinationToInstruments(sampled);
    updateHistories(result, sampled);
    return result;
}
