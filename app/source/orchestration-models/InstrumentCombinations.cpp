#include "VirtualOrch/orchestration-models/InstrumentCombinations.h"
#include "VirtualOrch/InstrumentConstants.h"
#include "VirtualOrch/NoteWindow.h"
#include "VirtualOrch/OrtEnv.h"
#include "VirtualOrch/SamplingRelativeTime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

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

    float *rowData = logits.data() + row * vocab;
    std::vector<std::pair<size_t, float>> keep;
    keep.reserve(allowedTokens.size());
    for (const int32_t id: allowedTokens) {
        if (id < 0)
            continue;
        const auto idx = static_cast<size_t>(id);
        if (idx < vocab)
            keep.emplace_back(idx, rowData[idx]);
    }
    std::fill(rowData, rowData + vocab, kMaskedLogit);
    for (const auto &[idx, value]: keep)
        rowData[idx] = value;
}

} // namespace

InstrumentCombinations::InstrumentCombinations() = default;

auto InstrumentCombinations::init(const char *modelPath) -> void {
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

    // Token offsets are hardcoded per checkpoint. If they disagree with the graph, ids like
    // `mask` fall outside the embedding and the CUDA gather faults instead of erroring.
    modelVocabSize = 0;
    if (session->GetOutputCount() > 0) {
        const auto shape = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (! shape.empty() && shape.back() > 0)
            modelVocabSize = static_cast<size_t>(shape.back());
    }

    if (modelVocabSize != 0 && modelVocabSize != static_cast<size_t>(expectedVocabSize)) {
        DBG("InstrumentCombinations vocab mismatch: model " + juce::String(modelVocabSize)
            + " vs expected " + juce::String(expectedVocabSize));
    }
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
    if (modelVocabSize != 0 && modelVocabSize != static_cast<size_t>(expectedVocabSize)) {
        reportSamplingError("IC token layout expects vocab " + juce::String(expectedVocabSize)
                            + " but model has " + juce::String(modelVocabSize));
        return {};
    }

    const std::vector input_shape{1, static_cast<int64_t>(tokens.size())};

    std::vector<int64_t> token_ids(tokens.begin(), tokens.end());
    std::vector<int64_t> position_ids(tokens.size(), 0);
    std::iota(position_ids.begin(), position_ids.end(), 0);
    std::vector<float> attention_mask(tokens.size(), 1.0f);

    // Feed only what the graph declares: exports vary (input_ids alone, or with
    // position_ids/attention_mask). Name count and tensor count must always match.
    std::vector<const char *> inputNames;
    std::vector<Ort::Value> input_tensors;
    for (const auto &name: allocatedInputNames) {
        if (name == "input_ids") {
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo,
                                                                     token_ids.data(),
                                                                     token_ids.size(),
                                                                     input_shape.data(),
                                                                     input_shape.size()));
        } else if (name == "position_ids") {
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo,
                                                                     position_ids.data(),
                                                                     position_ids.size(),
                                                                     input_shape.data(),
                                                                     input_shape.size()));
        } else if (name == "attention_mask") {
            input_tensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo,
                                                                   attention_mask.data(),
                                                                   attention_mask.size(),
                                                                   input_shape.data(),
                                                                   input_shape.size()));
        } else {
            reportSamplingError("Unsupported model input: " + juce::String(name));
            return {};
        }
        inputNames.push_back(name.c_str());
    }

    // Only logits are consumed; asking for the KV-cache outputs would just cost time.
    std::vector<const char *> outputNames;
    if (! allocatedOutputNames.empty())
        outputNames.push_back(allocatedOutputNames.front().c_str());

    try {
        std::vector<Ort::Value> output_tensors;
        {
            const ScopedMs onnxMs(getOutputTimings != nullptr ? &getOutputTimings->onnx : nullptr);
            output_tensors = session->Run(Ort::RunOptions{nullptr},
                                          inputNames.data(),
                                          input_tensors.data(),
                                          inputNames.size(),
                                          outputNames.data(),
                                          outputNames.size());
        }

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

        const ScopedMs logitsMs(getOutputTimings != nullptr ? &getOutputTimings->logits : nullptr);
        const float *data = logits_tensor.GetTensorMutableData<float>();
        return {{data, data + vocab * seq}, vocab};
    } catch (const Ort::Exception &exception) {
        const auto detail = juce::String(exception.what());
        DBG("Error running model: " + detail);
        reportSamplingError(detail);
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
        tokens.push_back(groupsTokenForCombo(combo));
    return tokens;
}

auto InstrumentCombinations::maskLogits(std::vector<float> &logits,
                                        size_t numTokens,
                                        size_t vocab,
                                        const std::vector<int32_t> &instruments,
                                        const std::vector<int32_t> &newTokens) const -> void {
    if (numTokens == 0 || vocab == 0 || logits.size() != numTokens * vocab)
        return;
    if (newTokens.size() < numTokens)
        return;

    const size_t numNotes = numTokens / static_cast<size_t>(tokensPerNote);
    for (size_t note = 0; note < numNotes; ++note) {
        const size_t base = note * static_cast<size_t>(tokensPerNote);
        const int32_t noteToken = newTokens[base + 2];
        const int32_t pitch =
            noteToken >= noteOffset ? noteToken - noteOffset : noteToken;
        const auto filtered =
            InstrumentConstants::filterLocalInstrumentsForPitch(instruments, pitch);

        std::vector<int32_t> groupsAllowed;
        std::array<std::vector<int32_t>, 4> familyAllowed{};
        if (filtered.empty()) {
            // No id encodes "no families", so force every family slot to endNote instead:
            // decode then yields no instruments whatever groups says.
            groupsAllowed = {groupsTokenForCombo(1)};
            for (size_t family = 0; family < 4; ++family)
                familyAllowed[family] = {endNote};
        } else {
            groupsAllowed = allowedTokensForGroups(filtered);
            for (size_t family = 0; family < 4; ++family)
                familyAllowed[family] = allowedTokensForFamily(family, filtered);
        }

        maskRowToAllowed(logits, base + 4, vocab, groupsAllowed);
        for (size_t family = 0; family < 4; ++family)
            maskRowToAllowed(logits, base + 5 + family, vocab, familyAllowed[family]);
    }
}

auto InstrumentCombinations::applyProportionBias(
    std::vector<float> &logits,
    size_t numTokens,
    size_t vocab,
    const OrchestrationBalanceTracker::BiasView &bias) const -> void {
    if (! bias.apply || bias.totalNotes == 0 || numTokens == 0 || vocab == 0)
        return;
    if (logits.size() != numTokens * vocab)
        return;

    static constexpr std::array<int32_t, 4> familyOffsets = {
        stringsOffset, woodwindOffset, brassOffset, otherOffset};
    static constexpr std::array<int32_t, 4> familySizes = {
        numStrings, numWoodwinds, numBrass, numOther};
    static constexpr std::array<int32_t, 4> instrumentIdOffsets = {
        0,
        numStrings,
        numStrings + numWoodwinds,
        numStrings + numWoodwinds + numBrass,
    };

    const size_t numNotes = numTokens / static_cast<size_t>(tokensPerNote);
    for (size_t note = 0; note < numNotes; ++note) {
        const size_t base = note * static_cast<size_t>(tokensPerNote);

        // Groups decides which families fire; without biasing it, family-slot boosts for
        // newly added (low-p) families never take effect.
        {
            float *rowData = logits.data() + (base + 4) * vocab;
            for (int32_t combo = 1; combo <= numGroupTokens; ++combo) {
                const int32_t token = groupsTokenForCombo(combo);
                if (token < 0 || static_cast<size_t>(token) >= vocab)
                    continue;
                if (! std::isfinite(rowData[static_cast<size_t>(token)]))
                    continue;

                std::vector<int32_t> instrumentsInFamilies;
                for (size_t family = 0; family < 4; ++family) {
                    if ((combo & (1 << static_cast<int32_t>(family))) == 0)
                        continue;
                    for (int32_t bit = 0; bit < familySizes[family]; ++bit)
                        instrumentsInFamilies.push_back(instrumentIdOffsets[family] + bit);
                }
                rowData[static_cast<size_t>(token)] -= bias.comboPenalty(instrumentsInFamilies);
            }
        }

        for (size_t family = 0; family < 4; ++family) {
            const size_t row = base + 5 + family;
            float *rowData = logits.data() + row * vocab;
            const int32_t offset = familyOffsets[family];
            const int32_t familySize = familySizes[family];
            const int32_t idOffset = instrumentIdOffsets[family];
            const int32_t numCombos = 1 << familySize;

            for (int32_t combo = 1; combo < numCombos; ++combo) {
                const int32_t token = offset + combo;
                if (token < 0 || static_cast<size_t>(token) >= vocab)
                    continue;
                if (! std::isfinite(rowData[static_cast<size_t>(token)]))
                    continue;

                std::vector<int32_t> instrumentsInCombo;
                instrumentsInCombo.reserve(static_cast<size_t>(familySize));
                for (int32_t bit = 0; bit < familySize; ++bit) {
                    if ((combo & (1 << bit)) != 0)
                        instrumentsInCombo.push_back(idOffset + bit);
                }
                rowData[static_cast<size_t>(token)] -= bias.comboPenalty(instrumentsInCombo);
            }
        }
    }
}

auto InstrumentCombinations::publishSingletonInstrumentLogits(const std::vector<float> &logits,
                                                              size_t numTokens,
                                                              size_t vocab,
                                                              bool biasApplied) const -> void {
    if (instrumentLogitSink == nullptr || numTokens == 0 || vocab == 0)
        return;
    if (logits.size() != numTokens * vocab)
        return;

    static constexpr std::array<int32_t, 4> familyOffsets = {
        stringsOffset, woodwindOffset, brassOffset, otherOffset};
    static constexpr std::array<int32_t, 4> familySizes = {
        numStrings, numWoodwinds, numBrass, numOther};
    static constexpr std::array<int32_t, 4> instrumentIdOffsets = {
        0,
        numStrings,
        numStrings + numWoodwinds,
        numStrings + numWoodwinds + numBrass,
    };

    InstrumentLogitSnapshot snap;
    snap.biasApplied = biasApplied;

    const size_t numNotes = numTokens / static_cast<size_t>(tokensPerNote);
    const size_t note = numNotes > 0 ? numNotes - 1 : 0;
    const size_t base = note * static_cast<size_t>(tokensPerNote);

    for (size_t family = 0; family < 4; ++family) {
        const float *rowData = logits.data() + (base + 5 + family) * vocab;
        const int32_t offset = familyOffsets[family];
        const int32_t familySize = familySizes[family];
        const int32_t idOffset = instrumentIdOffsets[family];
        for (int32_t bit = 0; bit < familySize; ++bit) {
            const int32_t localId = idOffset + bit;
            if (localId < 0
                || static_cast<size_t>(localId) >= InstrumentLogitSnapshot::kNumInstruments)
                continue;
            const int32_t token = offset + (1 << bit); // singleton combo
            if (token < 0 || static_cast<size_t>(token) >= vocab)
                continue;
            const float value = rowData[static_cast<size_t>(token)];
            if (! std::isfinite(value))
                continue;
            snap.logits[static_cast<size_t>(localId)] = value;
            snap.valid[static_cast<size_t>(localId)] = true;
        }
    }

    *instrumentLogitSink = snap;
}

auto InstrumentCombinations::recordBalanceFromSampled(
    OrchestrationBalanceTracker &balance,
    const std::vector<int32_t> &sampledTokens) const -> void {
    if (sampledTokens.size() % static_cast<size_t>(tokensPerNote) != 0)
        return;

    const std::array<int32_t, 4> familyOffsets = {
        stringsOffset, woodwindOffset, brassOffset, otherOffset};
    const std::array<int32_t, 4> familySizes = {numStrings, numWoodwinds, numBrass, numOther};
    const std::array<int32_t, 4> instrumentIdOffsets = {
        0,
        numStrings,
        numStrings + numWoodwinds,
        numStrings + numWoodwinds + numBrass,
    };

    const size_t numNotes = sampledTokens.size() / static_cast<size_t>(tokensPerNote);
    for (size_t noteIndex = 0; noteIndex < numNotes; ++noteIndex) {
        const size_t base = noteIndex * static_cast<size_t>(tokensPerNote);
        const int32_t groupsCombo = comboFromGroupsToken(sampledTokens[base + 4]);

        std::vector<int32_t> instrumentsOnNote;
        for (size_t family = 0; family < 4; ++family) {
            if ((groupsCombo & (1 << static_cast<int32_t>(family))) == 0)
                continue;

            const int32_t familyToken = sampledTokens[base + 5 + family];
            if (familyToken == endNote)
                continue;

            const int32_t groupCombination = familyToken - familyOffsets[family];
            for (int32_t bit = 0; bit < familySizes[family]; ++bit) {
                if ((groupCombination & (1 << bit)) != 0)
                    instrumentsOnNote.push_back(instrumentIdOffsets[family] + bit);
            }
        }
        balance.recordNote(instrumentsOnNote);
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

        LogitSpan row{logits.data() + t * vocab, vocab};
        sampled.push_back(sampleTopP(row, 1.0f, sampleTemperature));
    }

    // Groups decides which families are active; clear inactive family slots for history/decode.
    const size_t numNotes = sampled.size() / static_cast<size_t>(tokensPerNote);
    for (size_t noteIndex = 0; noteIndex < numNotes; ++noteIndex) {
        const size_t base = noteIndex * static_cast<size_t>(tokensPerNote);
        const int32_t groupsCombo = comboFromGroupsToken(sampled[base + 4]);
        for (size_t family = 0; family < 4; ++family) {
            if ((groupsCombo & (1 << static_cast<int32_t>(family))) == 0)
                sampled[base + 5 + family] = endNote;
        }
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
        const int32_t groupsCombo = comboFromGroupsToken(sampledTokens[base + 4]);

        const int32_t pitch = noteToken >= noteOffset ? noteToken - noteOffset : noteToken;
        const int32_t velocity = velToken >= velocityOffset ? velToken - velocityOffset : velToken;

        std::vector<int32_t> allInstruments;
        for (size_t family = 0; family < 4; ++family) {
            if ((groupsCombo & (1 << static_cast<int32_t>(family))) == 0)
                continue;

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
    NoteWindow::trimToLastNotes(history);
    NoteWindow::trimStridedToLastNotes(tokenHistorySequence,
                                       static_cast<size_t>(tokensPerNote));
}

auto InstrumentCombinations::getOutput(const std::vector<Token> &incomingTokens,
                                       const std::vector<int32_t> &instruments,
                                       const ConditioningSignal &conditioningSignal,
                                       OrchestrationBalanceTracker *balance,
                                       const OrchestrationBalanceTracker::BiasView *bias)
    -> std::vector<OrchestrationNote> {
    juce::ignoreUnused(conditioningSignal);

    if (incomingTokens.empty() || session == nullptr)
        return {};

    if (getOutputTimings != nullptr)
        ++getOutputTimings->getOutputCalls;

    std::vector<int32_t> newTokens;
    std::vector<int32_t> context;
    {
        const ScopedMs prepMs(getOutputTimings != nullptr ? &getOutputTimings->prep : nullptr);
        newTokens = tokensToInput(incomingTokens);
        const size_t perNote = static_cast<size_t>(tokensPerNote);
        const size_t maxAligned =
            (static_cast<size_t>(maxContextLength) / perNote) * perNote;
        if (newTokens.size() > maxAligned) {
            newTokens.erase(newTokens.begin(),
                            newTokens.end() - static_cast<std::ptrdiff_t>(maxAligned));
        }

        context = prependHistoryToInput(newTokens);

        const int32_t timeOffset =
            SamplingRelativeTime::minStridedOnset(context, perNote);
        SamplingRelativeTime::relativizeStridedOnsets(context, perNote, timeOffset);
        if (getOutputTimings != nullptr)
            getOutputTimings->contextTokens =
                juce::jmax(getOutputTimings->contextTokens, static_cast<int>(context.size()));
    }

    auto [logitsAll, vocab] = runModelAndGetLogits(context);
    if (logitsAll.empty() || vocab == 0 || logitsAll.size() != context.size() * vocab)
        return {};

    const size_t newTokenCount = newTokens.size();
    jassert(context.size() >= newTokenCount);
    const size_t historyTokenCount = context.size() - newTokenCount;
    std::vector<float> logits;
    {
        const ScopedMs logitsMs(getOutputTimings != nullptr ? &getOutputTimings->logits : nullptr);
        logits.assign(logitsAll.begin() + static_cast<std::ptrdiff_t>(historyTokenCount * vocab),
                      logitsAll.end());
        if (logits.size() != newTokenCount * vocab)
            return {};
    }

    {
        const ScopedMs maskMs(getOutputTimings != nullptr ? &getOutputTimings->mask : nullptr);
        maskLogits(logits, newTokenCount, vocab, instruments, newTokens);
        if (bias != nullptr)
            applyProportionBias(logits, newTokenCount, vocab, *bias);
        publishSingletonInstrumentLogits(logits,
                                         newTokenCount,
                                         vocab,
                                         bias != nullptr && bias->apply);
    }

    std::vector<int32_t> sampled;
    {
        const ScopedMs sampleMs(getOutputTimings != nullptr ? &getOutputTimings->sample : nullptr);
        sampled = sample(newTokens, logits, vocab);
    }

    std::vector<OrchestrationNote> result;
    {
        const ScopedMs decodeMs(getOutputTimings != nullptr ? &getOutputTimings->decode : nullptr);
        result = encodedCombinationToInstruments(sampled);
        updateHistories(result, sampled);
        if (balance != nullptr)
            recordBalanceFromSampled(*balance, sampled);
    }
    return result;
}
