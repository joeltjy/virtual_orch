#include "VirtualOrch/orchestration-models/IodPretrained.h"

#include "VirtualOrch/NoteWindow.h"
#include "VirtualOrch/OrtEnv.h"
#include "VirtualOrch/OrchLoopProfile.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

auto findNameIndex(const std::vector<std::string> &names, const char *want) -> int {
    for (size_t i = 0; i < names.size(); ++i) {
        if (names[i] == want)
            return static_cast<int>(i);
    }
    return -1;
}

auto argmaxRange(const float *logits, int32_t lo, int32_t hi) -> int32_t {
    int32_t best = lo;
    float bestVal = kNegInf;
    bool any = false;
    for (int32_t id = lo; id < hi; ++id) {
        const float v = logits[id];
        if (! std::isfinite(v))
            continue;
        if (! any || v > bestVal) {
            bestVal = v;
            best = id;
            any = true;
        }
    }
    return best;
}

auto greedySampleRange(const float *vocabLogits,
                       int32_t lo,
                       int32_t hi,
                       int32_t forbidBits,
                       int32_t forbidGroupsFamilies,
                       bool excludeZero) -> int32_t {
    const int32_t start = excludeZero ? lo + 1 : lo;
    std::vector<float> local(static_cast<size_t>(hi - start), kNegInf);
    bool any = false;
    for (int32_t id = start; id < hi; ++id) {
        const int32_t value = id - lo;
        if (forbidBits != 0 && (value & forbidBits) != 0)
            continue;
        if (forbidGroupsFamilies != 0) {
            const int32_t groupsMask = value + 1; // GROUPS: offset k → mask k+1
            if ((groupsMask & forbidGroupsFamilies) != 0)
                continue;
        }
        const float v = vocabLogits[id];
        if (! std::isfinite(v))
            continue;
        local[static_cast<size_t>(id - start)] = v;
        any = true;
    }
    if (! any) {
        if (forbidGroupsFamilies != 0) {
            for (int32_t groupsMask = 1; groupsMask <= hi - lo; ++groupsMask) {
                if ((groupsMask & forbidGroupsFamilies) == 0)
                    return lo + (groupsMask - 1);
            }
        }
        return start < hi ? start : lo;
    }
    return start + argmaxRange(local.data(), 0, static_cast<int32_t>(local.size()));
}

auto logitsIndex(int32_t noteIdx, int32_t step) -> size_t {
    return (static_cast<size_t>(noteIdx) * static_cast<size_t>(IodPretrainedTypes::comboLen)
            + static_cast<size_t>(step))
           * static_cast<size_t>(IodPretrainedTypes::comboVocabSize);
}

} // namespace

IodPretrained::IodPretrained() = default;

auto IodPretrained::init(const char *modelPath) -> void {
    session.reset();
    juce::String lastError;

    const auto tryCreate = [&](Ort::SessionOptions &opts) {
        session = std::make_unique<Ort::Session>(sharedOrtEnv(), modelPath, opts);
    };

#ifdef __linux__
    try {
        Ort::SessionOptions cudaOptions;
        OrtCUDAProviderOptions cudaProvider{};
        cudaOptions.AppendExecutionProvider_CUDA(cudaProvider);
        tryCreate(cudaOptions);
    } catch (const Ort::Exception &ex) {
        lastError = ex.what();
        session.reset();
    }
#endif

    if (session == nullptr) {
        try {
            Ort::SessionOptions cpuOptions;
            tryCreate(cpuOptions);
        } catch (const Ort::Exception &ex) {
            lastError = ex.what();
            session.reset();
            reportSamplingError("IodPretrained init failed: " + lastError);
            return;
        }
    }

    allocatedInputNames.clear();
    allocatedOutputNames.clear();
    const Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session->GetInputCount(); ++i)
        allocatedInputNames.emplace_back(session->GetInputNameAllocated(i, allocator).get());
    for (size_t i = 0; i < session->GetOutputCount(); ++i)
        allocatedOutputNames.emplace_back(session->GetOutputNameAllocated(i, allocator).get());

    modelVocabSize = 0;
    if (session->GetOutputCount() > 0) {
        const auto shape = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (! shape.empty() && shape.back() > 0)
            modelVocabSize = static_cast<size_t>(shape.back());
    }

    if (modelVocabSize != 0
        && modelVocabSize != static_cast<size_t>(IodPretrainedTypes::comboVocabSize)) {
        const auto detail = "IodPretrained vocab mismatch: model "
                            + juce::String(static_cast<int>(modelVocabSize)) + " vs expected "
                            + juce::String(IodPretrainedTypes::comboVocabSize);
        DBG(detail);
        reportSamplingError(detail);
    }
}

auto IodPretrained::getName() const -> std::string {
    return "IodPretrained";
}

auto IodPretrained::tokenHistory() const -> const std::vector<OrchestrationNote> & {
    return history;
}

auto IodPretrained::runLogits(IodPretrainedTypes::EncoderWindow &window) -> std::vector<float> {
    if (session == nullptr)
        return {};

    const int inIds = findNameIndex(allocatedInputNames, "input_ids");
    const int inAttn = findNameIndex(allocatedInputNames, "attention_mask");
    const int inCombo = findNameIndex(allocatedInputNames, "combo_in");
    const int inValid = findNameIndex(allocatedInputNames, "note_valid");
    const int outLogits = findNameIndex(allocatedOutputNames, "logits");
    if (inIds < 0 || inAttn < 0 || inCombo < 0 || inValid < 0 || outLogits < 0) {
        reportSamplingError("IodPretrained: missing ONNX I/O names");
        return {};
    }

    const std::array<int64_t, 2> idsShape{1, IodPretrainedTypes::noteSeqLen};
    const std::array<int64_t, 2> validShape{1, IodPretrainedTypes::nNotesMax};
    const std::array<int64_t, 3> comboShape{1, IodPretrainedTypes::nNotesMax,
                                            IodPretrainedTypes::comboLen};

    std::array<int64_t, IodPretrainedTypes::nNotesMax * IodPretrainedTypes::comboLen> comboFlat{};
    for (int32_t n = 0; n < IodPretrainedTypes::nNotesMax; ++n) {
        for (int32_t s = 0; s < IodPretrainedTypes::comboLen; ++s) {
            comboFlat[static_cast<size_t>(n * IodPretrainedTypes::comboLen + s)] =
                window.comboIn[static_cast<size_t>(n)][static_cast<size_t>(s)];
        }
    }

    try {
        Ort::Value tIds = Ort::Value::CreateTensor<int64_t>(
            memoryInfo, window.inputIds.data(), window.inputIds.size(), idsShape.data(),
            idsShape.size());
        Ort::Value tAttn = Ort::Value::CreateTensor<int64_t>(
            memoryInfo, window.attentionMask.data(), window.attentionMask.size(), idsShape.data(),
            idsShape.size());
        Ort::Value tCombo = Ort::Value::CreateTensor<int64_t>(
            memoryInfo, comboFlat.data(), comboFlat.size(), comboShape.data(), comboShape.size());
        Ort::Value tValid = Ort::Value::CreateTensor<int64_t>(
            memoryInfo, window.noteValid.data(), window.noteValid.size(), validShape.data(),
            validShape.size());

        std::vector<Ort::Value> inputs;
        inputs.reserve(4);
        std::vector<const char *> inputNamePtrs;
        inputNamePtrs.reserve(4);
        for (const auto &name: allocatedInputNames) {
            inputNamePtrs.push_back(name.c_str());
            if (name == "input_ids")
                inputs.push_back(std::move(tIds));
            else if (name == "attention_mask")
                inputs.push_back(std::move(tAttn));
            else if (name == "combo_in")
                inputs.push_back(std::move(tCombo));
            else if (name == "note_valid")
                inputs.push_back(std::move(tValid));
            else {
                reportSamplingError("IodPretrained: unexpected input " + juce::String(name));
                return {};
            }
        }

        std::vector<const char *> outputNamePtrs;
        for (const auto &name: allocatedOutputNames)
            outputNamePtrs.push_back(name.c_str());

        auto outputs = session->Run(Ort::RunOptions{nullptr}, inputNamePtrs.data(), inputs.data(),
                                    inputs.size(), outputNamePtrs.data(), outputNamePtrs.size());
        if (outputs.empty())
            return {};

        const auto &info = outputs[static_cast<size_t>(outLogits)].GetTensorTypeAndShapeInfo();
        const auto count = info.GetElementCount();
        const float *data = outputs[static_cast<size_t>(outLogits)].GetTensorData<float>();
        return std::vector<float>(data, data + count);
    } catch (const Ort::Exception &ex) {
        reportSamplingError(juce::String("IodPretrained ORT: ") + ex.what());
        return {};
    }
}

auto IodPretrained::sampleComboForNote(IodPretrainedTypes::EncoderWindow &window,
                                       int32_t noteIdx,
                                       const std::vector<int32_t> &instruments)
    -> std::array<int32_t, IodPretrainedTypes::comboLen> {
    using namespace IodPretrainedTypes;

    std::array<int32_t, comboLen> combo{};
    combo.fill(endNote);

    if (noteIdx < 0 || noteIdx >= window.nNotes)
        return combo;

    const auto forbid = forbidBitsForPitch(instruments, window.basePitch[static_cast<size_t>(noteIdx)]);
    if (forbid.fullyBannedFamilyBits == (1 << numFamilies) - 1) {
        return combo;
    }

    window.comboIn[static_cast<size_t>(noteIdx)].fill(bos);

    {
        auto logits = runLogits(window);
        if (logits.size() < logitsIndex(noteIdx, 0) + static_cast<size_t>(comboVocabSize))
            return combo;
        const float *row = logits.data() + logitsIndex(noteIdx, 0);
        const int32_t groupsTok = greedySampleRange(row, groupsOffset, groupsHi,
                                                    /*forbidBits*/ 0, forbid.fullyBannedFamilyBits,
                                                    /*excludeZero*/ false);
        combo[0] = groupsTok;
        window.comboIn[static_cast<size_t>(noteIdx)][1] = groupsTok;
    }

    const int32_t groupsMask = maskFromGroupsToken(combo[0]);
    int32_t step = 1;
    int32_t comboWrite = 1;

    for (int32_t familyIdx = 0; familyIdx < numFamilies; ++familyIdx) {
        if ((groupsMask & (1 << familyIdx)) == 0)
            continue;
        if (step >= comboLen)
            break;

        auto logits = runLogits(window);
        if (logits.size() < logitsIndex(noteIdx, step) + static_cast<size_t>(comboVocabSize))
            break;

        const auto &family = familySpecs[static_cast<size_t>(familyIdx)];
        const float *row = logits.data() + logitsIndex(noteIdx, step);
        const int32_t famTok =
            greedySampleRange(row, family.tokenOffset, family.tokenOffset + family.tokenSize,
                              forbid.familyForbid[static_cast<size_t>(familyIdx)],
                              /*forbidGroupsFamilies*/ 0,
                              /*excludeZero*/ true);
        combo[static_cast<size_t>(comboWrite++)] = famTok;
        if (step + 1 < comboLen)
            window.comboIn[static_cast<size_t>(noteIdx)][static_cast<size_t>(step + 1)] = famTok;
        ++step;
    }

    while (comboWrite < comboLen)
        combo[static_cast<size_t>(comboWrite++)] = endNote;

    for (int32_t s = 0; s < comboLen - 1; ++s)
        window.comboIn[static_cast<size_t>(noteIdx)][static_cast<size_t>(s + 1)] =
            combo[static_cast<size_t>(s)];

    return combo;
}

auto IodPretrained::sampleWindowCombos(IodPretrainedTypes::EncoderWindow &window,
                                       const std::vector<int32_t> &instruments)
    -> std::vector<std::array<int32_t, IodPretrainedTypes::comboLen>> {
    std::vector<std::array<int32_t, IodPretrainedTypes::comboLen>> out;
    out.reserve(static_cast<size_t>(window.nNotes));
    for (int32_t i = 0; i < window.nNotes; ++i)
        out.push_back(sampleComboForNote(window, i, instruments));
    return out;
}

auto IodPretrained::getOutput(const std::vector<Token> &incomingTokens,
                              const std::vector<int32_t> &instruments,
                              const ConditioningSignal &conditioningSignal,
                              OrchestrationBalanceTracker *balance,
                              const OrchestrationBalanceTracker::BiasView *bias)
    -> std::vector<OrchestrationNote> {
    juce::ignoreUnused(conditioningSignal, bias);

    if (incomingTokens.empty() || session == nullptr)
        return {};

    if (getOutputTimings != nullptr)
        ++getOutputTimings->getOutputCalls;

    std::vector<OrchestrationNote> result;

    const auto starts = IodPretrainedTypes::encoderWindowStarts(incomingTokens.size());
    for (const size_t start: starts) {
        IodPretrainedTypes::EncoderWindow window;
        {
            const ScopedMs prepMs(getOutputTimings != nullptr ? &getOutputTimings->prep : nullptr);
            window = IodPretrainedTypes::packEncoderWindow(incomingTokens, start);
            if (window.nNotes <= 0)
                continue;

            int32_t minOnset = window.onsetCs[0];
            for (int32_t i = 1; i < window.nNotes; ++i)
                minOnset = std::min(minOnset, window.onsetCs[static_cast<size_t>(i)]);
            for (int32_t i = 0; i < window.nNotes; ++i) {
                const size_t base = static_cast<size_t>(i) * 3;
                window.inputIds[base] =
                    static_cast<int64_t>(window.onsetCs[static_cast<size_t>(i)] - minOnset);
            }

            if (getOutputTimings != nullptr) {
                getOutputTimings->contextTokens = juce::jmax(
                    getOutputTimings->contextTokens,
                    window.nNotes * IodPretrainedTypes::noteTokensPerNote);
            }
        }

        std::vector<std::array<int32_t, IodPretrainedTypes::comboLen>> combos;
        {
            const ScopedMs sampleMs(getOutputTimings != nullptr ? &getOutputTimings->sample
                                                                : nullptr);
            combos = sampleWindowCombos(window, instruments);
        }

        {
            const ScopedMs decodeMs(getOutputTimings != nullptr ? &getOutputTimings->decode
                                                                : nullptr);
            for (size_t i = 0; i < combos.size(); ++i) {
                auto notes = IodPretrainedTypes::decodeComboToNotes(
                    window.onsetCs[i], window.durationToken[i], window.basePitch[i],
                    window.velocity[i], combos[i]);
                if (balance != nullptr && ! notes.empty()) {
                    std::vector<int32_t> ids;
                    ids.reserve(notes.size());
                    for (const auto &n: notes)
                        ids.push_back(n.localInstrumentId);
                    balance->recordNote(ids);
                }
                result.insert(result.end(), notes.begin(), notes.end());
            }
        }
    }

    {
        history.insert(history.end(), result.begin(), result.end());
        NoteWindow::trimToLastNotes(history);
    }

    return result;
}
