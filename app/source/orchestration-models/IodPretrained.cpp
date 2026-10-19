#include "VirtualOrch/orchestration-models/IodPretrained.h"

#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/NoteWindow.h"
#include "VirtualOrch/OrtEnv.h"
#include "VirtualOrch/OrchLoopProfile.h"
#include "VirtualOrch/orchestration-models/VocsepEnsemble.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
/** Softmax temperature for all IOD combo AR steps (GROUPS + family). */
constexpr float kComboSampleTemperature = 1.0f;

auto findNameIndex(const std::vector<std::string> &names, const char *want) -> int {
    for (size_t i = 0; i < names.size(); ++i) {
        if (names[i] == want)
            return static_cast<int>(i);
    }
    return -1;
}

auto sampleRange(const float *vocabLogits,
                 int32_t lo,
                 int32_t hi,
                 int32_t forbidBits,
                 int32_t forbidGroupsFamilies,
                 bool excludeZero,
                 const IodPretrainedTypes::FamilySpec *family = nullptr,
                 int32_t claimedDeltas = 0) -> int32_t {
    const int32_t start = excludeZero ? lo + 1 : lo;
    std::vector<float> local(static_cast<size_t>(hi - start), kNegInf);
    bool any = false;
    for (int32_t id = start; id < hi; ++id) {
        const int32_t value = id - lo;
        if (family != nullptr) {
            if (! IodPretrainedTypes::isLegalFamilyMask(value, forbidBits, family, claimedDeltas))
                continue;
        } else {
            if (forbidBits != 0 && (value & forbidBits) != 0)
                continue;
            if (forbidGroupsFamilies != 0) {
                const int32_t groupsMask = value + 1; // GROUPS: offset k → mask k+1
                if ((groupsMask & forbidGroupsFamilies) != 0)
                    continue;
            }
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
        if (family != nullptr) {
            for (int32_t value = (excludeZero ? 1 : 0); value < hi - lo; ++value) {
                if (IodPretrainedTypes::isLegalFamilyMask(value, forbidBits, family, claimedDeltas))
                    return lo + value;
            }
        }
        return start < hi ? start : lo;
    }
    const int32_t localIdx = sampleTopP(local, /*p*/ 1.0f, kComboSampleTemperature);
    return start + localIdx;
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
                                       const std::vector<int32_t> &instruments,
                                       bool applyGroupsBias)
    -> std::array<int32_t, IodPretrainedTypes::comboLen> {
    using namespace IodPretrainedTypes;

    std::array<int32_t, comboLen> combo{};
    combo.fill(endNote);

    if (noteIdx < 0 || noteIdx >= window.nNotes)
        return combo;

    const auto role = outerVoiceDeltaAllows(window.onsetCs.data(), window.durationToken.data(),
                                            window.basePitch.data(), window.nNotes, noteIdx);
    auto forbid = forbidBitsForPitch(instruments, window.basePitch[static_cast<size_t>(noteIdx)]);
    applyOuterVoiceDeltaForbid(forbid, role.allowMinus1, role.allowPlus1);
    if (forbid.fullyBannedFamilyBits == (1 << numFamilies) - 1) {
        return combo;
    }

    window.comboIn[static_cast<size_t>(noteIdx)].fill(bos);

    {
        auto logits = runLogits(window);
        if (logits.size() < logitsIndex(noteIdx, 0) + static_cast<size_t>(comboVocabSize))
            return combo;
        float *row = logits.data() + logitsIndex(noteIdx, 0);
        if (applyGroupsBias && ! recentGroupsMasks.empty()) {
            const std::vector<int32_t> masks(recentGroupsMasks.begin(), recentGroupsMasks.end());
            applyGroupsTokenLogitsBias(row, static_cast<size_t>(comboVocabSize), masks.data(),
                                       masks.size(), groupsBiasTau.load());
        }
        const int32_t groupsTok = sampleRange(row, groupsOffset, groupsHi,
                                              /*forbidBits*/ 0, forbid.fullyBannedFamilyBits,
                                              /*excludeZero*/ false);
        combo[0] = groupsTok;
        window.comboIn[static_cast<size_t>(noteIdx)][1] = groupsTok;
        recordGroupsMask(maskFromGroupsToken(groupsTok));
    }

    const int32_t groupsMask = maskFromGroupsToken(combo[0]);
    int32_t step = 1;
    int32_t comboWrite = 1;
    int32_t claimedDeltas = 0; // bit0/1/2 ↔ Δ −1/0/+1 across families

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
            sampleRange(row, family.tokenOffset, family.tokenOffset + family.tokenSize,
                        forbid.familyForbid[static_cast<size_t>(familyIdx)],
                        /*forbidGroupsFamilies*/ 0,
                        /*excludeZero*/ true, &family, claimedDeltas);
        combo[static_cast<size_t>(comboWrite++)] = famTok;
        claimedDeltas |=
            deltasUsedByFamilyMask(family, maskFromFamilyToken(family, famTok));
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

auto IodPretrained::recordGroupsMask(int32_t groupsMask) -> void {
    using namespace IodPretrainedTypes;
    if (groupsMask < 1 || groupsMask > groupsSize)
        return;
    recentGroupsMasks.push_back(groupsMask);
    while (static_cast<int32_t>(recentGroupsMasks.size()) > groupsBiasWindowNotes)
        recentGroupsMasks.pop_front();
}

auto IodPretrained::comboFromData(const std::vector<int32_t> &data)
    -> std::array<int32_t, IodPretrainedTypes::comboLen> {
    std::array<int32_t, IodPretrainedTypes::comboLen> combo;
    combo.fill(IodPretrainedTypes::endNote);
    for (size_t i = 0; i < combo.size() && i < data.size(); ++i)
        combo[i] = data[i];
    return combo;
}

auto IodPretrained::dataFromCombo(const std::array<int32_t, IodPretrainedTypes::comboLen> &combo)
    -> std::vector<int32_t> {
    return {combo.begin(), combo.end()};
}

auto IodPretrained::writeComboInFromStored(
    IodPretrainedTypes::EncoderWindow &window,
    int32_t noteIdx,
    const std::array<int32_t, IodPretrainedTypes::comboLen> &combo) -> void {
    using namespace IodPretrainedTypes;
    window.comboIn[static_cast<size_t>(noteIdx)].fill(bos);
    for (int32_t s = 0; s < comboLen - 1; ++s)
        window.comboIn[static_cast<size_t>(noteIdx)][static_cast<size_t>(s + 1)] =
            combo[static_cast<size_t>(s)];
}

auto IodPretrained::ensembleComboB(size_t listIdx, size_t windowBase)
    -> std::array<int32_t, IodPretrainedTypes::comboLen> {
    using namespace IodPretrainedTypes;

    const auto &cur = inputNotes[listIdx];
    const auto comboA = comboFromData(cur.dataA);

    // Shared, model-neutral part: smooth the (instrument x octave-delta) pairs over the voice.
    std::vector<int32_t> bPairs = ensembleKeysB(listIdx);

    // IOD-only part: outer-voice delta roles among temporally overlapping notes in the
    // encoder window, then at most one instrument per octave delta.
    const int32_t onset = cur.token.time;
    const int32_t dur = std::max(1, cur.token.getRealDuration());
    const int32_t pitch = cur.token.getPitch();
    OuterVoiceDeltaAllows role;
    for (size_t i = windowBase; i < inputNotes.size(); ++i) {
        if (i == listIdx)
            continue;
        const auto &other = inputNotes[i].token;
        const int32_t otherDur = std::max(1, other.getRealDuration());
        if (! notesOverlapCs(onset, dur, other.time, otherDur))
            continue;
        const int32_t otherPitch = other.getPitch();
        if (otherPitch > pitch)
            role.allowPlus1 = false;
        if (otherPitch < pitch)
            role.allowMinus1 = false;
    }

    const auto filtered = filterPairsUniqueOctave(
        filterPairsByDeltaAllows(bPairs, role.allowMinus1, role.allowPlus1));
    if (filtered.empty())
        return comboA; // an empty B would silence the note: fall back to the raw choice A
    return comboFromPairs(filtered);
}

auto IodPretrained::sampleWindowCombos(IodPretrainedTypes::EncoderWindow &window,
                                       const std::vector<int32_t> &instruments,
                                       bool applyGroupsBias)
    -> std::vector<std::array<int32_t, IodPretrainedTypes::comboLen>> {
    std::vector<std::array<int32_t, IodPretrainedTypes::comboLen>> out;
    out.reserve(static_cast<size_t>(window.nNotes));
    for (int32_t i = 0; i < window.nNotes; ++i)
        out.push_back(sampleComboForNote(window, i, instruments, applyGroupsBias));
    return out;
}

auto IodPretrained::getOutput(const std::vector<Token> &incomingTokens,
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

    const bool applyGroupsBias = bias == nullptr || bias->apply;
    const bool useEnsemble = getVocsepEnsembleEnabled();

    using namespace IodPretrainedTypes;

    bool cleared = false;
    const size_t listNewBegin = recordIncoming(incomingTokens, historyGapClearCs, &cleared);
    if (cleared)
        recentGroupsMasks.clear();
    if (inputNotes.size() <= listNewBegin)
        return {};

    // The encoder looks at the last nNotesMax notes of the shared list.
    const size_t windowBase =
        inputNotes.size() > static_cast<size_t>(nNotesMax)
            ? inputNotes.size() - static_cast<size_t>(nNotesMax)
            : 0;
    const size_t newBegin = listNewBegin >= windowBase ? listNewBegin - windowBase : 0;

    std::vector<Token> streamTokens;
    streamTokens.reserve(inputNotes.size() - windowBase);
    for (size_t i = windowBase; i < inputNotes.size(); ++i)
        streamTokens.push_back(inputNotes[i].token);

    std::vector<OrchestrationNote> result;

    // Single window: last <=128 notes (encoderWindowStarts is identity when N<=128).
    const auto starts = encoderWindowStarts(streamTokens.size());
    for (const size_t start: starts) {
        EncoderWindow window;
        {
            const ScopedMs prepMs(getOutputTimings != nullptr ? &getOutputTimings->prep : nullptr);
            window = packEncoderWindow(streamTokens, start);
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

            // Teacher-force already-decoded combos (effective B) into combo_in.
            for (int32_t i = 0; i < window.nNotes; ++i) {
                const size_t global = start + static_cast<size_t>(i);
                if (windowBase + global >= inputNotes.size())
                    break;
                const auto &rec = inputNotes[windowBase + global];
                if (rec.hasResult)
                    writeComboInFromStored(window, i, comboFromData(rec.dataB));
            }

            if (getOutputTimings != nullptr) {
                getOutputTimings->contextTokens = juce::jmax(
                    getOutputTimings->contextTokens,
                    window.nNotes * noteTokensPerNote);
            }
        }

        {
            const ScopedMs sampleMs(getOutputTimings != nullptr ? &getOutputTimings->sample
                                                                : nullptr);
            for (int32_t i = 0; i < window.nNotes; ++i) {
                const size_t global = start + static_cast<size_t>(i);
                const size_t listIdx = windowBase + global;
                if (listIdx >= inputNotes.size())
                    break;
                if (inputNotes[listIdx].hasResult)
                    continue;
                const auto comboA = sampleComboForNote(window, i, instruments, applyGroupsBias);
                {
                    auto &rec = inputNotes[listIdx];
                    rec.dataA = dataFromCombo(comboA);
                    rec.keysA = VocsepEnsemble::sortedUnique(pairsFromCombo(comboA));
                }
                auto comboB = comboA;
                if (useEnsemble) {
                    comboB = ensembleComboB(listIdx, windowBase);
                    // Feed B (not A) into combo_in for subsequent notes in this window.
                    writeComboInFromStored(window, i, comboB);
                }
                auto &rec = inputNotes[listIdx];
                rec.dataB = dataFromCombo(comboB);
                rec.keysB = VocsepEnsemble::sortedUnique(pairsFromCombo(comboB));
                rec.hasResult = true;
            }
        }

        {
            const ScopedMs decodeMs(getOutputTimings != nullptr ? &getOutputTimings->decode
                                                                : nullptr);
            for (int32_t i = 0; i < window.nNotes; ++i) {
                const size_t global = start + static_cast<size_t>(i);
                const size_t listIdx = windowBase + global;
                if (global < newBegin || listIdx >= inputNotes.size())
                    continue;
                if (! inputNotes[listIdx].hasResult)
                    continue;
                const int32_t basePitch = window.basePitch[static_cast<size_t>(i)];
                const auto role = outerVoiceDeltaAllows(window.onsetCs.data(),
                                                        window.durationToken.data(),
                                                        window.basePitch.data(), window.nNotes, i);
                auto safePairs = filterPairsUniqueOctave(filterPairsByDeltaAllows(
                    pairsFromCombo(comboFromData(inputNotes[listIdx].dataB)), role.allowMinus1,
                    role.allowPlus1));
                const auto safeCombo = comboFromPairs(safePairs);
                auto notes = decodeComboToNotes(
                    window.onsetCs[static_cast<size_t>(i)],
                    window.durationToken[static_cast<size_t>(i)],
                    basePitch,
                    window.velocity[static_cast<size_t>(i)],
                    safeCombo);
                if (balance != nullptr && ! notes.empty()) {
                    std::vector<int32_t> ids;
                    ids.reserve(notes.size());
                    for (const auto &n: notes)
                        ids.push_back(n.localInstrumentId);
                    balance->recordNote(ids);
                }
                if (octaveDeltaSink != nullptr) {
                    for (const auto &n: notes) {
                        const int32_t delta = (n.token.getPitch() - basePitch) / 12;
                        octaveDeltaSink->record(delta);
                    }
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
