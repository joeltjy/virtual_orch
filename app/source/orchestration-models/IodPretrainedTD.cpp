#include "VirtualOrch/orchestration-models/IodPretrainedTD.h"

#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/NoteWindow.h"
#include "VirtualOrch/OrtEnv.h"
#include "VirtualOrch/OrchLoopProfile.h"
#include "VirtualOrch/orchestration-models/VocsepEnsemble.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

using namespace IodPretrainedTypes;

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

/** Sample one of `candidates` (vocab ids) from the logit row: temperature 1, no top-p. */
auto pickAmong(const std::vector<float> &row, const std::vector<int32_t> &candidates) -> int32_t {
    std::vector<float> local(candidates.size(), kNegInf);
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto id = static_cast<size_t>(candidates[i]);
        if (id < row.size() && std::isfinite(row[id]))
            local[i] = row[id];
    }
    return candidates[static_cast<size_t>(sampleTopP(local, 1.0f, 1.0f))];
}

auto allEndCombo() -> std::array<int32_t, comboLen> {
    std::array<int32_t, comboLen> combo{};
    combo.fill(endNote);
    return combo;
}

auto roleAmongOverlapping(const std::vector<OrchestrationModel::InputNote> &notes,
                          size_t listIdx,
                          size_t windowBegin) -> OuterVoiceDeltaAllows {
    const auto &cur = notes[listIdx].token;
    const int32_t onset = cur.time;
    const int32_t dur = std::max(1, cur.getRealDuration());
    const int32_t pitch = cur.getPitch();
    OuterVoiceDeltaAllows role;
    for (size_t i = windowBegin; i < notes.size(); ++i) {
        if (i == listIdx)
            continue;
        const auto &other = notes[i].token;
        if (! notesOverlapCs(onset, dur, other.time, std::max(1, other.getRealDuration())))
            continue;
        if (other.getPitch() > pitch)
            role.allowPlus1 = false;
        if (other.getPitch() < pitch)
            role.allowMinus1 = false;
    }
    return role;
}

} // namespace

IodPretrainedTD::IodPretrainedTD() {
    temporalDensitySource = [] { return TemporalDensity::get(); };
}

auto IodPretrainedTD::init(const char *modelPath) -> void {
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
            session.reset();
            reportSamplingError(juce::String("IodPretrainedTD init failed: ") + ex.what());
            return;
        }
    }

    const Ort::AllocatorWithDefaultOptions allocator;
    inputName = session->GetInputNameAllocated(0, allocator).get();
    outputName = session->GetOutputNameAllocated(0, allocator).get();

    modelVocabSize = 0;
    const auto shape = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (! shape.empty() && shape.back() > 0)
        modelVocabSize = static_cast<size_t>(shape.back());
    if (modelVocabSize != 0 && modelVocabSize != static_cast<size_t>(kVocabSize)) {
        const auto detail = "IodPretrainedTD vocab mismatch: model "
                            + juce::String(static_cast<int>(modelVocabSize)) + " vs expected "
                            + juce::String(kVocabSize);
        DBG(detail);
        reportSamplingError(detail);
    }
}

// ---- pure helpers ----

auto IodPretrainedTD::quadIds(const Token &token, int32_t rebaseCs) -> std::array<int64_t, 4> {
    const int32_t onset = std::clamp(token.time - rebaseCs, 0, kMaxOnsetCs);
    const int32_t durCs = std::clamp(((token.getRealDuration() + 2) / 5) * 5, 5, 995);
    const int32_t pitch = std::clamp(token.getPitch(), 0, 127);
    const int32_t vel = token.velocity > 0 ? token.velocity : 64;
    const int32_t velQ = std::clamp(((vel + 4) / 8) * 8, 0, 120);
    return {onset, kDurOffset + durCs, kNoteOffset + pitch, kVelocityOffset + velQ};
}

auto IodPretrainedTD::slotsFromCompact(const std::array<int32_t, comboLen> &compact)
    -> std::array<int64_t, comboLen> {
    std::array<int64_t, comboLen> slots{};
    slots.fill(kEndNote);
    slots[0] = compact[0] + kComboShift;
    const int32_t mask = maskFromGroupsToken(compact[0]);
    size_t k = 1;
    for (int32_t f = 0; f < numFamilies; ++f) {
        if ((mask & (1 << f)) == 0)
            continue;
        slots[static_cast<size_t>(1 + f)] = compact[k++] + kComboShift;
    }
    return slots;
}

auto IodPretrainedTD::compactFromSlots(const std::array<int64_t, comboLen> &slots)
    -> std::array<int32_t, comboLen> {
    auto compact = allEndCombo();
    compact[0] = static_cast<int32_t>(slots[0] - kComboShift);
    const int32_t mask = maskFromGroupsToken(compact[0]);
    size_t k = 1;
    for (int32_t f = 0; f < numFamilies; ++f) {
        if ((mask & (1 << f)) == 0)
            continue;
        compact[k++] = static_cast<int32_t>(slots[static_cast<size_t>(1 + f)] - kComboShift);
    }
    return compact;
}

auto IodPretrainedTD::comboFromData(const std::vector<int32_t> &data)
    -> std::array<int32_t, comboLen> {
    auto combo = allEndCombo();
    for (size_t i = 0; i < combo.size() && i < data.size(); ++i)
        combo[i] = data[i];
    return combo;
}

auto IodPretrainedTD::dataFromCombo(const std::array<int32_t, comboLen> &combo)
    -> std::vector<int32_t> {
    return {combo.begin(), combo.end()};
}

auto IodPretrainedTD::buildContext(size_t begin, size_t target, int32_t rebaseCs) const
    -> std::vector<int64_t> {
    std::vector<int64_t> ids;
    ids.reserve(static_cast<size_t>(kMaxContext));
    int32_t nextBoundary = 0; // relative to rebaseCs
    for (size_t j = begin; j <= target && j < inputNotes.size(); ++j) {
        const auto &rec = inputNotes[j];
        const auto quad = quadIds(rec.token, rebaseCs);
        ids.insert(ids.end(), quad.begin(), quad.end());
        if (j == target)
            break;
        const auto slots = slotsFromCompact(comboFromData(rec.dataB));
        ids.insert(ids.end(), slots.begin(), slots.end());
        const int32_t rel = rec.token.time - rebaseCs;
        while (nextBoundary <= rel) {
            const auto it = boundaryTd.find(rebaseCs + nextBoundary);
            ids.push_back(it != boundaryTd.end() ? it->second : TdTokens::kIodUnknown);
            nextBoundary += TdTokens::kBoundaryCs;
        }
    }
    return ids;
}

// ---- model calls ----

auto IodPretrainedTD::lastRowLogits(const std::vector<int64_t> &ids) -> std::vector<float> {
    if (session == nullptr || ids.empty())
        return {};
    try {
        std::vector<int64_t> data = ids;
        const std::array<int64_t, 2> shape{1, static_cast<int64_t>(data.size())};
        Ort::Value input = Ort::Value::CreateTensor<int64_t>(memoryInfo, data.data(), data.size(),
                                                             shape.data(), shape.size());
        const char *inNames[] = {inputName.c_str()};
        const char *outNames[] = {outputName.c_str()};
        auto outputs = session->Run(Ort::RunOptions{nullptr}, inNames, &input, 1, outNames, 1);
        if (outputs.empty())
            return {};
        const auto info = outputs[0].GetTensorTypeAndShapeInfo();
        const auto outShape = info.GetShape();
        if (outShape.size() != 3 || outShape[2] <= 0)
            return {};
        const auto vocab = static_cast<size_t>(outShape[2]);
        const auto positions = static_cast<size_t>(outShape[1]);
        if (positions == 0)
            return {};
        const float *row = outputs[0].GetTensorData<float>() + (positions - 1) * vocab;
        return {row, row + vocab};
    } catch (const Ort::Exception &ex) {
        reportSamplingError(juce::String("IodPretrainedTD ORT: ") + ex.what());
        return {};
    }
}

auto IodPretrainedTD::assignBoundaries(int32_t onsetCs) -> void {
    if (! boundariesStarted) {
        boundariesStarted = true;
        // the first note of the stream carries the boundary at or below its onset
        lastAssignedBoundary = (std::max(onsetCs, 0) / TdTokens::kBoundaryCs) * TdTokens::kBoundaryCs
                               - TdTokens::kBoundaryCs;
    }
    const int32_t id = TdTokens::iodIdForLaunchpad(temporalDensitySource());
    for (int32_t b = lastAssignedBoundary + TdTokens::kBoundaryCs; b <= onsetCs;
         b += TdTokens::kBoundaryCs) {
        boundaryTd.emplace(b, id);
        lastAssignedBoundary = b;
    }
}

auto IodPretrainedTD::sampleCombo(size_t listIdx,
                                  size_t windowBegin,
                                  int32_t rebaseCs,
                                  const std::vector<int32_t> &instruments)
    -> std::array<int32_t, comboLen> {
    const auto &cur = inputNotes[listIdx];
    const auto role = roleAmongOverlapping(inputNotes, listIdx, windowBegin);
    auto forbid = forbidBitsForPitch(instruments, cur.token.getPitch());
    applyOuterVoiceDeltaForbid(forbid, role.allowMinus1, role.allowPlus1);
    if (forbid.fullyBannedFamilyBits == (1 << numFamilies) - 1)
        return allEndCombo();

    auto ids = buildContext(windowBegin, listIdx, rebaseCs);
    std::array<int64_t, comboLen> slots{};
    slots.fill(kEndNote);

    // GROUPS
    {
        const auto row = lastRowLogits(ids);
        if (row.empty())
            return allEndCombo();
        std::vector<int32_t> candidates;
        for (int32_t mask = 1; mask <= groupsSize; ++mask)
            if (isLegalGroupsMask(mask, forbid.fullyBannedFamilyBits))
                candidates.push_back(mask + kComboShift);
        if (candidates.empty())
            return allEndCombo();
        slots[0] = pickAmong(row, candidates);
        ids.push_back(slots[0]);
    }

    const int32_t groupsMask = maskFromGroupsToken(static_cast<int32_t>(slots[0] - kComboShift));
    int32_t claimedDeltas = 0;
    for (int32_t f = 0; f < numFamilies; ++f) {
        const auto slot = static_cast<size_t>(1 + f);
        if ((groupsMask & (1 << f)) == 0) {
            ids.push_back(kEndNote);
            continue;
        }
        const auto &family = familySpecs[static_cast<size_t>(f)];
        // Legal masks that avoid the octaves already taken by earlier families; when none is
        // left, relax that (decode and the unique-octave filter drop the clashing pairs).
        std::vector<int32_t> candidates;
        for (const int32_t claimed: {claimedDeltas, 0}) {
            for (int32_t mask = 1; mask < family.tokenSize; ++mask)
                if (isLegalFamilyMask(mask, forbid.familyForbid[static_cast<size_t>(f)], &family,
                                      claimed))
                    candidates.push_back(family.tokenOffset + mask + kComboShift);
            if (! candidates.empty())
                break;
        }
        const auto row = lastRowLogits(ids);
        if (candidates.empty() || row.empty())
            return allEndCombo();
        slots[slot] = pickAmong(row, candidates);
        ids.push_back(slots[slot]);
        claimedDeltas |= deltasUsedByFamilyMask(
            family, maskFromFamilyToken(family, static_cast<int32_t>(slots[slot] - kComboShift)));
    }
    return compactFromSlots(slots);
}

auto IodPretrainedTD::ensembleComboB(size_t listIdx, size_t windowBegin, const std::vector<int32_t> &instruments)
    -> std::array<int32_t, comboLen> {
    const auto comboA = comboFromData(inputNotes[listIdx].dataA);
    const auto bPairs = ensembleKeysB(listIdx);
    const auto role = roleAmongOverlapping(inputNotes, listIdx, windowBegin);
    const auto filtered = filterPairsUniqueOctave(
        filterPairsByDeltaAllows(filterPairsPlayable(bPairs, inputNotes[listIdx].token.getPitch(), instruments),
                                 role.allowMinus1, role.allowPlus1));
    if (filtered.empty())
        return comboA;
    return comboFromPairs(filtered);
}

auto IodPretrainedTD::getOutput(const std::vector<Token> &incomingTokens,
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

    const bool useEnsemble = getVocsepEnsembleEnabled();

    bool cleared = false;
    const size_t newBegin = recordIncoming(incomingTokens, historyGapClearCs, &cleared);
    if (cleared) {
        boundaryTd.clear();
        boundariesStarted = false;
    }
    if (inputNotes.size() <= newBegin)
        return {};

    std::vector<OrchestrationNote> result;
    for (size_t idx = newBegin; idx < inputNotes.size(); ++idx) {
        if (inputNotes[idx].hasResult)
            continue;

        // Window: the newest notes that fit the context and the 0..9999 cs onset range.
        size_t begin = idx > static_cast<size_t>(kMaxNotesInContext)
                           ? idx - static_cast<size_t>(kMaxNotesInContext)
                           : 0;
        int32_t rebase = 0;
        std::vector<int64_t> probe;
        for (;;) {
            int32_t minOnset = inputNotes[idx].token.time;
            int32_t maxOnset = minOnset;
            for (size_t j = begin; j <= idx; ++j) {
                minOnset = std::min(minOnset, inputNotes[j].token.time);
                maxOnset = std::max(maxOnset, inputNotes[j].token.time);
            }
            rebase = (std::max(minOnset, 0) / TdTokens::kBoundaryCs) * TdTokens::kBoundaryCs;
            if (begin < idx && maxOnset - rebase > kMaxOnsetCs) {
                ++begin;
                continue;
            }
            probe = buildContext(begin, idx, rebase);
            if (begin < idx && static_cast<int32_t>(probe.size()) + comboLen + 1 > kMaxContext) {
                ++begin;
                continue;
            }
            break;
        }

        assignBoundaries(inputNotes[idx].token.time);
        for (auto it = boundaryTd.begin(); it != boundaryTd.end() && it->first < rebase;)
            it = boundaryTd.erase(it);

        {
            const ScopedMs sampleMs(getOutputTimings != nullptr ? &getOutputTimings->sample
                                                                : nullptr);
            const auto comboA = sampleCombo(idx, begin, rebase, instruments);
            auto &rec = inputNotes[idx];
            rec.dataA = dataFromCombo(comboA);
            rec.keysA = VocsepEnsemble::sortedUnique(pairsFromCombo(comboA));
            auto comboB = comboA;
            if (useEnsemble)
                comboB = ensembleComboB(idx, begin, instruments);
            rec.dataB = dataFromCombo(comboB);
            rec.keysB = VocsepEnsemble::sortedUnique(pairsFromCombo(comboB));
            rec.hasResult = true;
            if (getOutputTimings != nullptr)
                getOutputTimings->contextTokens =
                    juce::jmax(getOutputTimings->contextTokens, static_cast<int>(probe.size()));
        }

        {
            const ScopedMs decodeMs(getOutputTimings != nullptr ? &getOutputTimings->decode
                                                                : nullptr);
            const auto &rec = inputNotes[idx];
            const auto role = roleAmongOverlapping(inputNotes, idx, begin);
            const auto safePairs = filterPairsUniqueOctave(filterPairsByDeltaAllows(
                pairsFromCombo(comboFromData(rec.dataB)), role.allowMinus1, role.allowPlus1));
            auto notes = decodeComboToNotes(
                rec.token.time, rec.token.duration, rec.token.getPitch(),
                rec.token.velocity > 0 ? rec.token.velocity : 64, comboFromPairs(safePairs));
            for (auto &n: notes)
                n.token.keyId = rec.token.keyId;
            if (balance != nullptr && ! notes.empty()) {
                std::vector<int32_t> noteInstruments;
                noteInstruments.reserve(notes.size());
                for (const auto &n: notes)
                    noteInstruments.push_back(n.localInstrumentId);
                balance->recordNote(noteInstruments);
            }
            if (octaveDeltaSink != nullptr) {
                for (const auto &n: notes)
                    octaveDeltaSink->record((n.token.getPitch() - rec.token.getPitch()) / 12);
            }
            result.insert(result.end(), notes.begin(), notes.end());
        }
    }

    history.insert(history.end(), result.begin(), result.end());
    NoteWindow::trimToLastNotes(history);
    return result;
}
