#include "VirtualOrch/orchestration-models/IodPretrained.h"

#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/NoteWindow.h"
#include "VirtualOrch/OrtEnv.h"
#include "VirtualOrch/OrchLoopProfile.h"
#include "VirtualOrch/OpLog.h"
#include "VirtualOrch/OpLogDraw.h"
#include "VirtualOrch/orchestration-models/VocsepEnsemble.h"
#include "VirtualOrch/OrchestraTdMeter.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_map>

#if defined(__APPLE__) && defined(__aarch64__) && ORT_API_VERSION >= 20
#include <sys/stat.h>
#endif

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
/** Softmax temperature for all IOD combo AR steps (GROUPS + family). */
constexpr float kComboSampleTemperature = 1.0f;

#if defined(__APPLE__) && defined(__aarch64__) && ORT_API_VERSION >= 20
auto coreMlCacheDirectory(const char *path) -> juce::File {
    struct stat model {};
    if (stat(path, &model) != 0)
        return {};

    // A replaced or edited ONNX file needs a fresh cache even when its path is unchanged.
    const auto identity = juce::String(static_cast<juce::int64>(model.st_dev)) + "-"
                          + juce::String(static_cast<juce::int64>(model.st_ino)) + "-"
                          + juce::String(static_cast<juce::int64>(model.st_size)) + "-"
                          + juce::String(static_cast<juce::int64>(model.st_ctimespec.tv_sec)) + "-"
                          + juce::String(static_cast<juce::int64>(model.st_ctimespec.tv_nsec));
    return juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
        .getChildFile("Caches/virtual-orch/CoreML")
        .getChildFile("mlprogram-gpu-v1")
        .getChildFile(Ort::GetVersionString())
        .getChildFile(identity);
}
#endif

auto createOrchestrationSession(const char *path) -> std::unique_ptr<Ort::Session> {
    auto &env = sharedOrtEnv();
#if defined(__APPLE__) && defined(__aarch64__) && ORT_API_VERSION >= 20
    const auto *backend = std::getenv("VIRTUAL_ORCH_ORT_BACKEND");
    if (backend == nullptr || std::strcmp(backend, "cpu") != 0) {
        try {
            Ort::SessionOptions options;
            // Both the full model and the split pair receive one encoder window per call.
            Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverrideByName(options, "batch", 1));
            std::unordered_map<std::string, std::string> providerOptions{
                {"ModelFormat", "MLProgram"},
                {"MLComputeUnits", "CPUAndGPU"},
                {"RequireStaticInputShapes", "1"},
            };
            const auto cache = coreMlCacheDirectory(path);
            if (cache != juce::File{} && cache.createDirectory().wasOk())
                providerOptions.emplace("ModelCacheDirectory", cache.getFullPathName().toStdString());
            options.AppendExecutionProvider("CoreML", providerOptions);
            auto session = std::make_unique<Ort::Session>(env, path, options);
            std::fprintf(stderr, "[ort] %s: Core ML enabled with CPU fallback\n", path);
            return session;
        } catch (const Ort::Exception &ex) {
            std::fprintf(stderr, "[ort] %s: Core ML init failed (%s), using CPU\n", path, ex.what());
        }
    }
#endif
    return std::make_unique<Ort::Session>(env, path, Ort::SessionOptions{});
}

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
                 int32_t claimedDeltas = 0,
                 std::optional<int32_t> forced = std::nullopt,
                 std::vector<OpLogDraw::Draw> *draws = nullptr,
                 const char *fieldName = "",
                 const float *rawForLog = nullptr) -> int32_t {
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
    // Operation log: tokens are recorded relative to the field (GROUPS: mask - 1; a family: its bit mask).
    const auto record = [&](int32_t token) {
        if (draws != nullptr)
            // p_model is the model's own probability: from the logits before any bias (rawForLog).
            draws->push_back(OpLogDraw::describe(fieldName, (rawForLog != nullptr ? rawForLog : vocabLogits) + lo,
                                                 static_cast<size_t>(hi - lo), local,
                                                 start - lo, 1.0f, kComboSampleTemperature, token - lo,
                                                 forced.has_value()));
        return token;
    };
    if (forced.has_value())
        return record(*forced);
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
    return record(start + localIdx);
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
    encoderSession.reset();
    decoderSession.reset();
    cachedHidden.reset();
    juce::String lastError;

    // The split export next to the model: used instead of it (the backbone once per window, the decoder
    // per step; ~10-13 ms a note instead of ~27, the same logits). Nov 24 2026.
    {
        const juce::File full(modelPath);
        const auto enc = full.getSiblingFile(full.getFileNameWithoutExtension() + "_enc.onnx");
        const auto dec = full.getSiblingFile(full.getFileNameWithoutExtension() + "_dec.onnx");
        if (enc.existsAsFile() && dec.existsAsFile()) {
#ifdef __linux__
            (void) sharedOrtEnv();
            try {
                Ort::SessionOptions e, d;
                OrtCUDAProviderOptions cudaProvider{};
                e.AppendExecutionProvider_CUDA(cudaProvider);
                d.AppendExecutionProvider_CUDA(cudaProvider);
                encoderSession = std::make_unique<Ort::Session>(sharedOrtEnv(), enc.getFullPathName().toRawUTF8(), e);
                decoderSession = std::make_unique<Ort::Session>(sharedOrtEnv(), dec.getFullPathName().toRawUTF8(), d);
                splitOnCuda = true;
            } catch (const Ort::Exception &ex) {
                lastError = ex.what();
                encoderSession.reset();
                decoderSession.reset();
            }
#endif
            if (decoderSession == nullptr) {
                try {
                    encoderSession = createOrchestrationSession(enc.getFullPathName().toRawUTF8());
                    decoderSession = createOrchestrationSession(dec.getFullPathName().toRawUTF8());
                    splitOnCuda = false;
                } catch (const Ort::Exception &ex) {
                    lastError = ex.what();
                    encoderSession.reset();
                    decoderSession.reset();
                }
            }
            if (decoderSession != nullptr) {
                modelVocabSize = static_cast<size_t>(IodPretrainedTypes::comboVocabSize);
                return;
            }
            reportSamplingError("IodPretrained: split export failed (" + lastError + "), using the full model");
        }
    }

#ifdef __linux__
    (void) sharedOrtEnv(); // the ORT environment must exist before the CUDA provider is appended
    try {
        Ort::SessionOptions cudaOptions;
        OrtCUDAProviderOptions cudaProvider{};
        cudaOptions.AppendExecutionProvider_CUDA(cudaProvider);
        session = std::make_unique<Ort::Session>(sharedOrtEnv(), modelPath, cudaOptions);
    } catch (const Ort::Exception &ex) {
        lastError = ex.what();
        session.reset();
    }
#endif

    if (session == nullptr) {
        try {
            session = createOrchestrationSession(modelPath);
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

auto IodPretrained::runNoteLogits(IodPretrainedTypes::EncoderWindow &window, int32_t noteIdx) -> std::vector<float> {
    using namespace IodPretrainedTypes;
    const size_t rows = static_cast<size_t>(comboLen) * static_cast<size_t>(comboVocabSize);
    if (decoderSession == nullptr) {
        const auto all = runLogits(window);
        const size_t at = logitsIndex(noteIdx, 0);
        if (all.size() < at + rows)
            return {};
        return std::vector<float>(all.begin() + static_cast<std::ptrdiff_t>(at),
                                  all.begin() + static_cast<std::ptrdiff_t>(at + rows));
    }
    try {
        const std::array<int64_t, 2> idsShape{1, noteSeqLen};
        const std::array<int64_t, 2> validShape{1, nNotesMax};
        const std::array<int64_t, 3> comboShape{1, nNotesMax, comboLen};
        const std::array<int64_t, 1> indexShape{1};
        // the backbone: once while the window's tokens stay the same (only combo_in changes between steps)
        std::vector<int64_t> ids(window.inputIds.begin(), window.inputIds.end());
        std::vector<int64_t> attention(window.attentionMask.begin(), window.attentionMask.end());
        if (! cachedHidden.has_value() || ids != cachedIds || attention != cachedAttention) {
            cachedIds = ids;
            cachedAttention = attention;
            Ort::IoBinding b(*encoderSession);
            b.BindInput("input_ids", Ort::Value::CreateTensor<int64_t>(memoryInfo, cachedIds.data(), cachedIds.size(),
                                                                       idsShape.data(), idsShape.size()));
            b.BindInput("attention_mask", Ort::Value::CreateTensor<int64_t>(memoryInfo, cachedAttention.data(),
                                                                            cachedAttention.size(), idsShape.data(),
                                                                            idsShape.size()));
            if (splitOnCuda)
                b.BindOutput("/backbone/Reshape_2_output_0", Ort::MemoryInfo("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault));
            else
                b.BindOutput("/backbone/Reshape_2_output_0", memoryInfo);
            encoderSession->Run(Ort::RunOptions{nullptr}, b);
            b.SynchronizeOutputs();
            auto out = b.GetOutputValues();
            if (out.empty())
                return {};
            cachedHidden = std::move(out[0]);
        }
        std::array<int64_t, nNotesMax * comboLen> comboFlat{};
        for (int32_t n = 0; n < nNotesMax; ++n)
            for (int32_t s = 0; s < comboLen; ++s)
                comboFlat[static_cast<size_t>(n * comboLen + s)] = window.comboIn[static_cast<size_t>(n)][static_cast<size_t>(s)];
        std::vector<int64_t> valid(window.noteValid.begin(), window.noteValid.end());
        std::array<int64_t, 1> index{noteIdx};
        Ort::IoBinding b(*decoderSession);
        b.BindInput("hidden", *cachedHidden);
        b.BindInput("combo_in", Ort::Value::CreateTensor<int64_t>(memoryInfo, comboFlat.data(), comboFlat.size(),
                                                                  comboShape.data(), comboShape.size()));
        b.BindInput("note_valid", Ort::Value::CreateTensor<int64_t>(memoryInfo, valid.data(), valid.size(),
                                                                    validShape.data(), validShape.size()));
        b.BindInput("note_index", Ort::Value::CreateTensor<int64_t>(memoryInfo, index.data(), index.size(),
                                                                    indexShape.data(), indexShape.size()));
        b.BindOutput("logits", memoryInfo);
        decoderSession->Run(Ort::RunOptions{nullptr}, b);
        b.SynchronizeOutputs();
        auto out = b.GetOutputValues();
        if (out.empty() || out[0].GetTensorTypeAndShapeInfo().GetElementCount() < rows)
            return {};
        const float *data = out[0].GetTensorData<float>();
        return std::vector<float>(data, data + rows);
    } catch (const Ort::Exception &ex) {
        reportSamplingError(juce::String("IodPretrained ORT (split): ") + ex.what());
        cachedHidden.reset();
        return {};
    }
}

auto IodPretrained::sampleComboForNote(IodPretrainedTypes::EncoderWindow &window,
                                       int32_t noteIdx,
                                       const std::vector<int32_t> &instruments,
                                       bool applyGroupsBias,
                                       const std::optional<std::array<int32_t, IodPretrainedTypes::comboLen>> &forced,
                                       std::vector<OpLogDraw::Draw> *draws)
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

    const float tremoloBonus = tremoloBias.load();
    const bool tremoloOn = tremoloBonus > 0.0f
                           && localIdIsActive(instruments, OrchestraTdMeter::kTremoloLocalId);

    {
        auto logits = runNoteLogits(window, noteIdx);
        if (logits.size() < static_cast<size_t>(comboVocabSize))
            return combo;
        float *row = logits.data();
        // The operation log's p_model is the model's own: keep the logits from before the bias.
        std::vector<float> rawRow;
        if (draws != nullptr)
            rawRow.assign(row, row + comboVocabSize);
        const float *raw = rawRow.empty() ? nullptr : rawRow.data();
        if (applyGroupsBias && ! recentGroupsMasks.empty()) {
            const std::vector<int32_t> masks(recentGroupsMasks.begin(), recentGroupsMasks.end());
            applyGroupsTokenLogitsBias(row, static_cast<size_t>(comboVocabSize), masks.data(),
                                       masks.size(), groupsBiasTau.load());
        }
        // TD lever: tremolo strings (strings = family 0, tremolo = its instrument 0, bits 0-2) when the pad is on
        if (tremoloOn && ! forced.has_value())
            for (int32_t id = groupsOffset; id < groupsHi; ++id)
                if ((maskFromGroupsToken(id) & 1) != 0 && std::isfinite(row[id]))
                    row[id] += tremoloBonus;
        if (forced.has_value() && (*forced)[0] == endNote) {
            // The logged note had no orchestra notes: record the model's view of GROUPS, play nothing.
            sampleRange(row, groupsOffset, groupsHi, 0, forbid.fullyBannedFamilyBits, false, nullptr, 0,
                        std::nullopt, draws, "groups", raw);
            if (draws != nullptr && ! draws->empty()) {
                auto &d = draws->back();
                d.forced = true;
                d.chosen = -1;
                d.p = 0.0f;
                d.pModel = 0.0f;
            }
            return combo;
        }
        const auto forcedGroups = forced.has_value() ? std::optional<int32_t>((*forced)[0]) : std::nullopt;
        const int32_t groupsTok = sampleRange(row, groupsOffset, groupsHi,
                                              /*forbidBits*/ 0, forbid.fullyBannedFamilyBits,
                                              /*excludeZero*/ false, nullptr, 0, forcedGroups, draws, "groups", raw);
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

        auto logits = runNoteLogits(window, noteIdx);
        const size_t stepAt = static_cast<size_t>(step) * static_cast<size_t>(comboVocabSize);
        if (logits.size() < stepAt + static_cast<size_t>(comboVocabSize))
            break;

        const auto &family = familySpecs[static_cast<size_t>(familyIdx)];
        if (familyIdx == 0 && tremoloOn && ! forced.has_value()) {
            float *strings = logits.data() + stepAt;
            for (int32_t mask = 1; mask < family.tokenSize; ++mask)
                if ((mask & 0b111) != 0 && std::isfinite(strings[family.tokenOffset + mask]))
                    strings[family.tokenOffset + mask] += tremoloBonus;
        }
        const float *row = logits.data() + stepAt;
        static constexpr const char *kFamilyNames[] = {"strings", "woodwinds", "brass", "other"};
        const auto forcedFam = forced.has_value() && (*forced)[static_cast<size_t>(comboWrite)] != endNote
                                   ? std::optional<int32_t>((*forced)[static_cast<size_t>(comboWrite)])
                                   : std::nullopt;
        const int32_t famTok =
            sampleRange(row, family.tokenOffset, family.tokenOffset + family.tokenSize,
                        forbid.familyForbid[static_cast<size_t>(familyIdx)],
                        /*forbidGroupsFamilies*/ 0,
                        /*excludeZero*/ true, &family, claimedDeltas, forcedFam, draws,
                        kFamilyNames[familyIdx]);
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

auto IodPretrained::ensembleComboB(size_t listIdx, size_t windowBase, const std::vector<int32_t> &instruments)
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
    for (size_t i = windowBase; i < inputNotes.size() && UserSettings::outerVoicesOutward(); ++i) {
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

    const auto filtered = filterPairsUniqueOctave(filterPairsByDeltaAllows(
        filterPairsPlayable(bPairs, pitch, instruments), role.allowMinus1, role.allowPlus1));
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

    if (incomingTokens.empty() || ! isModelLoaded())
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

        if (OpLog::on()) {
            // OT_INPUT (docs/OPLOGS.txt): the encoder input this window's draws start from. The
            // combos decoded within the window are teacher-forced into combo_in as they are drawn
            // (each OT_DRAW's combo), so every draw's input follows from this record.
            OpLog::Strings ids;
            OpLog::Ints inputIds, combos;
            int32_t baseOnset = window.onsetCs[0]; // input_ids' onsets count from the earliest note
            for (int32_t i = 1; i < window.nNotes; ++i)
                baseOnset = std::min(baseOnset, window.onsetCs[static_cast<size_t>(i)]);
            for (int32_t i = 0; i < window.nNotes; ++i) {
                const size_t listIdx = windowBase + start + static_cast<size_t>(i);
                const auto &t = inputNotes[std::min(listIdx, inputNotes.size() - 1)].token;
                ids.push_back(t.keyId >= 0 ? OpLog::id('k', t.keyId) : OpLog::id('r', t.noteId));
                for (int32_t k = 0; k < noteTokensPerNote; ++k)
                    inputIds.push_back(static_cast<int32_t>(window.inputIds[static_cast<size_t>(i * noteTokensPerNote + k)]));
                for (int32_t k = 0; k < comboLen; ++k)
                    combos.push_back(static_cast<int32_t>(window.comboIn[static_cast<size_t>(i)][static_cast<size_t>(k)]));
            }
            OpLog::emit(OpLog::Src::Ot, "OT_INPUT",
                        {{"notes", std::move(ids)}, {"base_onset", baseOnset}, {"input_ids", std::move(inputIds)},
                         {"combo_in", std::move(combos)}, {"tokens_per_note", noteTokensPerNote}, {"combo_len", comboLen}});
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
                std::optional<std::array<int32_t, comboLen>> forced;
                if (forcedCombo)
                    forced = forcedCombo(inputNotes[listIdx].token);
                std::vector<OpLogDraw::Draw> draws;
                const bool logging = OpLog::on();
                const auto comboA = sampleComboForNote(window, i, instruments, applyGroupsBias, forced,
                                                       logging ? &draws : nullptr);
                if (logging) {
                    const auto &in = inputNotes[listIdx].token;
                    std::string steps = "[";
                    for (size_t d = 0; d < draws.size(); ++d)
                        steps += (d ? "," : "") + draws[d].json();
                    steps += "]";
                    OpLog::emit(OpLog::Src::Ot, "OT_DRAW",
                                {{"for", in.keyId >= 0 ? OpLog::id('k', in.keyId) : OpLog::id('r', in.noteId)},
                                 {"pitch", in.getPitch()}, {"onset", in.time},
                                 {"instruments", OpLog::Ints(instruments.begin(), instruments.end())},
                                 {"steps", OpLog::Raw{steps}}, {"combo", OpLog::Ints(comboA.begin(), comboA.end())}});
                }
                {
                    auto &rec = inputNotes[listIdx];
                    rec.dataA = dataFromCombo(comboA);
                    rec.keysA = VocsepEnsemble::sortedUnique(pairsFromCombo(comboA));
                }
                auto comboB = comboA;
                if (useEnsemble) {
                    comboB = ensembleComboB(listIdx, windowBase, instruments);
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
                for (auto &n: notes) {
                    n.token.keyId = inputNotes[listIdx].token.keyId;
                    n.sourceNoteId = inputNotes[listIdx].token.noteId;
                }
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
