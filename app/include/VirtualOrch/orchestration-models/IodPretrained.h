#pragma once

#include "VirtualOrch/OpLogDraw.h"
#include <functional>
#include <optional>

#include <algorithm>
#include <array>
#include <atomic>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "VirtualOrch/orchestration-models/IodPretrainedTypes.h"
#include "VirtualOrch/orchestration-models/OrchestrationModel.h"

/** iod_pretrained orchestration backend: ot_robust.onnx (robust, expressive-timing; see its .md). */
class IodPretrained : public OrchestrationModel {
public:
    IodPretrained();

    auto init(const char *modelPath) -> void;

    [[nodiscard]] auto getName() const -> std::string override;

    [[nodiscard]] auto tokenHistory() const -> const std::vector<OrchestrationNote> & override;

    [[nodiscard]] auto isModelLoaded() const -> bool { return session != nullptr || decoderSession != nullptr; }

    /** Logits layout: [note][step][vocab] = nNotesMax × comboLen × comboVocabSize. Full model only. */
    [[nodiscard]] auto runLogits(IodPretrainedTypes::EncoderWindow &window)
        -> std::vector<float>;

    /**
     * One note's logits, [step][vocab] = comboLen × comboVocabSize. With the split export
     * (<model>_enc.onnx + <model>_dec.onnx next to the model) the backbone runs once per window
     * (cached on the GPU while input_ids / attention_mask stay the same) and each call runs only
     * the combo decoder for that note; otherwise the full model runs and the note's rows are copied.
     */
    [[nodiscard]] auto runNoteLogits(IodPretrainedTypes::EncoderWindow &window, int32_t noteIdx)
        -> std::vector<float>;

    /** The split export is loaded (backbone + decoder). */
    [[nodiscard]] auto usesSplitModel() const -> bool { return decoderSession != nullptr; }

    [[nodiscard]] auto getOutput(
        const std::vector<Token> &incomingTokens,
        const std::vector<int32_t> &instruments,
        const ConditioningSignal &conditioningSignal,
        OrchestrationBalanceTracker *balance = nullptr,
        const OrchestrationBalanceTracker::BiasView *bias = nullptr)
        -> std::vector<OrchestrationNote> override;

    /** Sample combos for every note in `window` (tests / offline). Mutates combo_in. */
    [[nodiscard]] auto sampleWindowCombos(IodPretrainedTypes::EncoderWindow &window,
                                          const std::vector<int32_t> &instruments,
                                          bool applyGroupsBias = true)
        -> std::vector<std::array<int32_t, IodPretrainedTypes::comboLen>>;

    [[nodiscard]] auto supportsVocsepEnsemble() const -> bool override { return true; }

    auto resetState() -> void override {
        OrchestrationModel::resetState();
        history.clear();
        recentGroupsMasks.clear();
    }

    auto setGroupsBiasTau(float tau) -> void { groupsBiasTau.store(tau); }
    /** The TD lever: + bonus on the GROUPS tokens with strings and the strings tokens with tremolo (if active). */
    auto setTremoloBias(float bonus) -> void override { tremoloBias.store(std::max(0.0f, bonus)); }
    [[nodiscard]] auto getTremoloBias() const -> float { return tremoloBias.load(); }
    [[nodiscard]] auto getGroupsBiasTau() const -> float { return groupsBiasTau.load(); }

private:
    [[nodiscard]] auto sampleComboForNote(IodPretrainedTypes::EncoderWindow &window,
                                          int32_t noteIdx,
                                          const std::vector<int32_t> &instruments,
                                          bool applyGroupsBias,
                                          const std::optional<std::array<int32_t, IodPretrainedTypes::comboLen>> &forced = std::nullopt,
                                          std::vector<OpLogDraw::Draw> *draws = nullptr)
        -> std::array<int32_t, IodPretrainedTypes::comboLen>;

    auto recordGroupsMask(int32_t groupsMask) -> void;

    static auto writeComboInFromStored(
        IodPretrainedTypes::EncoderWindow &window,
        int32_t noteIdx,
        const std::array<int32_t, IodPretrainedTypes::comboLen> &combo) -> void;

    /**
     * Effective combo B for the note at `listIdx` of the shared input-note list (keysA/dataA
     * already set): shared vocsep ensemble on instrument x octave-delta pairs, then the IOD-only
     * outer-voice and unique-octave filters over the encoder window [windowBase, end). An empty
     * result falls back to A.
     */
    [[nodiscard]] auto ensembleComboB(size_t listIdx, size_t windowBase, const std::vector<int32_t> &instruments)
        -> std::array<int32_t, IodPretrainedTypes::comboLen>;

    [[nodiscard]] static auto comboFromData(const std::vector<int32_t> &data)
        -> std::array<int32_t, IodPretrainedTypes::comboLen>;

    [[nodiscard]] static auto dataFromCombo(
        const std::array<int32_t, IodPretrainedTypes::comboLen> &combo) -> std::vector<int32_t>;

    std::vector<OrchestrationNote> history;

    /** Last GROUPS masks (1…15) from sampled notes — drives −τ log π_y. */
    std::deque<int32_t> recentGroupsMasks;

public:
    /**
     * Replay hook: the combo to force for an input note (the logged choice), or nullopt to sample.
     * The model still runs, so the operation log records its probabilities for the forced tokens.
     */
    std::function<std::optional<std::array<int32_t, IodPretrainedTypes::comboLen>>(const Token &input)> forcedCombo;

private:

    std::atomic<float> groupsBiasTau{IodPretrainedTypes::groupsBiasTau};
    std::atomic<float> tremoloBias{0.0f};

    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    // the split export: backbone (input_ids, attention_mask -> hidden) and combo decoder
    // (hidden, combo_in, note_valid, note_index -> one note's logits)
    std::unique_ptr<Ort::Session> encoderSession;
    std::unique_ptr<Ort::Session> decoderSession;
    bool splitOnCuda = false;
    std::optional<Ort::Value> cachedHidden;
    std::vector<int64_t> cachedIds;
    std::vector<int64_t> cachedAttention;

    std::vector<std::string> allocatedInputNames;
    std::vector<std::string> allocatedOutputNames;

    size_t modelVocabSize = 0;
};
