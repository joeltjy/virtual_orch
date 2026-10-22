#pragma once

#include <array>
#include <atomic>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "VirtualOrch/orchestration-models/IodPretrainedTypes.h"
#include "VirtualOrch/orchestration-models/OrchestrationModel.h"

/** iod_pretrained / instrument_octave_predictions.onnx orchestration backend. */
class IodPretrained : public OrchestrationModel {
public:
    IodPretrained();

    auto init(const char *modelPath) -> void;

    [[nodiscard]] auto getName() const -> std::string override;

    [[nodiscard]] auto tokenHistory() const -> const std::vector<OrchestrationNote> & override;

    [[nodiscard]] auto isModelLoaded() const -> bool { return session != nullptr; }

    /** Logits layout: [note][step][vocab] = nNotesMax × comboLen × comboVocabSize. */
    [[nodiscard]] auto runLogits(IodPretrainedTypes::EncoderWindow &window)
        -> std::vector<float>;

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
    [[nodiscard]] auto getGroupsBiasTau() const -> float { return groupsBiasTau.load(); }

private:
    [[nodiscard]] auto sampleComboForNote(IodPretrainedTypes::EncoderWindow &window,
                                          int32_t noteIdx,
                                          const std::vector<int32_t> &instruments,
                                          bool applyGroupsBias)
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

    std::atomic<float> groupsBiasTau{IodPretrainedTypes::groupsBiasTau};

    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<std::string> allocatedInputNames;
    std::vector<std::string> allocatedOutputNames;

    size_t modelVocabSize = 0;
};
