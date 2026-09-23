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

    auto setEnsembleEnabled(bool enabled) -> void { ensembleEnabled.store(enabled); }
    [[nodiscard]] auto getEnsembleEnabled() const -> bool { return ensembleEnabled.load(); }

    auto setEnsembleAddP(float p) -> void { ensembleAddP.store(p); }
    [[nodiscard]] auto getEnsembleAddP() const -> float { return ensembleAddP.load(); }

    auto setEnsembleRemoveP(float p) -> void { ensembleRemoveP.store(p); }
    [[nodiscard]] auto getEnsembleRemoveP() const -> float { return ensembleRemoveP.load(); }

private:
    struct StreamNote {
        Token token{};
        bool hasCombo = false;
        /** Raw model sample (A). */
        std::array<int32_t, IodPretrainedTypes::comboLen> comboA{};
        /** Effective combo for combo_in + decode (B). */
        std::array<int32_t, IodPretrainedTypes::comboLen> comboB{};
    };

    [[nodiscard]] auto sampleComboForNote(IodPretrainedTypes::EncoderWindow &window,
                                          int32_t noteIdx,
                                          const std::vector<int32_t> &instruments,
                                          bool applyGroupsBias)
        -> std::array<int32_t, IodPretrainedTypes::comboLen>;

    auto recordGroupsMask(int32_t groupsMask) -> void;

    auto clearStream() -> void;

    /** Drop stream if `nextOnset` is more than historyGapClearCs after the last note. */
    auto clearStreamIfGapBefore(int32_t nextOnset) -> void;

    /** Append RT notes; returns index of first newly appended note (0 if stream was cleared). */
    auto appendIncoming(const std::vector<Token> &incoming) -> size_t;

    /** Keep last nNotesMax; adjust `newBegin` for dropped prefix. */
    auto trimStreamToEncoderCap(size_t &newBegin) -> void;

    static auto writeComboInFromStored(
        IodPretrainedTypes::EncoderWindow &window,
        int32_t noteIdx,
        const std::array<int32_t, IodPretrainedTypes::comboLen> &combo) -> void;

    /** Same-voice ensemble: A → B using previous B + past-10 A proportions. */
    [[nodiscard]] auto ensembleComboB(size_t noteIdx,
                                      const std::array<int32_t, IodPretrainedTypes::comboLen> &comboA)
        -> std::array<int32_t, IodPretrainedTypes::comboLen>;

    std::vector<OrchestrationNote> history;
    std::vector<StreamNote> noteStream;

    /** Last GROUPS masks (1…15) from sampled notes — drives −τ log π_y. */
    std::deque<int32_t> recentGroupsMasks;

    std::atomic<bool> ensembleEnabled{false};
    std::atomic<float> ensembleAddP{IodPretrainedTypes::ensembleAddPDefault};
    std::atomic<float> ensembleRemoveP{IodPretrainedTypes::ensembleRemovePDefault};

    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<std::string> allocatedInputNames;
    std::vector<std::string> allocatedOutputNames;

    size_t modelVocabSize = 0;
};
