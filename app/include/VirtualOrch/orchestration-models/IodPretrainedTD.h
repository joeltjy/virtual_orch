#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "VirtualOrch/TdTokens.h"
#include "VirtualOrch/orchestration-models/IodPretrainedTypes.h"
#include "VirtualOrch/orchestration-models/OrchestrationModel.h"

/**
 * IOD orchestration model with anticipated temporal-density tokens (T14, td_grpo_unk_*.onnx).
 *
 * A plain causal GPT: input_ids [1, n] -> logits [1, n, 20178], recomputed over a window of at
 * most 512 tokens for every sampled slot (no KV cache: learned positions). The stream is
 *   [onset, duration, note, velocity] [GROUPS, strings, woodwind, brass, other] {TD...}
 * per note. The quad is given (the reduction note), the five combo slots are sampled, and one
 * TD token per 2 s boundary follows the first note whose onset reaches that boundary. TD is read
 * from the Launchpad (AUTO = TD unknown) and never sampled.
 */
class IodPretrainedTD : public OrchestrationModel {
public:
    // ---- token layout (README of onnx_export_101826, file 4) ----
    static constexpr int32_t kVocabSize = 20178;
    static constexpr int32_t kDurOffset = 10000;
    static constexpr int32_t kNoteOffset = 11000;
    static constexpr int32_t kVelocityOffset = 11256;
    /** IodPretrainedTypes combo id (>= 1) + kComboShift = id in this vocab. */
    static constexpr int32_t kComboShift = 11383;
    static constexpr int32_t kEndNote = 20167;
    static constexpr int32_t kMaxContext = 512;
    static constexpr int32_t kTokensPerNote = 9;
    /** Notes of context, beyond which the oldest are dropped (9 tokens + TD each). */
    static constexpr int32_t kMaxNotesInContext = 48;
    static constexpr int32_t kMaxOnsetCs = 9999;

    IodPretrainedTD();

    auto init(const char *modelPath) -> void;

    [[nodiscard]] auto getName() const -> std::string override { return "IodPretrainedTD"; }
    [[nodiscard]] auto tokenHistory() const -> const std::vector<OrchestrationNote> & override {
        return history;
    }
    [[nodiscard]] auto isModelLoaded() const -> bool { return session != nullptr; }
    [[nodiscard]] auto supportsVocsepEnsemble() const -> bool override { return true; }

    auto resetState() -> void override {
        OrchestrationModel::resetState();
        history.clear();
        boundaryTd.clear();
        boundariesStarted = false;
        lastAssignedBoundary = 0;
    }

    [[nodiscard]] auto getOutput(
        const std::vector<Token> &incomingTokens,
        const std::vector<int32_t> &instruments,
        const ConditioningSignal &conditioningSignal,
        OrchestrationBalanceTracker *balance = nullptr,
        const OrchestrationBalanceTracker::BiasView *bias = nullptr)
        -> std::vector<OrchestrationNote> override;

    /** Launchpad value for the TD tokens (tests replace it). Defaults to TemporalDensity::get. */
    std::function<int32_t()> temporalDensitySource;

    /** The ids written for each 2 s boundary (absolute cs -> id), for tests. */
    [[nodiscard]] auto boundaryValues() const -> const std::map<int32_t, int32_t> & {
        return boundaryTd;
    }

    // ---- pure helpers (public for tests) ----

    /** Quad ids of a note: onset (rebased), duration (q5), note (track 0), velocity (q8). */
    [[nodiscard]] static auto quadIds(const Token &token, int32_t rebaseCs)
        -> std::array<int64_t, 4>;

    /** Compact combo of IodPretrainedTypes -> the five fixed slots of this model (ids). */
    [[nodiscard]] static auto slotsFromCompact(
        const std::array<int32_t, IodPretrainedTypes::comboLen> &compact)
        -> std::array<int64_t, IodPretrainedTypes::comboLen>;

    /** Inverse of slotsFromCompact. */
    [[nodiscard]] static auto compactFromSlots(
        const std::array<int64_t, IodPretrainedTypes::comboLen> &slots)
        -> std::array<int32_t, IodPretrainedTypes::comboLen>;

    /**
     * Context for sampling the combo of list note `target`: notes [begin, target) with their
     * stored (B) combos and TD tokens, then the quad of `target`. `rebaseCs` is a multiple of
     * the TD boundary spacing.
     */
    [[nodiscard]] auto buildContext(size_t begin, size_t target, int32_t rebaseCs) const
        -> std::vector<int64_t>;

private:
    [[nodiscard]] auto lastRowLogits(const std::vector<int64_t> &ids) -> std::vector<float>;

    auto sampleCombo(size_t listIdx,
                     size_t windowBegin,
                     int32_t rebaseCs,
                     const std::vector<int32_t> &instruments)
        -> std::array<int32_t, IodPretrainedTypes::comboLen>;

    /** Write the TD id of every boundary <= onsetCs that has none yet. */
    auto assignBoundaries(int32_t onsetCs) -> void;

    [[nodiscard]] auto ensembleComboB(size_t listIdx, size_t windowBegin, const std::vector<int32_t> &instruments)
        -> std::array<int32_t, IodPretrainedTypes::comboLen>;

    [[nodiscard]] static auto comboFromData(const std::vector<int32_t> &data)
        -> std::array<int32_t, IodPretrainedTypes::comboLen>;
    [[nodiscard]] static auto dataFromCombo(
        const std::array<int32_t, IodPretrainedTypes::comboLen> &combo) -> std::vector<int32_t>;

    std::vector<OrchestrationNote> history;

    std::map<int32_t, int32_t> boundaryTd;
    bool boundariesStarted = false;
    int32_t lastAssignedBoundary = 0;

    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::string inputName;
    std::string outputName;
    size_t modelVocabSize = 0;
};
