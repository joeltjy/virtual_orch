#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "VirtualOrch/ModelSamplingAlert.h"
#include "VirtualOrch/MusicTransformer.h"
#include "VirtualOrch/OctaveDeltaTracker.h"
#include "VirtualOrch/OrchestrationBalanceTracker.h"
#include "VirtualOrch/OrchLoopProfile.h"
#include "VirtualOrch/InstrumentLogitSnapshot.h"

struct OrchestrationNote {
    Token token;
    int32_t velocity = 0;
    int32_t localInstrumentId = 0;
};

/**
 * Base class for orchestration models (non-threaded computation only).
 */
class OrchestrationModel {
public:
    static constexpr size_t CONDITIONING_SIGNAL_DIM = 64;
    using ConditioningSignal = std::array<float, CONDITIONING_SIGNAL_DIM>;

    virtual ~OrchestrationModel() = default;

    [[nodiscard]] virtual auto getName() const -> std::string = 0;

    [[nodiscard]] virtual auto tokenHistory() const -> const std::vector<OrchestrationNote> & = 0;

    /**
     * @param balance  When non-null, record per-note instrument proportions after decode.
     * @param bias     When non-null and bias->apply, adjust family-combo logits before sample.
     */
    [[nodiscard]] virtual auto getOutput(
        const std::vector<Token> &incomingTokens,
        const std::vector<int32_t> &instruments,
        const ConditioningSignal &conditioningSignal,
        OrchestrationBalanceTracker *balance = nullptr,
        const OrchestrationBalanceTracker::BiasView *bias = nullptr)
        -> std::vector<OrchestrationNote> = 0;

    int32_t maxContextLength = 256;
    std::vector<int32_t> tokenHistorySequence;

    // ---- Input-note list + vocsep ensemble (shared by every orchestration model) ----

    /** Which call a batch of incoming notes belongs to (the OT sets it before getOutput). */
    enum class InputStream : int32_t {
        User = 0,      // the user's MIDI notes (Edit mode, user instruments)
        Reduction = 1, // notes from the reduction transformer
    };

    /** Kept per incoming note: what the model chose (A) and the voice-smoothed choice (B). */
    struct InputNote {
        Token token;
        InputStream stream = InputStream::Reduction;
        bool hasResult = false;
        /** Model-neutral keys (sorted, unique) of the raw choice A and the effective choice B. */
        std::vector<int32_t> keysA;
        std::vector<int32_t> keysB;
        /** Model-specific encoding of A and B (e.g. combo tokens), owned by the model. */
        std::vector<int32_t> dataA;
        std::vector<int32_t> dataB;
    };

    /** Same cap as the reduction transformers' output history (NoteWindow::maxHistoryNotes). */
    static constexpr size_t kInputNoteCap = 256;
    static constexpr float kEnsembleAddPDefault = 0.67f;
    static constexpr float kEnsembleRemovePDefault = 0.3f;

    /**
     * Forget everything the model remembers about the performance so far (its note list, its
     * history, per-model windows). Called when the orchestration thread (re)starts, so a Start
     * or a force restart begins from a clean slate. Subclasses clear their own state and call
     * this base version.
     */
    virtual auto resetState() -> void {
        inputNotes.clear();
        tokenHistorySequence.clear();
    }

    /** Models that can smooth their per-note choice over a voice return true. */
    [[nodiscard]] virtual auto supportsVocsepEnsemble() const -> bool { return false; }

    auto setVocsepEnsembleEnabled(bool enabled) -> void { ensembleEnabled.store(enabled); }
    [[nodiscard]] auto getVocsepEnsembleEnabled() const -> bool { return ensembleEnabled.load(); }
    auto setVocsepEnsembleAddP(float p) -> void { ensembleAddP.store(p); }
    [[nodiscard]] auto getVocsepEnsembleAddP() const -> float { return ensembleAddP.load(); }
    auto setVocsepEnsembleRemoveP(float p) -> void { ensembleRemoveP.store(p); }
    [[nodiscard]] auto getVocsepEnsembleRemoveP() const -> float { return ensembleRemoveP.load(); }

    /** Written by the OT thread before each getOutput call (like the sinks below). */
    InputStream currentStream = InputStream::Reduction;

    [[nodiscard]] auto getInputNotes() const -> const std::vector<InputNote> & { return inputNotes; }

    /** Vocsep re-stamped an earlier note: update its voiceId in the list (OT thread). */
    auto applyVoiceRestamp(int32_t noteId, int32_t newVoiceId) -> void;

    /** Set by AppSession; used to surface ORT / sampling failures in the Mode widget. */
    ModelSamplingAlert *samplingAlert = nullptr;

    /** Accumulator for getOutput step timings; OT resets each iteration. */
    OrchLoopStepMs *getOutputTimings = nullptr;

    /**
     * Optional sink for post-bias singleton instrument logits (debug bar chart).
     * Written by InstrumentCombinations after masking/bias, before sample.
     */
    InstrumentLogitSnapshot *instrumentLogitSink = nullptr;

    /** Optional sink for IOD octave-Δ counts (debug bars). Written by IodPretrained decode. */
    OctaveDeltaTracker *octaveDeltaSink = nullptr;

protected:
    /**
     * Append the sounding notes of `incoming` (tagged with currentStream) to inputNotes and
     * trim to kInputNoteCap. Call first thing in getOutput, like the reductions call
     * pushOutputToken. Returns the index of the first note appended by this call (after the
     * trim). With gapClearCs > 0 the whole list is dropped when a note starts more than that
     * many cs after the last one; `cleared` is then set.
     */
    auto recordIncoming(const std::vector<Token> &incoming,
                        int32_t gapClearCs = 0,
                        bool *cleared = nullptr) -> size_t;

    /** The sounding notes of `tokens` (drops ClearQueue, rests, bar separators). */
    [[nodiscard]] static auto soundingNotes(const std::vector<Token> &tokens) -> std::vector<Token>;

    /** B keys for inputNotes[idx] (its keysA must be set) with the current Add P / Remove P. */
    [[nodiscard]] auto ensembleKeysB(size_t idx) const -> std::vector<int32_t>;

    std::vector<InputNote> inputNotes;

    auto reportSamplingError(const juce::String &detail) -> void {
        if (samplingAlert != nullptr)
            samplingAlert->report(juce::String(getName()), detail);
    }

private:
    std::atomic<bool> ensembleEnabled{false};
    std::atomic<float> ensembleAddP{kEnsembleAddPDefault};
    std::atomic<float> ensembleRemoveP{kEnsembleRemovePDefault};
};
