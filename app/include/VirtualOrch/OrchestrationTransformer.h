#pragma once

#include <JuceHeader.h>

#include <array>
#include <atomic>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "Fifo.h"
#include "VirtualOrch/Clock.h"
#include "VirtualOrch/HoldBackQueue.h"
#include "VirtualOrch/InstrumentLogitSnapshot.h"
#include "VirtualOrch/MusicTransformer.h"
#include "VirtualOrch/OctaveDeltaTracker.h"
#include "VirtualOrch/OrchestrationBalanceTracker.h"
#include "VirtualOrch/OrchLoopProfile.h"
#include "VirtualOrch/orchestration-models/OrchestrationModel.h"
#include "VirtualOrch/vocsep/VoiceSeparation.h"

enum class OrchestrationMode : uint8_t {
    Edit,
    Jam,
    /** Jam plus the polyphony cap; live input never clears the queue or rewinds the model. */
    Accompaniment
};

/** Jam and Accompaniment orchestrate the reduction stream with user + model instruments. */
[[nodiscard]] inline auto isJamLike(OrchestrationMode mode) -> bool {
    return mode == OrchestrationMode::Jam || mode == OrchestrationMode::Accompaniment;
}

enum class InstrumentUpdateTarget : uint8_t {
    User,
    Model
};

struct InstrumentUpdate {
    int32_t localInstrumentId = 0;
    InstrumentUpdateTarget target = InstrumentUpdateTarget::User;
    uint8_t state = 0; // 0 = clear, 1 = set
};

struct OrchestrationDebugSnapshot {
    std::vector<Token> midiHistory;
    std::vector<Token> midiPending;
    std::vector<Token> conditioningHistory;
    std::vector<Token> conditioningPending;
    std::vector<Token> reductionHistory;
    std::vector<Token> reductionPending;
    std::vector<OrchestrationNote> outputHistory;
    std::set<int32_t> userInstruments;
    std::set<int32_t> modelInstruments;
};

class OrchestrationTransformer : public juce::Thread {
public:
    static constexpr size_t CONDITIONING_SIGNAL_DIM = 64;
    static constexpr int32_t kNumInstruments = 128;
    using ConditioningSignal = std::array<float, CONDITIONING_SIGNAL_DIM>;

    OrchestrationTransformer();

    void run() override {
        threadInit();
        threadRun();
        threadStop();
    }

    void threadInit();
    void threadRun();
    void threadStop();

    CircularFifo<Token> midiInputIncoming;
    CircularFifo<Token> conditioningIncoming;
    CircularFifo<Token> reductionIncoming;
    CircularFifo<OrchestrationNote> outputTokenQueue;
    CircularFifo<InstrumentUpdate> instrumentUpdates;
    CircularFifo<TokenUpdate> updatesIncoming;

    /** Vocsep re-stamps of notes the reduction already sent (see VoiceRestamp). */
    CircularFifo<VoiceRestamp> restampsIncoming;

    /**
     * App clock (set by AppSession). Reduction notes are held back un-orchestrated until their
     * onset is within holdBackWindowCs of it (see HoldBackQueue.h). Without a clock every note
     * is orchestrated on arrival, as before.
     */
    Clock *clock = nullptr;

    /**
     * Edit mode: the user's piano notes get vocsep voice ids too (their own instance, voices and
     * history, apart from the reduction's), so the user instruments follow voices like the
     * model instruments do. nullptr / unloaded: user notes have no voice, as before.
     */
    VoiceSeparation *inputVoiceSeparation = nullptr;
    /** Note ids given to stamped user notes (far above the reduction's, so re-stamps never clash). */
    static constexpr int32_t kInputNoteIdBase = 1 << 30;
    /** Edit: stamp the voices of a batch of user notes (OT thread; exposed for tests). */
    auto stampInputVoices(std::vector<Token> &midiBatch) -> void;
    [[nodiscard]] auto inputVocsepHistoryForTest() const -> const std::vector<Token> & { return inputVocsepHistory; }
    std::atomic<int32_t> holdBackWindowCs{HoldBackQueue::kDefaultWindowCs};
    /** Reduction notes currently waiting for their turn (debug). */
    [[nodiscard]] auto heldCount() const -> size_t { return heldCountValue.load(); }
    /** Duration/content corrections for notes already pushed to OutputPlayback. */
    CircularFifo<TokenUpdate> outputDurationUpdates;

    /** Latest processed orchestra note. */
    int32_t currentOTTime = 0;

    [[nodiscard]] auto getCurrentOTTime() const -> int32_t { return currentOTTime; }

    /** Local instrument ids used by Edit/Jam and ActiveInstruments UI. */
    std::set<int32_t> userInstruments;
    std::set<int32_t> modelInstruments;

    /**
     * Apply one Launchpad pad/side update immediately (MIDI thread safe).
     * Prefer this over waiting for the OT loop — ONNX can block applyInstrumentUpdates
     * for hundreds of ms.
     */
    auto applyOneInstrumentUpdate(const InstrumentUpdate &update) -> void;

    /** Drain instrumentUpdates FIFO into applyOneInstrumentUpdate. */
    auto applyInstrumentUpdates() -> void;

    /** Replace both sets under lock (generation-start resync from Launchpad). */
    auto replaceInstrumentSets(std::set<int32_t> user, std::set<int32_t> model) -> void;

    /** Optional: rebuild user/model sets from Launchpad effective pads at thread start. */
    std::function<void()> rebuildInstrumentsFromPads;

    [[nodiscard]] auto getMode() const -> OrchestrationMode {
        return static_cast<OrchestrationMode>(modeStorage.get());
    }

    auto setMode(OrchestrationMode newMode) -> void {
        modeStorage.set(static_cast<int>(newMode));
    }

    /** Soft pause (manual / Launchpad): drain/history/ClearQueue continue; skip getOutput. */
    juce::Atomic<bool> paused{false};

    /**
     * When true, IC family-combo logits get the proportion diversity bias
     * (settings toggle; default on).
     */
    juce::Atomic<bool> proportionBiasEnabled{true};

    OrchestrationBalanceTracker userBalance;
    OrchestrationBalanceTracker modelBalance;
    OctaveDeltaTracker octaveDeltaTracker;

    /** Latest post-bias singleton logits (model stream preferred in Edit). */
    [[nodiscard]] auto getInstrumentLogitSnapshot() const -> InstrumentLogitSnapshot;

    [[nodiscard]] auto getOctaveDeltaSnapshot() const -> OctaveDeltaSnapshot {
        return octaveDeltaTracker.snapshot();
    }

    /** Most recent threadRun iteration duration in milliseconds. */
    [[nodiscard]] auto getLastThreadLoopMs() const -> float {
        return lastThreadLoopMs.load(std::memory_order_relaxed);
    }

    /** Last busy OT iteration + running max (Mode widget). */
    [[nodiscard]] auto getLastOrchProfile() const -> OrchLoopProfile;

    /** Written by OrchestrationModel::getOutput during the current OT iteration. */
    OrchLoopStepMs getOutputAccum;

    OrchestrationModel *orchestrationModel = nullptr;

    [[nodiscard]] auto getMidiInputHistory() const -> const std::vector<Token> & {
        return midiInputHistory;
    }

    [[nodiscard]] auto getConditioningHistory() const -> const std::vector<Token> & {
        return conditioningHistory;
    }

    [[nodiscard]] auto getReductionHistory() const -> const std::vector<Token> & {
        return reductionHistory;
    }

    static auto drainIncoming(CircularFifo<Token> &incoming) -> std::vector<Token>;
    static auto appendToHistory(std::vector<Token> &history, const std::vector<Token> &batch) -> void;
    static auto tokenWithInstrument(Token token, int32_t instrument) -> Token;
    static auto sortTokensByTimeThenDuration(std::vector<Token> &tokens) -> void;

    /** Move ClearQueue tokens out of `tokens` into returned list (order preserved). */
    static auto extractClearQueueTokens(std::vector<Token> &tokens) -> std::vector<Token>;

    /**
     * Cluster note onsets within 30 ms (3 cs) and set each cluster to the mean
     * onset, rounded to the 10 ms (1 cs) grid. Non-note / control tokens unchanged.
     * Mutates `tokens` in place (order preserved).
     */
    static auto snapNearbyOnsets(std::vector<Token> &tokens) -> void;

    /** Drain updatesIncoming; patch midiInputHistory and conditioningHistory. */
    auto applyTokenUpdates() -> void;

    auto publishDebugSnapshot(OrchestrationDebugSnapshot snapshot) -> void;

    [[nodiscard]] auto getDebugSnapshot() const -> OrchestrationDebugSnapshot;

    /** Snapshot with every history trimmed to what reaches cutoffCs (UI refresh path). */
    [[nodiscard]] auto getDebugSnapshotSince(int32_t cutoffCs) const -> OrchestrationDebugSnapshot;

private:
    juce::Atomic<int> modeStorage{static_cast<int>(OrchestrationMode::Jam)};
    std::atomic<float> lastThreadLoopMs{0.0f};

    mutable juce::CriticalSection orchProfileLock;
    OrchLoopProfile orchProfile;

    auto resetOrchProfile() -> void;
    auto publishBusyOrchProfile(OrchLoopStepMs step) -> void;

    std::vector<Token> midiInputHistory;
    std::vector<Token> inputVocsepHistory; // the user notes vocsep has seen (Edit)
    int32_t nextInputNoteId = kInputNoteIdBase;
    std::vector<Token> conditioningHistory;
    std::vector<Token> reductionHistory;
    std::vector<OrchestrationNote> outputHistory;

    mutable juce::CriticalSection debugSnapshotLock;
    OrchestrationDebugSnapshot debugSnapshot;

    mutable juce::CriticalSection instrumentLogitLock;
    InstrumentLogitSnapshot instrumentLogitSnapshot;

    auto clearDebugSnapshot() -> void;

    auto publishInstrumentLogits(InstrumentLogitSnapshot snapshot) -> void;

    mutable juce::CriticalSection instrumentsLock;

    [[nodiscard]] auto copyInstrumentLists() const
        -> std::pair<std::vector<int32_t>, std::vector<int32_t>>;

    auto getConditioningSignal(const std::vector<Token> &midiInput,
                               const std::vector<Token> &conditioning) -> ConditioningSignal;

    auto getEditOrchestrationOutput(const std::vector<Token> &midiInput,
                                    const std::vector<Token> &reductionInput,
                                    const std::vector<int32_t> &userInstruments,
                                    const std::vector<int32_t> &modelInstruments,
                                    const ConditioningSignal &signal)
        -> std::pair<std::vector<OrchestrationNote>, std::vector<OrchestrationNote>>;

    auto getJamOrchestrationOutput(const std::vector<Token> &orchestrationInput,
                                   const std::vector<int32_t> &orchestrationInstruments,
                                   const ConditioningSignal &signal) -> std::vector<OrchestrationNote>;

    auto packSortAndPushOutput(const std::vector<OrchestrationNote> &notes) -> void;

    /** Push one note to outputTokenQueue and append to outputHistory together. */
    auto pushOutputNote(const OrchestrationNote &note) -> void;

    auto clearOutputNotes() -> void;

    auto clearMidiInputIncoming() -> void {
        Token t{};
        while (midiInputIncoming.pull(t)) {
        }
    }

    auto clearConditioningIncoming() -> void {
        Token t{};
        while (conditioningIncoming.pull(t)) {
        }
    }

    auto clearReductionIncoming() -> void {
        Token t{};
        while (reductionIncoming.pull(t)) {
        }
    }

    auto clearInstrumentUpdates() -> void {
        InstrumentUpdate u{};
        while (instrumentUpdates.pull(u)) {
        }
    }

    auto clearRestampsIncoming() -> void {
        VoiceRestamp r{};
        while (restampsIncoming.pull(r)) {
        }
    }

    /**
     * Drain restampsIncoming: patch the voiceId of the batch just drained (a re-stamp can
     * arrive together with the note it changes), of reductionHistory, and of the model's own
     * note list. Runs on the OT thread, before the model sees the batch.
     */
    auto applyVoiceRestamps(std::vector<Token> &reductionBatch) -> void;

    HoldBackQueue holdBack;
    std::atomic<size_t> heldCountValue{0};

    auto clearUpdatesIncoming() -> void {
        TokenUpdate u{};
        while (updatesIncoming.pull(u)) {
        }
    }

    auto clearOutputDurationUpdates() -> void {
        TokenUpdate u{};
        while (outputDurationUpdates.pull(u)) {
        }
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(OrchestrationTransformer)
};
