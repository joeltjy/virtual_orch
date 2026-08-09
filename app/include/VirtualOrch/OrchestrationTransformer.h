#pragma once

#include <JuceHeader.h>

#include <array>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "Fifo.h"
#include "VirtualOrch/MusicTransformer.h"
#include "VirtualOrch/orchestration-models/OrchestrationModel.h"

enum class OrchestrationMode : uint8_t {
    Edit,
    Jam
};

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
    /** Duration/content corrections for notes already pushed to OutputPlayback. */
    CircularFifo<TokenUpdate> outputDurationUpdates;

    /** Latest processed orchestra note. */
    int32_t currentOTTime = 0;

    [[nodiscard]] auto getCurrentOTTime() const -> int32_t { return currentOTTime; }

    /** Local instrument ids used by Edit/Jam and ActiveInstruments UI. */
    std::set<int32_t> userInstruments;
    std::set<int32_t> modelInstruments;

    [[nodiscard]] auto getMode() const -> OrchestrationMode {
        return static_cast<OrchestrationMode>(modeStorage.get());
    }

    auto setMode(OrchestrationMode newMode) -> void {
        modeStorage.set(static_cast<int>(newMode));
    }

    /** Soft pause (manual / Launchpad): drain/history/ClearQueue continue; skip getOutput. */
    juce::Atomic<bool> paused{false};

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

    auto applyInstrumentUpdates() -> void;

    /** Drain updatesIncoming; patch midiInputHistory and conditioningHistory. */
    auto applyTokenUpdates() -> void;

    auto publishDebugSnapshot(OrchestrationDebugSnapshot snapshot) -> void;

    [[nodiscard]] auto getDebugSnapshot() const -> OrchestrationDebugSnapshot;

private:
    juce::Atomic<int> modeStorage{static_cast<int>(OrchestrationMode::Edit)};

    std::vector<Token> midiInputHistory;
    std::vector<Token> conditioningHistory;
    std::vector<Token> reductionHistory;
    std::vector<OrchestrationNote> outputHistory;

    mutable juce::CriticalSection debugSnapshotLock;
    OrchestrationDebugSnapshot debugSnapshot;

    auto clearDebugSnapshot() -> void;

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
