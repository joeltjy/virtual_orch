#pragma once

#include <JuceHeader.h>

#include <array>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "Fifo.h"
#include "VirtualOrch/MusicTransformer.h"

enum class OrchestrationMode : uint8_t {
    Edit,
    Jam
};

using OrchestrationNote = std::pair<Token, int32_t>;

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
    CircularFifo<Token> outputTokenQueue;
    CircularFifo<InstrumentUpdate> instrumentUpdates;
    CircularFifo<TokenUpdate> updatesIncoming;

    /** Local instrument ids used by Edit/Jam and ActiveInstruments UI. */
    std::set<int32_t> userInstruments;
    std::set<int32_t> modelInstruments;

    OrchestrationMode mode = OrchestrationMode::Edit;

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

    auto applyInstrumentUpdates() -> void;

    /** Drain updatesIncoming; patch midiInputHistory and conditioningHistory. */
    auto applyTokenUpdates() -> void;

    auto publishDebugSnapshot(OrchestrationDebugSnapshot snapshot) -> void;

    [[nodiscard]] auto getDebugSnapshot() const -> OrchestrationDebugSnapshot;

private:
    std::vector<Token> midiInputHistory;
    std::vector<Token> conditioningHistory;
    std::vector<Token> reductionHistory;

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

    auto clearOutputTokenQueue() -> void {
        Token t{};
        while (outputTokenQueue.pull(t)) {
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

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(OrchestrationTransformer)
};
