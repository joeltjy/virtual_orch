#pragma once

#include <JuceHeader.h>

#include <array>
#include <utility>
#include <vector>

#include "Fifo.h"
#include "VirtualOrch/MusicTransformer.h"

enum class OrchestrationMode : uint8_t {
    Edit,
    Jam
};

/** Token plus instrument id before packing into Token.note, for easier interpretation.*/
using OrchestrationNote = std::pair<Token, int32_t>;

class OrchestrationTransformer : public juce::Thread {
public:
    static constexpr size_t CONDITIONING_SIGNAL_DIM = 64;
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

private:
    std::vector<Token> midiInputHistory;
    std::vector<Token> conditioningHistory;
    std::vector<Token> reductionHistory;

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

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(OrchestrationTransformer)
};
