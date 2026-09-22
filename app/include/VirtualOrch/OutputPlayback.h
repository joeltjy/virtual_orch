#pragma once

#include <JuceHeader.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "Clock.h"
#include "Fifo.h"
#include "MusicToken.h"
#include "OSCBufferOutputProcessor.h"
#include "OrchestrationTransformer.h"
#include "OutputProcessor.h"
#include "orchestration-models/OrchestrationModel.h"

/** Which model stream OutputPlayback schedules to MIDI (keyboard thru is independent). */
enum class PlaybackSource : uint8_t {
    Orchestration = 0, // OT instruments (default)
    Reduction = 1,     // RT debug monitor → ch 16
};

/** One NoteOn that OutputPlayback successfully routed (has a MIDI channel). */
struct PlaybackNoteOnRecord {
    int32_t time = 0; // clock time in 1/100s when sent (debug / lag)
    int32_t onsetCs = 0; // token musical onset (1/100s); UI + duration-update matching
    int32_t durationCs = 0; // real duration in centiseconds
    int32_t pitch = 0;
    int32_t velocity = 0; // MIDI 0–127
    int32_t localInstrumentId = 0;
    int32_t channel = 0; // 1-based MIDI channel
};

/**
 * Timed NoteOn/Off → OutputProcessor from OT and/or RT output FIFOs
 * (selected by PlaybackSource). Keyboard thru bypasses this thread.
 */
class OutputPlayback : public juce::Thread {
public:
    OutputPlayback(Clock &clock,
                   OrchestrationTransformer &orchestrationTransformer,
                   std::unique_ptr<OutputProcessor> &outputProcessor,
                   std::unique_ptr<OSCBufferOutputProcessor> &bufferOutputProcessor,
                   int32_t &visualizationBufferSize,
                   std::atomic<PlaybackSource> &playbackSourceIn);

    void run() override;

    /** Bound by the generation-status ProgressBar. */
    double progress = 0.0;

    void resetProgress() { progress = 0.0; }

    /** Optional: true when reduction generation is stopped (manual or overflow). */
    std::function<bool()> isReductionPaused;

    /**
     * Active reduction's outputTokenQueue (not OT reductionIncoming).
     * Set by AppSession::bindActiveMusicBackend; always drained each tick.
     */
    CircularFifo<Token> *reductionOutputQueue = nullptr;

    /**
     * Snapshot of reduction outputHistory for seeding Out: RT after a source switch
     * (FIFO tokens were discarded while Out: OT was selected).
     */
    std::function<std::vector<Token>()> getReductionOutputHistory;

    [[nodiscard]] auto getNoteOnHistory() const -> std::vector<PlaybackNoteOnRecord>;

    /** Only the records still reaching cutoffCs; used by the UI so refresh cost stays bounded. */
    [[nodiscard]] auto getNoteOnHistorySince(int32_t cutoffCs) const
        -> std::vector<PlaybackNoteOnRecord>;

    auto clearNoteOnHistory() -> void;

private:
    enum TokenNoteType {
        TokenNoteOn,
        TokenNoteOff
    };

    void handleNoteForOutput(const OrchestrationNote &note, TokenNoteType tokenEventType);

    auto recordNoteOn(const OrchestrationNote &note, int32_t channel) -> void;

    /** Apply note-off duration correction to already-recorded NoteOns. */
    auto updateNoteOnHistoryDuration(const Token &oldNote, const Token &newNote) -> void;

    Clock &clock;
    OrchestrationTransformer &orchestrationTransformer;
    std::unique_ptr<OutputProcessor> &outputProcessor;
    std::unique_ptr<OSCBufferOutputProcessor> &bufferOutputProcessor;
    int32_t &visualizationBufferSize;
    std::atomic<PlaybackSource> &playbackSource;

    mutable juce::CriticalSection noteOnHistoryLock;
    std::vector<PlaybackNoteOnRecord> noteOnHistory;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(OutputPlayback)
};
