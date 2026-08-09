#pragma once

#include <JuceHeader.h>
#include <functional>
#include <memory>
#include <vector>

#include "Clock.h"
#include "OSCBufferOutputProcessor.h"
#include "OrchestrationTransformer.h"
#include "OutputProcessor.h"
#include "orchestration-models/OrchestrationModel.h"

/** One NoteOn that OutputPlayback successfully routed (has a MIDI channel). */
struct PlaybackNoteOnRecord {
    int32_t time = 0; // clock time in 1/100s when sent
    int32_t pitch = 0;
    int32_t velocity = 0; // MIDI 0–127
    int32_t localInstrumentId = 0;
    int32_t channel = 0; // 1-based MIDI channel
};

/**
 * OrchestrationTransformer outputTokenQueue -> timed NoteOn/Off -> OutputProcessor.
 */
class OutputPlayback : public juce::Thread {
public:
    OutputPlayback(Clock &clock,
                   OrchestrationTransformer &orchestrationTransformer,
                   std::unique_ptr<OutputProcessor> &outputProcessor,
                   std::unique_ptr<OSCBufferOutputProcessor> &bufferOutputProcessor,
                   int32_t &visualizationBufferSize);

    void run() override;

    /** Bound by the generation-status ProgressBar. */
    double progress = 0.0;

    void resetProgress() { progress = 0.0; }

    /** Optional: true when the active ReductionTransformer is soft-paused. */
    std::function<bool()> isReductionPaused;

    [[nodiscard]] auto getNoteOnHistory() const -> std::vector<PlaybackNoteOnRecord>;

    auto clearNoteOnHistory() -> void;

private:
    enum TokenNoteType {
        TokenNoteOn,
        TokenNoteOff
    };

    void handleNoteForOutput(const OrchestrationNote &note, TokenNoteType tokenEventType);

    auto recordNoteOn(const OrchestrationNote &note, int32_t channel) -> void;

    Clock &clock;
    OrchestrationTransformer &orchestrationTransformer;
    std::unique_ptr<OutputProcessor> &outputProcessor;
    std::unique_ptr<OSCBufferOutputProcessor> &bufferOutputProcessor;
    int32_t &visualizationBufferSize;

    mutable juce::CriticalSection noteOnHistoryLock;
    std::vector<PlaybackNoteOnRecord> noteOnHistory;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(OutputPlayback)
};
