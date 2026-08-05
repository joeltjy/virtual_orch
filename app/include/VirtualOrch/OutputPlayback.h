#pragma once

#include <JuceHeader.h>
#include <memory>

#include "Clock.h"
#include "OSCBufferOutputProcessor.h"
#include "OrchestrationTransformer.h"
#include "OutputProcessor.h"
#include "orchestration-models/OrchestrationModel.h"

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

private:
    enum TokenNoteType {
        TokenNoteOn,
        TokenNoteOff
    };

    void handleNoteForOutput(const OrchestrationNote &note, TokenNoteType tokenEventType);

    Clock &clock;
    OrchestrationTransformer &orchestrationTransformer;
    std::unique_ptr<OutputProcessor> &outputProcessor;
    std::unique_ptr<OSCBufferOutputProcessor> &bufferOutputProcessor;
    int32_t &visualizationBufferSize;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(OutputPlayback)
};
