#pragma once

#include <JuceHeader.h>

class Clock;
class OrchestrationTransformer;

/**
 * Test feeder: every 0.5s of transport time pushes synthetic tokens into OT FIFOs.
 * note = (n % 8) + 48 → midiInputIncoming
 * note = (n % 8) + 60 → reductionIncoming
 */
class TestOrchestrationTransformerThread : public juce::Thread {
public:
    TestOrchestrationTransformerThread();
    ~TestOrchestrationTransformerThread() override;

    auto start(Clock &clock, OrchestrationTransformer &orchestrationTransformer) -> void;

    auto stop() -> void;

private:
    void run() override;

    Clock *clock = nullptr;
    OrchestrationTransformer *orchestrationTransformer = nullptr;
    int32_t lastFiredBoundary = -1;
};
