#pragma once

#include <JuceHeader.h>

#include "Metrics.h"

class MetricsComponent : public juce::Component, juce::Thread {
public:
    MetricsComponent(Metrics &metrics);

    ~MetricsComponent() override;

    void paint(Graphics &g) override;

    void resized() override;

    void userTriedToCloseWindow() override;

    void run() override;

    CircularFifo<float> tokenLatencyFifo;
    CircularFifo<float> responsivenessFifo;

private:
    std::vector<std::vector<float> > y{{}};

    uint32_t MAX_SIZE = 500;
    uint32_t FRAME_RATE = 30;

    Metrics &metrics;

    std::deque<float> tokenLatencyBuffer;
    std::deque<float> responsivenessBuffer;
    float runningTokenLatencySum = 0.0f;
    float runningResponsivenessSum = 0.0f;

    static constexpr size_t maxSize = 500;

    juce::Label averageTokenLatencyLabel, averageTokenPerSecondLabel, averageResponsivenessLabel;
};
