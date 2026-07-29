#pragma once

#include <JuceHeader.h>

#include <cmp_plot.h>

#include "Metrics.h"

class MetricsComponent : public juce::Component, juce::Thread {
public:
    MetricsComponent(Metrics &metrics);

    ~MetricsComponent() override;

    void paint(Graphics &g) override;

    void resized() override;

    void userTriedToCloseWindow() override;

    void run() override;

private:
    cmp::Plot m_plot;
    std::vector<std::vector<float> > y{{}};

    uint32_t MAX_SIZE = 500;
    uint32_t FRAME_RATE = 30;

    Metrics &metrics;
};
