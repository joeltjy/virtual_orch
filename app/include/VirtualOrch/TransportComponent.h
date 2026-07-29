#pragma once

#include <JuceHeader.h>

#include "Clock.h"

class TransportComponent : public juce::Component, juce::Timer {
public:
    TransportComponent(bool onSecondDisplay, const Clock &clock);

    ~TransportComponent() override;

    void paint(Graphics &g) override;

    void resized() override;

    void userTriedToCloseWindow() override;

    void timerCallback() override;

private:
    const Clock &clock;

    float scale = 1;
};
