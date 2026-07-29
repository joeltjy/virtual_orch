#pragma once

#include <JuceHeader.h>

#include "Clock.h"
#include "MusicTransformer.h"

class TransportComponent : public juce::Component, juce::Timer {
public:
    TransportComponent(bool onSecondDisplay, const Clock &clock, const MusicTransformer &musicTransformer,
                       const ModelConfig &modelConfig);

    ~TransportComponent() override;

    void paint(Graphics &g) override;

    void resized() override;

    void userTriedToCloseWindow() override;

    void timerCallback() override;

private:
    const Clock &clock;
    const MusicTransformer &musicTransformer;
    const ModelConfig &modelConfig;

    float scale = 1;
};
