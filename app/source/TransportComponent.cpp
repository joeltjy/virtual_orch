#include "JordanAI/TransportComponent.h"

TransportComponent::TransportComponent(bool onSecondDisplay, const Clock &clock,
                                       const MusicTransformer &musicTransformer,
                                       const ModelConfig &modelConfig) : clock(clock),
                                                                         musicTransformer(musicTransformer),
                                                                         modelConfig(modelConfig) {
    Component::setName("Transport");
    setOpaque(true);
    setSize(300, 150);

    if (onSecondDisplay) {
        scale = 5;
    }

    startTimer(10);
}

TransportComponent::~TransportComponent() {
}

void TransportComponent::paint(Graphics &g) {
    g.setColour(juce::Colours::black);
    g.fillRect(0, 0, getWidth(), getHeight());
    g.setColour(juce::Colours::lightgrey);
    g.setFont(30 * scale);

    // Time
    const juce::Rectangle<int> timeRectangle(10, 0, getWidth() - 20, proportionOfHeight(.3F));
    auto [currentTime, currentBar] = clock.getTimeAndBar();
    auto time = currentTime;
    const auto hours = juce::String(time / (100 * 60 * 60)).paddedLeft('0', 2);
    time %= (100 * 60 * 60);
    const auto minutes = juce::String(time / (100 * 60)).paddedLeft('0', 2);
    time %= (100 * 60);
    const auto seconds = juce::String(time / 100).paddedLeft('0', 2);
    time %= 100;
    const juce::String timeStr = hours + ":" + minutes + ":" + seconds + "." + juce::String(time).paddedLeft('0', 2);
    g.drawFittedText(timeStr, timeRectangle, juce::Justification::centred, 1);

    // Bar
    const juce::Rectangle<int> barRectangle(10, proportionOfHeight(.3F), getWidth() - 20, proportionOfHeight(.3F));
    g.drawFittedText(juce::String(currentBar), barRectangle, juce::Justification::centred, 1);

    // Trading Info
    const juce::Rectangle<int> tradingRectangle(10, proportionOfHeight(.6F), getWidth() - 20, proportionOfHeight(.3F));
    juce::String tradingInfo = "0.0";
    auto tradingOffset = musicTransformer.tradingOffset.get();
    tradingInfo = juce::String(((currentBar - tradingOffset) % modelConfig.tradingInputNumberOfBars) + 1)
                  + "."
                  + juce::String(((currentTime / (modelConfig.outputBarLength / 4)) % 4) + 1);
    // TODO: Make beatsPerBar a parameter
    g.drawFittedText(tradingInfo, tradingRectangle, juce::Justification::centred, 1);
}

void TransportComponent::resized() {
}


void TransportComponent::userTriedToCloseWindow() {
    delete this;
}

void TransportComponent::timerCallback() {
    repaint();
}


