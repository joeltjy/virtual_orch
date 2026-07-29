#include "VirtualOrch/TransportComponent.h"

TransportComponent::TransportComponent(bool onSecondDisplay, const Clock &clock) : clock(clock) {
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
    const juce::Rectangle<int> timeRectangle(10, 0, getWidth() - 20, getHeight());
    auto currentTime = clock.getTime();
    auto time = currentTime;
    const auto hours = juce::String(time / (100 * 60 * 60)).paddedLeft('0', 2);
    time %= (100 * 60 * 60);
    const auto minutes = juce::String(time / (100 * 60)).paddedLeft('0', 2);
    time %= (100 * 60);
    const auto seconds = juce::String(time / 100).paddedLeft('0', 2);
    time %= 100;
    const juce::String timeStr = hours + ":" + minutes + ":" + seconds + "." + juce::String(time).paddedLeft('0', 2);
    g.drawFittedText(timeStr, timeRectangle, juce::Justification::centred, 1);
}

void TransportComponent::resized() {
}


void TransportComponent::userTriedToCloseWindow() {
    delete this;
}

void TransportComponent::timerCallback() {
    repaint();
}
