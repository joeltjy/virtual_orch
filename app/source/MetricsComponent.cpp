#include "JordanAI/MetricsComponent.h"

MetricsComponent::MetricsComponent(Metrics &metrics) : metrics(metrics), juce::Thread("Metrics") {
    Component::setName("Metrics");
    setOpaque(true);
    setSize(600, 300);

    addAndMakeVisible(averageTokenLatencyLabel);
    addAndMakeVisible(averageTokenPerSecondLabel);
    addAndMakeVisible(averageResponsivenessLabel);

    startThread();
}

MetricsComponent::~MetricsComponent() {
}


void MetricsComponent::paint(Graphics &g) {
    g.setColour(juce::Colours::grey);
}

void MetricsComponent::resized() {
    auto rect = getLocalBounds();

    averageTokenLatencyLabel.setBounds(rect.removeFromTop(30).reduced(10));
    averageTokenPerSecondLabel.setBounds(rect.removeFromTop(30).reduced(10));
    averageResponsivenessLabel.setBounds(rect.removeFromTop(30).reduced(10));
}

void MetricsComponent::userTriedToCloseWindow() {
    delete this;
}

void MetricsComponent::run() {
    while (!threadShouldExit()) {
        // tokenLatency
        float tokenLatency;
        while (tokenLatencyFifo.pull(tokenLatency)) {
            tokenLatencyBuffer.push_back(tokenLatency);
            runningTokenLatencySum += tokenLatency;

            if (tokenLatencyBuffer.size() > maxSize) {
                runningTokenLatencySum -= tokenLatencyBuffer.front();
                tokenLatencyBuffer.pop_front();
            }
        }

        // responsiveness
        float responsiveness;
        while (responsivenessFifo.pull(responsiveness)) {
            responsivenessBuffer.push_back(responsiveness);
            runningResponsivenessSum += responsiveness;

            if (responsivenessBuffer.size() > maxSize) {
                runningResponsivenessSum -= responsivenessBuffer.front();
                responsivenessBuffer.pop_front();
            }
        }

        // Update Labels
        float averageTokenLatency = 0.0f;
        if (!tokenLatencyBuffer.empty()) {
            averageTokenLatency = runningTokenLatencySum / static_cast<float>(tokenLatencyBuffer.size());
        }
        float averageTokensPerSecond = 0.0f;
        if (averageTokenLatency > 0.0f) {
            averageTokensPerSecond = 1000.0f / averageTokenLatency;
        }
        float averageResponsiveness = 0.0f;
        if (!responsivenessBuffer.empty()) {
            averageResponsiveness = runningResponsivenessSum / static_cast<float>(responsivenessBuffer.size());
        }
        averageTokenLatencyLabel.setText("Avg Token Latency: " + juce::String(averageTokenLatency, 2) + " ms",
                                         juce::dontSendNotification);
        averageTokenPerSecondLabel.setText(
            "Avg Tokens/Sec: " + juce::String(averageTokensPerSecond, 2) + " tokens/s",
            juce::dontSendNotification);
        averageResponsivenessLabel.setText("Avg Responsiveness: " + juce::String(averageResponsiveness, 2) + " ms",
                                           juce::dontSendNotification);
    }
}
