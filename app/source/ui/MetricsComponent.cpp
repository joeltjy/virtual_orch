#include "VirtualOrch/ui/MetricsComponent.h"

MetricsComponent::MetricsComponent(Metrics &metrics) : metrics(metrics), juce::Thread("Metrics") {
    Component::setName("Metrics");
    setOpaque(true);
    setSize(1200, 1200);

    addAndMakeVisible(m_plot);

    m_plot.setTitle("Clock Offset Error");
    m_plot.setYLabel("Offset Error (1/100s)");
    m_plot.setXLabel("Frame");

    m_plot.xLim(0, MAX_SIZE);

    startThread();
}

MetricsComponent::~MetricsComponent() {
}


void MetricsComponent::paint(Graphics &g) {
    g.setColour(juce::Colours::grey);

    g.drawRoundedRectangle(m_plot.getBounds().toFloat(), 5.0f, 5.0f);
}

void MetricsComponent::resized() {
    auto rect = getLocalBounds();
    m_plot.setBounds(rect);
}

void MetricsComponent::userTriedToCloseWindow() {
    delete this;
}

void MetricsComponent::run() {
    while (!threadShouldExit()) {
        if (y[0].size() >= MAX_SIZE) {
            signalThreadShouldExit();
        } else if (int32_t offsetError; metrics.Clock_offsetError.pull(offsetError)) {
            y[0].push_back(offsetError);
            juce::MessageManager::callAsync([this] {
                m_plot.realTimePlot(y);
            });
        }
    }
}
