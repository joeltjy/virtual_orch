#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/widgets/OrchestrationOutputVisualizerView.h"
#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

class OrchestrationOutputVisualizerWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit OrchestrationOutputVisualizerWidget(AppSession &session);

    ~OrchestrationOutputVisualizerWidget() override;

    auto resized() -> void override;

private:
    auto timerCallback() -> void override;

    auto refreshFromSession() -> void;

    AppSession &session;
    OrchestrationOutputVisualizerView table;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(OrchestrationOutputVisualizerWidget)
};
