#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/** Edit/Jam mode and Reduction/Orchestration Running|Paused status. */
class ModeWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit ModeWidget(AppSession &session);

    ~ModeWidget() override;

    auto resized() -> void override;

private:
    auto timerCallback() -> void override;

    auto refreshFromSession() -> void;

    AppSession &session;
    juce::TextButton editButton;
    juce::TextButton jamButton;
    juce::Label modeStatusLabel;
    juce::Label reductionLabel;
    juce::TextButton reductionStatusButton;
    juce::Label orchestrationLabel;
    juce::TextButton orchestrationStatusButton;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ModeWidget)
};
