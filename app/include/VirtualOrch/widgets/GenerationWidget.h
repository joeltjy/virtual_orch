#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/ui/CustomProgressBarLookAndFeel.h"
#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/** Start / Stop generation, latency progress, and transport clock. */
class GenerationWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit GenerationWidget(AppSession &session);

    ~GenerationWidget() override;

    auto resized() -> void override;

private:
    auto timerCallback() -> void override;

    auto updateTimeDisplay() -> void;

    AppSession &session;
    juce::TextButton startButton{"Start"};
    juce::TextButton stopButton{"Stop"};
    CustomProgressBarLookAndFeel progressBarLookAndFeel;
    std::unique_ptr<juce::ProgressBar> progressBar;
    juce::Label timeLabel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GenerationWidget)
};
