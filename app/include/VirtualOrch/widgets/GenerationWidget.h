#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/ui/CustomProgressBarLookAndFeel.h"
#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/** Start / Stop generation and latency progress bar (seconds), bound to AppSession. */
class GenerationWidget : public WorkspaceWidget {
public:
    explicit GenerationWidget(AppSession &session);

    ~GenerationWidget() override;

    auto resized() -> void override;

private:
    AppSession &session;
    juce::TextButton startButton{"Start"};
    juce::TextButton stopButton{"Stop"};
    CustomProgressBarLookAndFeel progressBarLookAndFeel;
    std::unique_ptr<juce::ProgressBar> progressBar;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GenerationWidget)
};
