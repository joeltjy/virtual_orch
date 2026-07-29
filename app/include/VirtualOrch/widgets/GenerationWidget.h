#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/**
 * Start / Stop generation and progress, bound to AppSession.
 */
class GenerationWidget : public WorkspaceWidget {
public:
    explicit GenerationWidget(AppSession &session);

    ~GenerationWidget() override = default;

    auto resized() -> void override;

private:
    AppSession &session;
    juce::TextButton startButton{"Start"};
    juce::TextButton stopButton{"Stop"};
    std::unique_ptr<juce::ProgressBar> progressBar;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GenerationWidget)
};
