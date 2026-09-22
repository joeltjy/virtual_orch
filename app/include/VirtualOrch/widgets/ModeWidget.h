#pragma once

#include <JuceHeader.h>

#include <cstdint>

#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/** Edit/Jam mode, Reduction/Orchestration pause, loop ms, step timings, and sampling alerts. */
class ModeWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit ModeWidget(AppSession &session);

    ~ModeWidget() override;

    auto resized() -> void override;

    auto mouseDown(const juce::MouseEvent &event) -> void override;

private:
    auto timerCallback() -> void override;

    auto refreshFromSession() -> void;

    AppSession &session;
    juce::TextButton editButton;
    juce::TextButton jamButton;
    juce::TextButton playbackOtButton;
    juce::TextButton playbackRtButton;
    juce::Label modeStatusLabel;
    juce::Label reductionLabel;
    juce::TextButton reductionStatusButton;
    juce::Label reductionLoopMsLabel;
    juce::Label reductionProfileLabel;
    juce::Label orchestrationLabel;
    juce::TextButton orchestrationStatusButton;
    juce::Label orchestrationLoopMsLabel;
    juce::Label orchestrationProfileLabel;
    juce::Label vocsepLabel;
    juce::Label vocsepLoopMsLabel;
    juce::Label vocsepProfileLabel;
    juce::Label samplingAlertLabel;
    uint64_t lastAlertSequence = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ModeWidget)
};
