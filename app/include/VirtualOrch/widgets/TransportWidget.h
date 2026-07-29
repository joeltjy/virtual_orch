#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/**
 * Live clock readout from AppSession::clock.
 */
class TransportWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit TransportWidget(AppSession &session);

    ~TransportWidget() override;

    auto resized() -> void override;

private:
    auto timerCallback() -> void override;

    auto updateTimeDisplay() -> void;

    AppSession &session;
    juce::Label timeLabel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TransportWidget)
};
