#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/widgets/PianoRollView.h"
#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/**
 * Shared workspace piano-roll chrome: timer refresh from AppSession.
 * Subclasses implement refreshFromSession() only.
 */
class SessionPianoRollWidget : public WorkspaceWidget, private juce::Timer {
public:
    SessionPianoRollWidget(AppSession &session, juce::String titleText, juce::String widgetType);

    ~SessionPianoRollWidget() override;

    auto resized() -> void override;

protected:
    virtual auto refreshFromSession() -> void = 0;

    AppSession &session;
    PianoRollView pianoRoll;

private:
    auto timerCallback() -> void override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SessionPianoRollWidget)
};
