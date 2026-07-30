#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/widgets/PianoRollView.h"
#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/** Piano roll of InputFilter pastTokens. */
class TokenInputPianoRollWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit TokenInputPianoRollWidget(AppSession &session);

    ~TokenInputPianoRollWidget() override;

    auto resized() -> void override;

private:
    auto timerCallback() -> void override;

    auto refreshFromSession() -> void;

    AppSession &session;
    PianoRollView pianoRoll;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TokenInputPianoRollWidget)
};
