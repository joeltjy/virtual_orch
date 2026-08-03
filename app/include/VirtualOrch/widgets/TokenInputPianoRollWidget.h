#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/widgets/SessionPianoRollWidget.h"

class AppSession;

/** Piano roll of InputFilter pastInput. */
class TokenInputPianoRollWidget : public SessionPianoRollWidget {
public:
    explicit TokenInputPianoRollWidget(AppSession &session);

protected:
    auto refreshFromSession() -> void override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TokenInputPianoRollWidget)
};
