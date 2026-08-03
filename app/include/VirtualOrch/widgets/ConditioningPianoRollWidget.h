#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/widgets/SessionPianoRollWidget.h"

class AppSession;

/** Piano roll of InputFilter pastConditioning. */
class ConditioningPianoRollWidget : public SessionPianoRollWidget {
public:
    explicit ConditioningPianoRollWidget(AppSession &session);

protected:
    auto refreshFromSession() -> void override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ConditioningPianoRollWidget)
};
