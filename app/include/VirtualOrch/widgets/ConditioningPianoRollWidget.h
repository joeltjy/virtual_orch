#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/widgets/NoteIoWidget.h"

class AppSession;

class ConditioningPianoRollWidget : public NoteIoWidget {
public:
    explicit ConditioningPianoRollWidget(AppSession &session);

protected:
    auto refreshFromSession() -> void override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ConditioningPianoRollWidget)
};
