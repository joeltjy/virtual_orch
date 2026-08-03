#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/widgets/SessionPianoRollWidget.h"

class AppSession;

class OrchestrationMidiPianoRollWidget : public SessionPianoRollWidget {
public:
    explicit OrchestrationMidiPianoRollWidget(AppSession &session);

protected:
    auto refreshFromSession() -> void override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(OrchestrationMidiPianoRollWidget)
};
