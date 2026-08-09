#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/widgets/NoteIoWidget.h"

class AppSession;

/** Tokens emitted by the active ReductionTransformer (AMT or Dense). */
class ReductionTransformerOutputWidget : public NoteIoWidget {
public:
    explicit ReductionTransformerOutputWidget(AppSession &session);

protected:
    auto refreshFromSession() -> void override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ReductionTransformerOutputWidget)
};
