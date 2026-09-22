#pragma once

#include "VirtualOrch/widgets/NoteIoWidget.h"

/** Reduction output coloured by Token::voiceId (vocsep). */
class VocsepOutputWidget : public NoteIoWidget {
public:
    explicit VocsepOutputWidget(AppSession &session);

protected:
    auto refreshFromSession() -> void override;

private:
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(VocsepOutputWidget)
};
