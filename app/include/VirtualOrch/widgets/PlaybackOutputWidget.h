#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/widgets/NoteIoWidget.h"

class AppSession;

class PlaybackOutputWidget : public NoteIoWidget {
public:
    explicit PlaybackOutputWidget(AppSession &session);

protected:
    auto refreshFromSession() -> void override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PlaybackOutputWidget)
};
