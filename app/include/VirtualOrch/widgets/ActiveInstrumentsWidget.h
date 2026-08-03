#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/widgets/ActiveInstrumentsView.h"
#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

class ActiveInstrumentsWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit ActiveInstrumentsWidget(AppSession &session);

    ~ActiveInstrumentsWidget() override;

    auto resized() -> void override;

private:
    auto timerCallback() -> void override;

    auto refreshFromSession() -> void;

    AppSession &session;
    ActiveInstrumentsView table;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ActiveInstrumentsWidget)
};
