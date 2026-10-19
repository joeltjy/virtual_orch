#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/widgets/WorkspaceWidget.h"

/** The selected temporal density on a log-spaced bar of 7 steps, or AUTO. */
class TdBarView : public juce::Component {
public:
    auto setValue(int32_t next) -> void;
    auto paint(juce::Graphics &g) -> void override;

private:
    int32_t value = -1;
};

class TdViewerWidget : public WorkspaceWidget, private juce::Timer {
public:
    TdViewerWidget();
    ~TdViewerWidget() override;

    auto resized() -> void override;

private:
    auto timerCallback() -> void override;

    TdBarView view;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TdViewerWidget)
};
