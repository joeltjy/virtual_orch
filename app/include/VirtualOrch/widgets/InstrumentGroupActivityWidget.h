#pragma once

#include <JuceHeader.h>

#include <array>

#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/** Three bars: strings / woodwinds / brass share of the playback sounding time, last 5 s. */
class GroupBarsView : public juce::Component {
public:
    auto setProportions(const std::array<float, 3> &next) -> void;
    auto paint(juce::Graphics &g) -> void override;

private:
    std::array<float, 3> proportions{};
};

class InstrumentGroupActivityWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit InstrumentGroupActivityWidget(AppSession &session);
    ~InstrumentGroupActivityWidget() override;

    auto resized() -> void override;

private:
    auto timerCallback() -> void override;

    AppSession &session;
    GroupBarsView view;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(InstrumentGroupActivityWidget)
};
