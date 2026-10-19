#pragma once

#include <JuceHeader.h>

#include <array>
#include <vector>

#include "VirtualOrch/PerformanceActivity.h"
#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/** Draws the orchestra seating plan; sections whose instrument is not sounding are dimmed. */
class SeatingPlanView : public juce::Component {
public:
    SeatingPlanView();

    auto setActive(const std::array<bool, PerformanceActivity::kSectorCount> &active) -> void;
    auto paint(juce::Graphics &g) -> void override;

private:
    auto rebuildOverlay() -> void;

    juce::Image plan;
    int maskWidth = 0;
    int maskHeight = 0;
    std::vector<uint8_t> sectorOfPixel; // 0 = none, else sector + 1
    juce::Image overlay;                // black, translucent over the inactive sectors
    std::array<bool, PerformanceActivity::kSectorCount> active{};
};

class InstrumentActiveLightWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit InstrumentActiveLightWidget(AppSession &session);
    ~InstrumentActiveLightWidget() override;

    auto resized() -> void override;

private:
    auto timerCallback() -> void override;

    AppSession &session;
    SeatingPlanView view;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(InstrumentActiveLightWidget)
};
