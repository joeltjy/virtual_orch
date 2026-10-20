#pragma once

#include <JuceHeader.h>

#include <array>
#include <vector>

#include "VirtualOrch/PerformanceActivity.h"
#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/**
 * Draws the orchestra seating plan; sections whose instrument is not sounding are dimmed.
 *
 * Everything is prepared up front so an update costs almost nothing: a dimmed copy of the plan
 * (the background) and, per section, a bright patch cropped to the section's bounding box with
 * the sector mask as its alpha. resized() pre-scales both to the widget; paint draws the
 * background plus one patch per lit section, and setActive repaints only the changed sections.
 */
class SeatingPlanView : public juce::Component {
public:
    SeatingPlanView();

    auto setActive(const std::array<bool, PerformanceActivity::kSectorCount> &active) -> void;
    auto paint(juce::Graphics &g) -> void override;
    auto resized() -> void override;

private:
    static constexpr int kSectors = PerformanceActivity::kSectorCount;

    juce::Image dimBase;                              // plan, sections dimmed (source size)
    std::array<juce::Image, kSectors> brightPatch;    // per section, cropped (source size)
    std::array<juce::Rectangle<int>, kSectors> patchBounds; // in source pixels; empty = none
    int sourceWidth = 0;
    int sourceHeight = 0;

    juce::Rectangle<int> dest;                        // where the plan is drawn
    juce::Image scaledBase;
    std::array<juce::Image, kSectors> scaledPatch;
    std::array<juce::Rectangle<int>, kSectors> scaledBounds;

    std::array<bool, kSectors> active{};
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
