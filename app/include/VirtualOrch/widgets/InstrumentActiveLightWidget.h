#pragma once

#include <JuceHeader.h>

#include <array>
#include <vector>

#include "VirtualOrch/PerformanceActivity.h"
#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/** Draws the orchestra seating plan; sections whose instrument is not sounding are dimmed. */
class SeatingPlanView : public juce::Component, private juce::Timer {
public:
    SeatingPlanView();
    ~SeatingPlanView() override;

    auto setActive(const std::array<bool, PerformanceActivity::kSectorCount> &active) -> void;
    auto paint(juce::Graphics &g) -> void override;

    /**
     * Frozen = the orchestra is stopped because the pedal is down: the plan is blurred and a
     * "Pianist soloing" caption fades in over it. `immediate` skips the fade (tests).
     */
    auto setFrozen(bool frozen, bool immediate = false) -> void;
    [[nodiscard]] auto isFrozen() const -> bool { return freezeTarget; }
    [[nodiscard]] auto hasPrecomputedBlur() const -> bool { return blurred.isValid(); }
    [[nodiscard]] auto freezeOpacity() const -> float { return freezeAlpha; }

private:
    auto timerCallback() -> void override;
    auto precomputeBlur() -> void;
    auto rebuildOverlay() -> void;

    juce::Image plan;
    int maskWidth = 0;
    int maskHeight = 0;
    std::vector<uint8_t> sectorOfPixel; // 0 = none, else sector + 1
    juce::Image overlay;                // black, translucent over the inactive sectors
    std::array<bool, PerformanceActivity::kSectorCount> active{};

    bool freezeTarget = false;
    float freezeAlpha = 0.0f;
    /** The plain plan, blurred once when the view is created (quarter size); drawn when frozen. */
    juce::Image blurred;
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
