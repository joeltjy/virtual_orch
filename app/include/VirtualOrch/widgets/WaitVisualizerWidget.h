#pragma once

#include <JuceHeader.h>

#include <string>

#include "VirtualOrch/reduction/WaitTracker.h"
#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/**
 * The WAIT Visualizer, drawn from a WaitTracker::Snapshot.
 *
 *  - Left half, the pending WAIT: a line from NOW to PREDICTED with the hit window (+-6 cs)
 *    shaded around PREDICTED. When a WAIT is drawn a circle with its COUNT starts at NOW and
 *    moves at a constant speed so that it reaches PREDICTED at the predicted time. Empty when
 *    nothing is pending.
 *  - Right half, the result of the last window: #1..#5 (the rank of the pianist's note among
 *    the anticipated pitches), >5, HIT (no anticipation was ready) or EXPIRED, with the last
 *    few results below.
 */
class WaitVisualizerView : public juce::Component {
public:
    /** `nowCs`: the app clock; `source`: which model draws the WAITs ("" = none). */
    auto setState(WaitTracker::Snapshot next, int32_t nowCs, std::string source) -> void;
    auto paint(juce::Graphics &g) -> void override;

    float textScale = 1.0f;

    /** The text a result is shown as: "#2", ">5", "HIT", "EXPIRED". */
    [[nodiscard]] static auto resultText(const WaitTracker::Result &r) -> juce::String;

private:
    WaitTracker::Snapshot state;
    int32_t now = 0;
    std::string sourceName;
};

class WaitVisualizerWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit WaitVisualizerWidget(AppSession &session);
    ~WaitVisualizerWidget() override;

    auto resized() -> void override;

protected:
    auto textScaleChanged() -> void override {
        view.textScale = getTextScale();
        view.repaint();
    }

private:
    auto timerCallback() -> void override;

    AppSession &session;
    WaitVisualizerView view;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(WaitVisualizerWidget)
};
