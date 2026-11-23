#pragma once

#include <JuceHeader.h>

#include <deque>
#include <string>
#include <vector>

#include "VirtualOrch/reduction/TdController.h"
#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/**
 * Dynamic LogitTD at a glance (TdController), one row per reduction model (W / T3, and v1 for
 * the Hybrid): the target TD, the orchestra's measured TD, the controller's step size K, its
 * gamma and sensitivity S, greyed with the reason while frozen; and a graph of the last
 * kHistorySeconds: target (dashed) and measured TD on a log scale, gamma below.
 */
class TdControlView : public juce::Component {
public:
    static constexpr double kHistorySeconds = 30.0;

    struct Row {
        std::string model;
        TdController::State state;
    };
    struct Sample {
        double seconds = 0.0; // AppTime seconds
        int32_t target = TemporalDensity::kAuto;
        double measured = 0.0;
        double gamma = 0.0;
        bool frozen = true;
    };

    auto setRows(std::vector<Row> rows, double nowSeconds) -> void;
    float textScale = 1.0f;
    auto paint(juce::Graphics &g) -> void override;

private:
    std::vector<Row> rows;
    std::vector<std::pair<std::string, std::deque<Sample>>> history; // per model
    double now = 0.0;
};

class TdControlWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit TdControlWidget(AppSession &session);
    ~TdControlWidget() override;

    auto resized() -> void override;

protected:
    auto textScaleChanged() -> void override { view.textScale = getTextScale(); view.repaint(); }

private:
    auto timerCallback() -> void override;

    AppSession &session;
    TdControlView view;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TdControlWidget)
};
