#include "VirtualOrch/widgets/DurationLogitsWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/DurationLogitSnapshot.h"
#include "VirtualOrch/ui/UiConstants.h"

#include <algorithm>
#include <cmath>

namespace {

class DurationLogitsChart : public juce::Component {
public:
    DurationLogitSnapshot snapshot;

    auto paint(juce::Graphics &g) -> void override {
        g.fillAll(juce::Colour(0xff1a1a1a));

        auto area = getLocalBounds().toFloat().reduced(10.0f, 6.0f);
        if (area.getWidth() < 16.0f || area.getHeight() < 16.0f)
            return;

        g.setColour(juce::Colours::grey);
        g.setFont(10.0f);
        juce::String header = "duration logits (cs, post-bias)";
        if (snapshot.sampledCs >= 0)
            header << "  |  sampled " << snapshot.sampledCs << " cs";
        g.drawText(header, area.removeFromTop(14.0f).toNearestIntEdges(),
                   juce::Justification::centredLeft);
        area.removeFromTop(2.0f);

        float minL = 0.0f;
        float maxL = 1.0f;
        bool any = false;
        size_t first = 0;
        size_t last = 0;
        for (size_t i = 0; i < DurationLogitSnapshot::kNumBins; ++i) {
            if (! snapshot.valid[i])
                continue;
            if (! any) {
                minL = maxL = snapshot.logits[i];
                first = last = i;
                any = true;
            } else {
                minL = std::min(minL, snapshot.logits[i]);
                maxL = std::max(maxL, snapshot.logits[i]);
                last = i;
            }
        }

        if (! any || snapshot.sequence == 0) {
            g.setColour(juce::Colours::grey);
            g.setFont(12.0f);
            g.drawText("Waiting for RT duration step…",
                       area.toNearestIntEdges(),
                       juce::Justification::centred);
            return;
        }

        if (std::abs(maxL - minL) < 1.0e-3f) {
            maxL = minL + 1.0f;
            minL = minL - 1.0f;
        }

        const float span = static_cast<float>(std::max<size_t>(1, last - first));
        auto xAt = [&](size_t cs) -> float {
            return area.getX() + static_cast<float>(cs - first) / span * area.getWidth();
        };
        auto yAt = [&](float logit) -> float {
            return area.getY() + (maxL - logit) / (maxL - minL) * area.getHeight();
        };

        // Zero line if in range.
        if (minL <= 0.0f && maxL >= 0.0f) {
            g.setColour(juce::Colours::darkgrey);
            g.drawHorizontalLine(juce::roundToInt(yAt(0.0f)), area.getX(), area.getRight());
        }

        juce::Path path;
        bool started = false;
        for (size_t i = first; i <= last; ++i) {
            if (! snapshot.valid[i])
                continue;
            const float x = xAt(i);
            const float y = yAt(snapshot.logits[i]);
            if (! started) {
                path.startNewSubPath(x, y);
                started = true;
            } else {
                path.lineTo(x, y);
            }
        }

        g.setColour(juce::Colours::cornflowerblue);
        g.strokePath(path, juce::PathStrokeType(1.5f));

        if (snapshot.sampledCs >= 0
            && static_cast<size_t>(snapshot.sampledCs) >= first
            && static_cast<size_t>(snapshot.sampledCs) <= last
            && snapshot.valid[static_cast<size_t>(snapshot.sampledCs)]) {
            const float x = xAt(static_cast<size_t>(snapshot.sampledCs));
            const float y = yAt(snapshot.logits[static_cast<size_t>(snapshot.sampledCs)]);
            g.setColour(juce::Colours::orange);
            g.fillEllipse(x - 3.0f, y - 3.0f, 6.0f, 6.0f);
            g.drawVerticalLine(juce::roundToInt(x), area.getY(), area.getBottom());
        }

        g.setColour(juce::Colours::grey);
        g.setFont(9.0f);
        g.drawText(juce::String(static_cast<int>(first)),
                   juce::Rectangle<float>(area.getX(), area.getBottom() - 12.0f, 40.0f, 12.0f)
                       .toNearestIntEdges(),
                   juce::Justification::centredLeft);
        g.drawText(juce::String(static_cast<int>(last)),
                   juce::Rectangle<float>(area.getRight() - 40.0f, area.getBottom() - 12.0f, 40.0f,
                                          12.0f)
                       .toNearestIntEdges(),
                   juce::Justification::centredRight);
    }
};

} // namespace

struct DurationLogitsWidget::Impl {
    DurationLogitsChart chart;
};

DurationLogitsWidget::DurationLogitsWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceDurationLogitsTitle,
                      UiConstants::workspaceWidgetTypeDurationLogits),
      session(sessionIn),
      impl(std::make_unique<Impl>()) {
    getContentComponent().addAndMakeVisible(impl->chart);
    startTimer(UiConstants::workspaceModeTimerIntervalMs);
    refreshFromSession();
}

DurationLogitsWidget::~DurationLogitsWidget() {
    stopTimer();
}

auto DurationLogitsWidget::resized() -> void {
    WorkspaceWidget::resized();
    if (impl != nullptr)
        impl->chart.setBounds(getContentComponent().getLocalBounds());
}

auto DurationLogitsWidget::timerCallback() -> void {
    refreshFromSession();
}

auto DurationLogitsWidget::refreshFromSession() -> void {
    const auto next = session.activeReduction().getDurationLogitSnapshot();
    if (next.sequence == lastSequence)
        return;
    lastSequence = next.sequence;
    if (impl != nullptr) {
        impl->chart.snapshot = next;
        impl->chart.repaint();
    }
}
