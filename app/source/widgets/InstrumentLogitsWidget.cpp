#include "VirtualOrch/widgets/InstrumentLogitsWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/InstrumentUiColours.h"
#include "VirtualOrch/ui/UiConstants.h"
#include "VirtualOrch/widgets/ActiveInstrumentsView.h"

#include <algorithm>
#include <cmath>

namespace {

class InstrumentLogitsChart : public juce::Component {
public:
    InstrumentLogitSnapshot snapshot;

    auto paint(juce::Graphics &g) -> void override {
        g.fillAll(juce::Colour(0xff1a1a1a));

        auto area = getLocalBounds().toFloat().reduced(8.0f, 4.0f);
        if (area.getWidth() < 8.0f || area.getHeight() < 8.0f)
            return;

        const auto labelH = 28.0f;
        auto chart = area;
        auto labels = chart.removeFromBottom(labelH);
        g.setColour(juce::Colours::grey);
        g.setFont(10.0f);
        const auto subtitle = snapshot.biasApplied ? "post-bias singleton logits"
                                                   : "singleton logits (bias off)";
        g.drawText(subtitle,
                   chart.removeFromTop(14.0f).toNearestIntEdges(),
                   juce::Justification::centredLeft);

        float minL = 0.0f;
        float maxL = 1.0f;
        bool any = false;
        for (size_t i = 0; i < InstrumentLogitSnapshot::kNumInstruments; ++i) {
            if (! snapshot.valid[i])
                continue;
            if (! any) {
                minL = snapshot.logits[i];
                maxL = snapshot.logits[i];
                any = true;
            } else {
                minL = std::min(minL, snapshot.logits[i]);
                maxL = std::max(maxL, snapshot.logits[i]);
            }
        }

        if (! any) {
            g.setColour(juce::Colours::grey);
            g.setFont(12.0f);
            g.drawText("Waiting for orchestration…",
                       chart.toNearestIntEdges(),
                       juce::Justification::centred);
            return;
        }

        if (minL > 0.0f)
            minL = 0.0f;
        if (maxL < 0.0f)
            maxL = 0.0f;
        if (std::abs(maxL - minL) < 1.0e-3f) {
            maxL = minL + 1.0f;
            minL = minL - 1.0f;
        }

        const float zeroY =
            chart.getY() + (maxL - 0.0f) / (maxL - minL) * chart.getHeight();
        g.setColour(juce::Colours::darkgrey);
        g.drawHorizontalLine(juce::roundToInt(zeroY), chart.getX(), chart.getRight());

        const float slotW =
            chart.getWidth() / static_cast<float>(InstrumentLogitSnapshot::kNumInstruments);
        const float barW = std::max(2.0f, slotW * 0.7f);

        for (size_t i = 0; i < InstrumentLogitSnapshot::kNumInstruments; ++i) {
            const float cx = chart.getX() + (static_cast<float>(i) + 0.5f) * slotW;
            auto colour = InstrumentUiColours::forLocalInstrument(static_cast<int32_t>(i));
            if (! snapshot.valid[i]) {
                g.setColour(colour.withAlpha(0.15f));
                g.fillRect(cx - barW * 0.5f, zeroY - 1.0f, barW, 2.0f);
            } else {
                const float y =
                    chart.getY() + (maxL - snapshot.logits[i]) / (maxL - minL) * chart.getHeight();
                const float top = std::min(y, zeroY);
                const float bottom = std::max(y, zeroY);
                g.setColour(colour);
                g.fillRect(cx - barW * 0.5f, top, barW, std::max(1.0f, bottom - top));
            }

            g.setColour(juce::Colours::lightgrey);
            g.setFont(9.0f);
            auto name = ActiveInstrumentsView::shortNameForLocal(static_cast<int32_t>(i));
            if (name.length() > 4)
                name = name.substring(0, 4);
            g.drawText(
                name,
                juce::Rectangle<float>(cx - slotW * 0.5f, labels.getY(), slotW, labels.getHeight())
                    .toNearestIntEdges(),
                juce::Justification::centred);
        }
    }
};

} // namespace

struct InstrumentLogitsWidget::Impl {
    InstrumentLogitsChart chart;
};

InstrumentLogitsWidget::InstrumentLogitsWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceInstrumentLogitsTitle,
                      UiConstants::workspaceWidgetTypeInstrumentLogits),
      session(sessionIn),
      impl(std::make_unique<Impl>()) {
    getContentComponent().addAndMakeVisible(impl->chart);
    startTimer(UiConstants::workspaceModeTimerIntervalMs);
    refreshFromSession();
}

InstrumentLogitsWidget::~InstrumentLogitsWidget() {
    stopTimer();
}

auto InstrumentLogitsWidget::resized() -> void {
    WorkspaceWidget::resized();
    if (impl != nullptr)
        impl->chart.setBounds(getContentComponent().getLocalBounds());
}

auto InstrumentLogitsWidget::timerCallback() -> void {
    refreshFromSession();
}

auto InstrumentLogitsWidget::refreshFromSession() -> void {
    const auto next = session.orchestrationTransformer.getInstrumentLogitSnapshot();
    if (next.sequence == lastSequence)
        return;
    lastSequence = next.sequence;
    if (impl != nullptr) {
        impl->chart.snapshot = next;
        impl->chart.repaint();
    }
}
