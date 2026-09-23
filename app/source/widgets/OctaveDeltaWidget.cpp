#include "VirtualOrch/widgets/OctaveDeltaWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/OctaveDeltaTracker.h"
#include "VirtualOrch/ui/UiConstants.h"

#include <algorithm>

namespace {

class OctaveDeltaChart : public juce::Component {
public:
    OctaveDeltaSnapshot snapshot;

    auto paint(juce::Graphics &g) -> void override {
        g.fillAll(juce::Colour(0xff1a1a1a));

        auto area = getLocalBounds().toFloat().reduced(8.0f, 6.0f);
        if (area.getWidth() < 8.0f || area.getHeight() < 8.0f)
            return;

        g.setColour(juce::Colours::grey);
        g.setFont(10.0f);
        juce::String header = "IOD octave Δ";
        if (snapshot.total > 0)
            header << "  |  n=" << juce::String(static_cast<juce::int64>(snapshot.total));
        g.drawText(header, area.removeFromTop(14.0f).toNearestIntEdges(),
                   juce::Justification::centredLeft);
        area.removeFromTop(2.0f);

        if (snapshot.total == 0) {
            g.setColour(juce::Colours::grey);
            g.setFont(12.0f);
            g.drawText("Waiting for IOD decode…",
                       area.toNearestIntEdges(),
                       juce::Justification::centred);
            return;
        }

        auto labels = area.removeFromBottom(18.0f);
        auto values = area.removeFromBottom(16.0f);

        const float slotW = area.getWidth() / 3.0f;
        const float barW = std::max(10.0f, slotW * 0.45f);
        const uint64_t maxCount =
            std::max({snapshot.counts[0], snapshot.counts[1], snapshot.counts[2], uint64_t{1}});

        static constexpr const char *kNames[] = {"−1", "0", "+1"};
        static constexpr juce::uint32 kColours[] = {0xffe6a057, 0xff6fa8dc, 0xff82c091};

        for (int i = 0; i < 3; ++i) {
            const float cx = area.getX() + (static_cast<float>(i) + 0.5f) * slotW;
            const float fill =
                static_cast<float>(snapshot.counts[static_cast<size_t>(i)])
                / static_cast<float>(maxCount);
            const float barH = fill * area.getHeight();
            const float top = area.getBottom() - barH;

            g.setColour(juce::Colour(0xff2a2a2a));
            g.fillRect(cx - barW * 0.5f, area.getY(), barW, area.getHeight());

            g.setColour(juce::Colour(kColours[static_cast<size_t>(i)]));
            g.fillRect(cx - barW * 0.5f, top, barW, std::max(0.0f, barH));

            g.setColour(juce::Colours::lightgrey);
            g.setFont(11.0f);
            g.drawText(juce::String(static_cast<juce::int64>(snapshot.counts[static_cast<size_t>(i)])),
                       juce::Rectangle<float>(cx - slotW * 0.5f, values.getY(), slotW,
                                              values.getHeight())
                           .toNearestIntEdges(),
                       juce::Justification::centred);
            g.setFont(12.0f);
            g.drawText(kNames[static_cast<size_t>(i)],
                       juce::Rectangle<float>(cx - slotW * 0.5f, labels.getY(), slotW,
                                              labels.getHeight())
                           .toNearestIntEdges(),
                       juce::Justification::centred);
        }
    }
};

} // namespace

struct OctaveDeltaWidget::Impl {
    OctaveDeltaChart chart;
};

OctaveDeltaWidget::OctaveDeltaWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceOctaveDeltaTitle,
                      UiConstants::workspaceWidgetTypeOctaveDelta),
      session(sessionIn),
      impl(std::make_unique<Impl>()) {
    getContentComponent().addAndMakeVisible(impl->chart);
    startTimer(UiConstants::workspaceModeTimerIntervalMs);
    refreshFromSession();
}

OctaveDeltaWidget::~OctaveDeltaWidget() {
    stopTimer();
}

auto OctaveDeltaWidget::resized() -> void {
    WorkspaceWidget::resized();
    if (impl != nullptr)
        impl->chart.setBounds(getContentComponent().getLocalBounds());
}

auto OctaveDeltaWidget::timerCallback() -> void {
    refreshFromSession();
}

auto OctaveDeltaWidget::refreshFromSession() -> void {
    const auto next = session.orchestrationTransformer.getOctaveDeltaSnapshot();
    if (next.sequence == lastSequence)
        return;
    lastSequence = next.sequence;
    if (impl != nullptr) {
        impl->chart.snapshot = next;
        impl->chart.repaint();
    }
}
