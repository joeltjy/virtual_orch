#include "VirtualOrch/widgets/ReductionTauWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/reduction/DensePitchBias.h"
#include "VirtualOrch/ui/UiConstants.h"

#include <algorithm>

namespace {

/** Fixed axis for the bars (covers base 0.5 through the documented 5 s ramp peak). */
constexpr float kTauAxisMax = 4.0f;

class ReductionTauChart : public juce::Component {
public:
    float tau = DensePitchBias::TauBase;
    float tauPrime = DensePitchBias::TauBase;
    bool v2Active = false;

    auto paint(juce::Graphics &g) -> void override {
        g.fillAll(juce::Colour(0xff1a1a1a));

        auto area = getLocalBounds().toFloat().reduced(10.0f, 8.0f);
        if (area.getWidth() < 8.0f || area.getHeight() < 8.0f)
            return;

        g.setColour(juce::Colours::grey);
        g.setFont(11.0f);
        const juce::String subtitle =
            v2Active ? "pitch bias strength (0 – 4)" : "active only for V2 (smart sampling)";
        g.drawText(subtitle,
                   area.removeFromTop(16.0f).toNearestIntEdges(),
                   juce::Justification::centredLeft);
        area.removeFromTop(4.0f);

        auto labels = area.removeFromBottom(22.0f);
        auto values = area.removeFromBottom(18.0f);

        const float slotW = area.getWidth() * 0.5f;
        const float barW = std::max(12.0f, slotW * 0.45f);
        const float displayTau = v2Active ? tau : 0.0f;
        const float displayTauPrime = v2Active ? tauPrime : 0.0f;

        g.setColour(juce::Colours::darkgrey);
        g.drawHorizontalLine(juce::roundToInt(area.getY()), area.getX(), area.getRight());
        g.setFont(9.0f);
        g.drawText("4",
                   juce::Rectangle<float>(area.getX(), area.getY(), 16.0f, 12.0f).toNearestIntEdges(),
                   juce::Justification::centredLeft);

        const auto drawBar = [&](int index, float value, const juce::String &name,
                                 juce::Colour colour) {
            const float cx = area.getX() + (static_cast<float>(index) + 0.5f) * slotW;
            const float fill = juce::jlimit(0.0f, 1.0f, value / kTauAxisMax);
            const float barH = fill * area.getHeight();
            const float top = area.getBottom() - barH;

            g.setColour(juce::Colour(0xff2a2a2a));
            g.fillRect(cx - barW * 0.5f, area.getY(), barW, area.getHeight());

            g.setColour(colour);
            g.fillRect(cx - barW * 0.5f, top, barW, std::max(0.0f, barH));

            g.setColour(juce::Colours::lightgrey);
            g.setFont(12.0f);
            g.drawText(juce::String(value, 2),
                       juce::Rectangle<float>(cx - slotW * 0.5f, values.getY(), slotW,
                                              values.getHeight())
                           .toNearestIntEdges(),
                       juce::Justification::centred);
            g.setFont(13.0f);
            g.drawText(name,
                       juce::Rectangle<float>(cx - slotW * 0.5f, labels.getY(), slotW,
                                              labels.getHeight())
                           .toNearestIntEdges(),
                       juce::Justification::centred);
        };

        drawBar(0, displayTau, juce::CharPointer_UTF8("\xcf\x84"), juce::Colour(0xff6fa8dc));
        drawBar(1, displayTauPrime, juce::CharPointer_UTF8("\xcf\x84\xe2\x80\xb2"),
                juce::Colour(0xff82c091));
    }
};

} // namespace

struct ReductionTauWidget::Impl {
    ReductionTauChart chart;
};

ReductionTauWidget::ReductionTauWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceReductionTauTitle,
                      UiConstants::workspaceWidgetTypeReductionTau),
      session(sessionIn),
      impl(std::make_unique<Impl>()) {
    getContentComponent().addAndMakeVisible(impl->chart);
    startTimer(UiConstants::workspaceModeTimerIntervalMs);
    refreshFromSession();
}

ReductionTauWidget::~ReductionTauWidget() {
    stopTimer();
}

auto ReductionTauWidget::resized() -> void {
    WorkspaceWidget::resized();
    if (impl != nullptr)
        impl->chart.setBounds(getContentComponent().getLocalBounds());
}

auto ReductionTauWidget::timerCallback() -> void {
    refreshFromSession();
}

auto ReductionTauWidget::refreshFromSession() -> void {
    const bool v2Active = session.musicModelArch == MusicModelArch::DenseV2;
    float tau = DensePitchBias::TauBase;
    float tauPrime = DensePitchBias::TauBase;
    if (v2Active) {
        session.reductionTransformerV2.refreshPitchTauSchedule();
        tau = session.reductionTransformerV2.getPitchTau();
        tauPrime = session.reductionTransformerV2.getPitchTauPrime();
    }

    if (impl == nullptr)
        return;
    impl->chart.tau = tau;
    impl->chart.tauPrime = tauPrime;
    impl->chart.v2Active = v2Active;
    impl->chart.repaint();
}
