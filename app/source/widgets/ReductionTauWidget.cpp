#include "VirtualOrch/widgets/ReductionTauWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/reduction/DenseDurationBias.h"
#include "VirtualOrch/reduction/DensePitchBias.h"
#include "VirtualOrch/ui/UiConstants.h"

#include <algorithm>

namespace {

/** Fixed axis for the bars (covers base 0.5 through a few seconds of ramp). */
constexpr float kTauAxisMax = 4.0f;

class ReductionTauChart : public juce::Component {
public:
    float pitchTau = DensePitchBias::TauBase;
    float pitchTauPrime = DensePitchBias::TauBase;
    float durationTau = DenseDurationBias::TauBase;
    float durationTauPrime = DenseDurationBias::TauBase;
    bool v2Active = false;

    auto paint(juce::Graphics &g) -> void override {
        g.fillAll(juce::Colour(0xff1a1a1a));

        auto area = getLocalBounds().toFloat().reduced(8.0f, 6.0f);
        if (area.getWidth() < 8.0f || area.getHeight() < 8.0f)
            return;

        g.setColour(juce::Colours::grey);
        g.setFont(10.0f);
        const juce::String subtitle =
            v2Active ? "pitch / duration bias (0 – 4)" : "active only for V2";
        g.drawText(subtitle,
                   area.removeFromTop(14.0f).toNearestIntEdges(),
                   juce::Justification::centredLeft);
        area.removeFromTop(2.0f);

        auto labels = area.removeFromBottom(20.0f);
        auto values = area.removeFromBottom(16.0f);

        const float slotW = area.getWidth() * 0.25f;
        const float barW = std::max(8.0f, slotW * 0.5f);

        g.setColour(juce::Colours::darkgrey);
        g.drawHorizontalLine(juce::roundToInt(area.getY()), area.getX(), area.getRight());

        const auto drawBar = [&](int index, float value, const juce::String &name,
                                 juce::Colour colour) {
            const float display = v2Active ? value : 0.0f;
            const float cx = area.getX() + (static_cast<float>(index) + 0.5f) * slotW;
            const float fill = juce::jlimit(0.0f, 1.0f, display / kTauAxisMax);
            const float barH = fill * area.getHeight();
            const float top = area.getBottom() - barH;

            g.setColour(juce::Colour(0xff2a2a2a));
            g.fillRect(cx - barW * 0.5f, area.getY(), barW, area.getHeight());

            g.setColour(colour);
            g.fillRect(cx - barW * 0.5f, top, barW, std::max(0.0f, barH));

            g.setColour(juce::Colours::lightgrey);
            g.setFont(10.0f);
            g.drawText(juce::String(display, 2),
                       juce::Rectangle<float>(cx - slotW * 0.5f, values.getY(), slotW,
                                              values.getHeight())
                           .toNearestIntEdges(),
                       juce::Justification::centred);
            g.setFont(11.0f);
            g.drawText(name,
                       juce::Rectangle<float>(cx - slotW * 0.5f, labels.getY(), slotW,
                                              labels.getHeight())
                           .toNearestIntEdges(),
                       juce::Justification::centred);
        };

        drawBar(0, pitchTau, juce::CharPointer_UTF8("\xcf\x84"), juce::Colour(0xff6fa8dc));
        drawBar(1, pitchTauPrime, juce::CharPointer_UTF8("\xcf\x84\xe2\x80\xb2"),
                juce::Colour(0xff82c091));
        drawBar(2, durationTau, juce::CharPointer_UTF8("\xcf\x84d"), juce::Colour(0xffe6a057));
        drawBar(3, durationTauPrime, juce::CharPointer_UTF8("\xcf\x84d\xe2\x80\xb2"),
                juce::Colour(0xffc97b84));
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
    float pitchTau = DensePitchBias::TauBase;
    float pitchTauPrime = DensePitchBias::TauBase;
    float durationTau = DenseDurationBias::TauBase;
    float durationTauPrime = DenseDurationBias::TauBase;
    if (v2Active) {
        session.reductionTransformerV2.refreshPitchTauSchedule();
        session.reductionTransformerV2.refreshDurationTauSchedule();
        pitchTau = session.reductionTransformerV2.getPitchTau();
        pitchTauPrime = session.reductionTransformerV2.getPitchTauPrime();
        durationTau = session.reductionTransformerV2.getDurationTau();
        durationTauPrime = session.reductionTransformerV2.getDurationTauPrime();
    }

    if (impl == nullptr)
        return;
    impl->chart.pitchTau = pitchTau;
    impl->chart.pitchTauPrime = pitchTauPrime;
    impl->chart.durationTau = durationTau;
    impl->chart.durationTauPrime = durationTauPrime;
    impl->chart.v2Active = v2Active;
    impl->chart.repaint();
}
