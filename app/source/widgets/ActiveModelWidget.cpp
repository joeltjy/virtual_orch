#include "VirtualOrch/widgets/ActiveModelWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

namespace {

const juce::Colour kV1Colour{0xff3498db};   // T3'' v1
const juce::Colour kWaitColour{0xffe67e22}; // W2/W3

auto reasonText(TwoInstrumentAnticipatedReductionTransformer::HybridReason r) -> juce::String {
    using R = TwoInstrumentAnticipatedReductionTransformer::HybridReason;
    switch (r) {
        case R::PedalReleased: return "pedal released";
        case R::Playing: return "you have been playing for 1 s";
        case R::Stopped: return "you stopped for 0.5 s (its future notes were deleted)";
        case R::PredictedSilence: return "W2/W3 predicted 2 s of silence (its future was deleted)";
        case R::Start: break;
    }
    return "start";
}

} // namespace

ActiveModelWidget::ActiveModelWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceActiveModelTitle, UiConstants::workspaceWidgetTypeActiveModel),
      session(sessionIn) {
    getContentComponent().addAndMakeVisible(view);
    startTimerHz(15);
}

ActiveModelWidget::~ActiveModelWidget() {
    stopTimer();
}

auto ActiveModelWidget::resized() -> void {
    WorkspaceWidget::resized();
    view.setBounds(getContentComponent().getLocalBounds());
}

auto ActiveModelWidget::timerCallback() -> void {
    view.repaint();
}

auto ActiveModelWidget::View::paint(juce::Graphics &g) -> void {
    g.fillAll(UiConstants::pianoRollBackground);
    auto area = getLocalBounds().toFloat().reduced(10.0f);
    const auto text = UiConstants::workspaceWidgetTitleTextColour;

    if (session.musicModelArch == MusicModelArch::DenseHybridInformativeWait) {
        paintHybrid(g, area, session.reductionTransformerHybridIW, "m2  informative WAIT");
        return;
    }
    if (session.musicModelArch == MusicModelArch::DenseHybridChordToken) {
        paintHybrid(g, area, session.reductionTransformerHybridChord, "m2  chord-token WAIT");
        return;
    }
    if (session.musicModelArch != MusicModelArch::DenseHybrid) {
        g.setColour(text);
        g.setFont(juce::Font(18.0f, juce::Font::bold));
        g.drawFittedText(session.activeReduction().getThreadName(), area.toNearestInt(),
                         juce::Justification::centredLeft, 2);
        return;
    }

    const auto &hybrid = session.reductionTransformerHybrid;
    const bool v1 = hybrid.hybridUsesV1();
    const auto colour = v1 ? kV1Colour : kWaitColour;

    auto badge = area.removeFromTop(juce::jmin(46.0f, area.getHeight() * 0.45f));
    g.setColour(colour);
    g.fillRoundedRectangle(badge, 6.0f);
    g.setColour(juce::Colours::black);
    g.setFont(juce::Font(20.0f, juce::Font::bold));
    g.drawFittedText(juce::String(v1 ? "T3'' v2 (v1)" : "WAIT model") + "   "
                         + juce::String(v1 ? hybrid.hybridV1Name : hybrid.hybridWaitName),
                     badge.reduced(8.0f, 0.0f).toNearestInt(), juce::Justification::centredLeft, 1);

    area.removeFromTop(6.0f);
    g.setColour(text);
    g.setFont(juce::Font(13.0f));
    g.drawFittedText("why: " + reasonText(hybrid.hybridReason()),
                     area.removeFromTop(18.0f).toNearestInt(), juce::Justification::centredLeft, 1);

    // Progress towards the 1 s switch to W2/W3 (only while v1 draws and you are playing).
    const auto playingFor = hybrid.hybridPlayingForMs();
    area.removeFromTop(4.0f);
    auto bar = area.removeFromTop(12.0f);
    g.setColour(juce::Colours::white.withAlpha(0.1f));
    g.fillRoundedRectangle(bar, 3.0f);
    if (v1 && playingFor >= 0) {
        const float f = juce::jlimit(0.0f, 1.0f, static_cast<float>(playingFor)
                                                      / static_cast<float>(TwoInstrumentAnticipatedReductionTransformer::kHybridPlayToWaitMs));
        g.setColour(kWaitColour.withAlpha(0.8f));
        g.fillRoundedRectangle(bar.withWidth(bar.getWidth() * f), 3.0f);
    }
    g.setColour(text.withAlpha(0.8f));
    g.setFont(juce::Font(12.0f));
    const juce::String status = session.activeReduction().generationPause.get()
                                    ? "paused (orchestra waits for the pedal)"
                                    : playingFor >= 0 ? "playing " + juce::String(playingFor / 1000.0, 1) + " s"
                                                      : juce::String("not playing");
    g.drawText(status, area.removeFromTop(16.0f), juce::Justification::centredLeft);
}

template<typename Hybrid>
auto ActiveModelWidget::View::paintHybrid(juce::Graphics &g, juce::Rectangle<float> area, const Hybrid &hybrid,
                                          const char *m2Title) -> void {
    // Hybrid v1.1 / v1.2: model 1 (continuation) while the pianist is silent, model 2 (WAIT) while they play.
    using Reason = typename Hybrid::Reason;
    const bool m1 = hybrid.usesSilentModel();
    const auto colour = m1 ? kV1Colour : kWaitColour;
    const auto text = UiConstants::workspaceWidgetTitleTextColour;

    auto badge = area.removeFromTop(juce::jmin(46.0f, area.getHeight() * 0.45f));
    g.setColour(colour);
    g.fillRoundedRectangle(badge, 6.0f);
    g.setColour(juce::Colours::black);
    g.setFont(juce::Font(20.0f, juce::Font::bold));
    g.drawFittedText(juce::String(m1 ? "m1  continuation" : m2Title) + "   "
                         + juce::String(m1 ? hybrid.silentName : hybrid.waitName),
                     badge.reduced(8.0f, 0.0f).toNearestInt(), juce::Justification::centredLeft, 1, 0.7f);

    juce::String why = "start";
    switch (hybrid.reason()) {
        case Reason::PedalReleased: why = "pedal released"; break;
        case Reason::Playing: why = "you have been playing for 1 s"; break;
        case Reason::Stopped: why = "you stopped for 0.5 s (its future notes were deleted)"; break;
        case Reason::PredictedSilence: why = "m2 predicted 2 s of silence (its future was deleted)"; break;
        case Reason::Accompaniment: why = "Accompaniment: the pianist is ignored"; break;
        case Reason::Start: break;
    }
    area.removeFromTop(6.0f);
    g.setColour(text);
    g.setFont(juce::Font(13.0f));
    g.drawFittedText("why: " + why, area.removeFromTop(18.0f).toNearestInt(), juce::Justification::centredLeft, 1);

    // Progress towards the 1 s switch to m2 (only while m1 draws and you are playing).
    const auto playingFor = hybrid.playingForMs();
    area.removeFromTop(4.0f);
    auto bar = area.removeFromTop(12.0f);
    g.setColour(juce::Colours::white.withAlpha(0.1f));
    g.fillRoundedRectangle(bar, 3.0f);
    if (m1 && playingFor >= 0) {
        const float f = juce::jlimit(0.0f, 1.0f, static_cast<float>(playingFor)
                                                      / static_cast<float>(Hybrid::kPlayToWaitMs));
        g.setColour(kWaitColour.withAlpha(0.8f));
        g.fillRoundedRectangle(bar.withWidth(bar.getWidth() * f), 3.0f);
    }
    g.setColour(text.withAlpha(0.8f));
    g.setFont(juce::Font(12.0f));
    const juce::String status = hybrid.generationPause.get() ? "paused (orchestra waits for the pedal)"
                                : playingFor >= 0         ? "playing " + juce::String(playingFor / 1000.0, 1) + " s"
                                                          : juce::String("not playing");
    g.drawText(status, area.removeFromTop(16.0f), juce::Justification::centredLeft);
}
