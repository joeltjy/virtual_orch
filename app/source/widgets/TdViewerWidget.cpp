#include "VirtualOrch/widgets/TdViewerWidget.h"

#include "VirtualOrch/InstrumentUiColours.h"
#include "VirtualOrch/LaunchpadLighting.h"
#include "VirtualOrch/PerformanceActivity.h"
#include "VirtualOrch/TemporalDensity.h"
#include "VirtualOrch/ui/UiConstants.h"

auto TdBarView::setValue(int32_t next) -> void {
    if (next == value)
        return;
    value = next;
    repaint();
}

auto TdBarView::paint(juce::Graphics &g) -> void {
    g.fillAll(UiConstants::pianoRollBackground);
    const auto area = getLocalBounds().toFloat().reduced(10.0f);
    const auto fraction = PerformanceActivity::tdBarFraction(value);

    g.setColour(UiConstants::workspaceWidgetTitleTextColour);
    g.setFont(juce::Font(20.0f, juce::Font::bold));
    const auto header = area.withHeight(30.0f);
    g.drawText(fraction.has_value() ? "TD " + juce::String(value) : juce::String("AUTO"), header,
               juce::Justification::centredLeft);

    const auto bar = area.withTrimmedTop(38.0f).withTrimmedBottom(22.0f);
    const auto steps = static_cast<int>(TemporalDensity::kLevels.size());
    const float gap = 4.0f;
    const float stepW = (bar.getWidth() - gap * static_cast<float>(steps - 1)) / static_cast<float>(steps);
    const int litSteps = fraction.has_value() ? juce::roundToInt(*fraction * static_cast<float>(steps)) : 0;
    for (int i = 0; i < steps; ++i) {
        const auto cell = juce::Rectangle<float>(bar.getX() + static_cast<float>(i) * (stepW + gap),
                                                 bar.getY(), stepW, bar.getHeight());
        g.setColour(i < litSteps ? InstrumentUiColours::fromPalette(
                                       LaunchpadLighting::kTdGradient[static_cast<size_t>(i)])
                                 : juce::Colours::white.withAlpha(0.08f));
        g.fillRoundedRectangle(cell, 3.0f);
        g.setColour(UiConstants::workspaceWidgetTitleTextColour.withAlpha(0.8f));
        g.setFont(juce::Font(12.0f));
        g.drawText(juce::String(TemporalDensity::kLevels[static_cast<size_t>(i)]),
                   cell.withY(bar.getBottom() + 2.0f).withHeight(18.0f),
                   juce::Justification::centred);
    }
}

TdViewerWidget::TdViewerWidget()
    : WorkspaceWidget(UiConstants::workspaceTdViewerTitle, UiConstants::workspaceWidgetTypeTdViewer) {
    getContentComponent().addAndMakeVisible(view);
    startTimerHz(10);
    timerCallback();
}

TdViewerWidget::~TdViewerWidget() {
    stopTimer();
}

auto TdViewerWidget::resized() -> void {
    WorkspaceWidget::resized();
    view.setBounds(getContentComponent().getLocalBounds());
}

auto TdViewerWidget::timerCallback() -> void {
    view.setValue(TemporalDensity::get());
}
