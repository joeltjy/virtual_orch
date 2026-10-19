#include "VirtualOrch/widgets/UserSelectionWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

auto ConsoleView::setLines(std::vector<juce::String> next) -> void {
    if (next == lines)
        return;
    lines = std::move(next);
    repaint();
}

auto ConsoleView::paint(juce::Graphics &g) -> void {
    g.fillAll(juce::Colours::black);
    const auto area = getLocalBounds().toFloat().reduced(10.0f, 6.0f);
    const float lineH = area.getHeight() / static_cast<float>(kLines);
    g.setFont(juce::Font(juce::Font::getDefaultMonospacedFontName(),
                         juce::jlimit(11.0f, 18.0f, lineH * 0.6f), juce::Font::plain));
    // Newest line sits on the bottom row; the older ones fade upwards.
    for (size_t i = 0; i < lines.size(); ++i) {
        const size_t row = kLines - lines.size() + i;
        const float age = static_cast<float>(lines.size() - 1 - i);
        g.setColour(juce::Colour(0xff9be59b).withAlpha(juce::jmax(0.35f, 1.0f - 0.15f * age)));
        g.drawText(lines[i], area.withY(area.getY() + lineH * static_cast<float>(row)).withHeight(lineH),
                   juce::Justification::centredLeft, true);
    }
}

UserSelectionWidget::UserSelectionWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceUserSelectionTitle,
                      UiConstants::workspaceWidgetTypeUserSelection),
      session(sessionIn) {
    getContentComponent().addAndMakeVisible(view);
    startTimerHz(10);
    timerCallback();
}

UserSelectionWidget::~UserSelectionWidget() {
    stopTimer();
}

auto UserSelectionWidget::resized() -> void {
    WorkspaceWidget::resized();
    view.setBounds(getContentComponent().getLocalBounds());
}

auto UserSelectionWidget::timerCallback() -> void {
    view.setLines(session.consoleLog.latest(ConsoleView::kLines));
}
