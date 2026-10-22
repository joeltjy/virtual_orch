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
    // Bigger text: as many lines as fit at that size (at least 2), the newest ones.
    const float wantedFont = 18.0f * textScale;
    const size_t fit = textScale <= 1.0f
                           ? kLines
                           : juce::jlimit<size_t>(2, kLines, static_cast<size_t>(area.getHeight() / (wantedFont / 0.6f)));
    const float lineH = area.getHeight() / static_cast<float>(fit);
    g.setFont(juce::Font(juce::Font::getDefaultMonospacedFontName(),
                         juce::jlimit(11.0f, wantedFont, lineH * 0.6f),
                         textScale > 1.0f ? juce::Font::bold : juce::Font::plain));
    const size_t first = lines.size() > fit ? lines.size() - fit : 0;
    // Newest line sits on the bottom row; the older ones fade upwards.
    for (size_t i = first; i < lines.size(); ++i) {
        const size_t row = fit - (lines.size() - first) + (i - first);
        const float age = static_cast<float>(lines.size() - 1 - i);
        g.setColour(juce::Colour(0xff9be59b).withAlpha(juce::jmax(0.35f, 1.0f - 0.15f * age)));
        // A long line is squeezed a little before it is cut off.
        g.drawFittedText(lines[i],
                         area.withY(area.getY() + lineH * static_cast<float>(row)).withHeight(lineH).toNearestInt(),
                         juce::Justification::centredLeft, 1, 0.7f);
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
