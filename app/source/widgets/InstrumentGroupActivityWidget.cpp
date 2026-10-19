#include "VirtualOrch/widgets/InstrumentGroupActivityWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

auto GroupBarsView::setProportions(const std::array<float, 3> &next) -> void {
    if (next == proportions)
        return;
    proportions = next;
    repaint();
}

auto GroupBarsView::paint(juce::Graphics &g) -> void {
    g.fillAll(UiConstants::pianoRollBackground);
    static const std::array<juce::String, 3> names{"Strings", "Woodwinds", "Brass"};
    static const std::array<juce::Colour, 3> colours{juce::Colour(0xff6cc070), juce::Colour(0xff6aaef0),
                                                     juce::Colour(0xfff07a72)};
    const auto area = getLocalBounds().toFloat().reduced(8.0f);
    const float rowH = area.getHeight() / 3.0f;
    const float labelW = 86.0f;
    const float valueW = 48.0f;
    for (size_t i = 0; i < 3; ++i) {
        const auto row = area.withY(area.getY() + rowH * static_cast<float>(i)).withHeight(rowH).reduced(0, 4);
        g.setColour(UiConstants::workspaceWidgetTitleTextColour);
        g.setFont(juce::Font(15.0f, juce::Font::bold));
        g.drawText(names[i], row.withWidth(labelW), juce::Justification::centredLeft);

        const auto track = row.withTrimmedLeft(labelW).withTrimmedRight(valueW);
        g.setColour(juce::Colours::white.withAlpha(0.08f));
        g.fillRoundedRectangle(track, 3.0f);
        g.setColour(colours[i]);
        g.fillRoundedRectangle(track.withWidth(track.getWidth() * proportions[i]), 3.0f);

        g.setColour(UiConstants::workspaceWidgetTitleTextColour);
        g.setFont(juce::Font(14.0f));
        g.drawText(juce::String(juce::roundToInt(proportions[i] * 100.0f)) + "%",
                   row.withTrimmedLeft(row.getWidth() - valueW), juce::Justification::centredRight);
    }
}

InstrumentGroupActivityWidget::InstrumentGroupActivityWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceInstrumentGroupActivityTitle,
                      UiConstants::workspaceWidgetTypeInstrumentGroupActivity),
      session(sessionIn) {
    getContentComponent().addAndMakeVisible(view);
    startTimerHz(10);
    timerCallback();
}

InstrumentGroupActivityWidget::~InstrumentGroupActivityWidget() {
    stopTimer();
}

auto InstrumentGroupActivityWidget::resized() -> void {
    WorkspaceWidget::resized();
    view.setBounds(getContentComponent().getLocalBounds());
}

auto InstrumentGroupActivityWidget::timerCallback() -> void {
    view.setProportions(session.activityLog.groupProportions(juce::Time::currentTimeMillis()));
}
