#include "VirtualOrch/widgets/TransportWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

TransportWidget::TransportWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceTransportWidgetTitle,
                      UiConstants::workspaceWidgetTypeTransport),
      session(sessionIn) {
    timeLabel.setJustificationType(juce::Justification::centred);
    timeLabel.setColour(juce::Label::textColourId, UiConstants::workspaceTransportTimeColour);
    timeLabel.setFont(juce::Font(UiConstants::workspaceTransportTimeFontHeight));
    timeLabel.setInterceptsMouseClicks(false, false);
    getContentComponent().addAndMakeVisible(timeLabel);

    updateTimeDisplay();
    startTimer(UiConstants::workspaceTransportTimerIntervalMs);
}

TransportWidget::~TransportWidget() {
    stopTimer();
}

auto TransportWidget::resized() -> void {
    WorkspaceWidget::resized();
    timeLabel.setBounds(getContentComponent().getLocalBounds());
}

auto TransportWidget::timerCallback() -> void {
    updateTimeDisplay();
}

auto TransportWidget::updateTimeDisplay() -> void {
    auto time = static_cast<int>(session.clock.getTime());
    const auto hours = juce::String(time / (100 * 60 * 60)).paddedLeft('0', 2);
    time %= (100 * 60 * 60);
    const auto minutes = juce::String(time / (100 * 60)).paddedLeft('0', 2);
    time %= (100 * 60);
    const auto seconds = juce::String(time / 100).paddedLeft('0', 2);
    time %= 100;
    const juce::String timeStr =
        hours + ":" + minutes + ":" + seconds + "." + juce::String(time).paddedLeft('0', 2);
    timeLabel.setText(timeStr, juce::dontSendNotification);
}
