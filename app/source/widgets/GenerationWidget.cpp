#include "VirtualOrch/widgets/GenerationWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

GenerationWidget::GenerationWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceGenerationWidgetTitle,
                      UiConstants::workspaceWidgetTypeGeneration),
      session(sessionIn) {
    auto &body = getContentComponent();

    startButton.onClick = [this] { session.startGeneration(); };
    stopButton.onClick = [this] { session.stopGeneration(); };
    body.addAndMakeVisible(startButton);
    body.addAndMakeVisible(stopButton);

    progressBar = std::make_unique<juce::ProgressBar>(session.outputPlayback.progress);
    progressBar->setLookAndFeel(&progressBarLookAndFeel);
    progressBar->setPercentageDisplay(false);
    body.addAndMakeVisible(*progressBar);

    timeLabel.setJustificationType(juce::Justification::centredLeft);
    timeLabel.setColour(juce::Label::textColourId, UiConstants::workspaceTransportTimeColour);
    timeLabel.setFont(juce::Font(UiConstants::workspaceGenerationClockFontHeight));
    timeLabel.setInterceptsMouseClicks(false, false);
    body.addAndMakeVisible(timeLabel);

    updateTimeDisplay();
    startTimer(UiConstants::workspaceGenerationClockTimerIntervalMs);
}

GenerationWidget::~GenerationWidget() {
    stopTimer();
    if (progressBar != nullptr)
        progressBar->setLookAndFeel(nullptr);
}

auto GenerationWidget::resized() -> void {
    WorkspaceWidget::resized();

    auto area = getContentComponent().getLocalBounds().reduced(
        UiConstants::workspaceGenerationContentPadding);

    auto buttons = area.removeFromTop(UiConstants::workspaceGenerationButtonHeight);
    startButton.setBounds(buttons.removeFromLeft(UiConstants::workspaceGenerationButtonWidth));
    buttons.removeFromLeft(UiConstants::workspaceGenerationControlGap);
    stopButton.setBounds(buttons.removeFromLeft(UiConstants::workspaceGenerationButtonWidth));

    area.removeFromTop(UiConstants::workspaceGenerationControlGap);
    if (progressBar != nullptr)
        progressBar->setBounds(area.removeFromTop(UiConstants::workspaceGenerationProgressHeight));

    area.removeFromTop(UiConstants::workspaceGenerationControlGap);
    timeLabel.setBounds(area.removeFromTop(UiConstants::workspaceGenerationClockHeight));
}

auto GenerationWidget::timerCallback() -> void {
    updateTimeDisplay();
}

auto GenerationWidget::updateTimeDisplay() -> void {
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
