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
}

GenerationWidget::~GenerationWidget() {
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
}
