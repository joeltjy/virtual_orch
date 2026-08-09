#include "VirtualOrch/widgets/ModeWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

ModeWidget::ModeWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceModeWidgetTitle,
                      UiConstants::workspaceWidgetTypeMode),
      session(sessionIn),
      editButton(UiConstants::workspaceModeEditButtonText),
      jamButton(UiConstants::workspaceModeJamButtonText),
      reductionStatusButton(UiConstants::workspaceModeRunningText),
      orchestrationStatusButton(UiConstants::workspaceModeRunningText) {
    auto &body = getContentComponent();

    editButton.setClickingTogglesState(true);
    jamButton.setClickingTogglesState(true);
    editButton.setRadioGroupId(1);
    jamButton.setRadioGroupId(1);

    editButton.onClick = [this] {
        session.orchestrationTransformer.setMode(OrchestrationMode::Edit);
        refreshFromSession();
    };
    jamButton.onClick = [this] {
        session.orchestrationTransformer.setMode(OrchestrationMode::Jam);
        refreshFromSession();
    };

    reductionLabel.setText(UiConstants::workspaceModeReductionLabel, juce::dontSendNotification);
    reductionLabel.setColour(juce::Label::textColourId, UiConstants::workspaceWidgetTitleTextColour);
    orchestrationLabel.setText(UiConstants::workspaceModeOrchestrationLabel,
                               juce::dontSendNotification);
    orchestrationLabel.setColour(juce::Label::textColourId,
                                 UiConstants::workspaceWidgetTitleTextColour);
    modeStatusLabel.setColour(juce::Label::textColourId, UiConstants::workspaceWidgetTitleTextColour);
    modeStatusLabel.setJustificationType(juce::Justification::centredLeft);

    reductionStatusButton.onClick = [this] {
        auto &reduction = session.activeReduction();
        reduction.paused.set(! reduction.paused.get());
        session.syncPauseTopLeds();
        refreshFromSession();
    };
    orchestrationStatusButton.onClick = [this] {
        session.orchestrationTransformer.paused.set(! session.orchestrationTransformer.paused.get());
        session.syncPauseTopLeds();
        refreshFromSession();
    };

    body.addAndMakeVisible(editButton);
    body.addAndMakeVisible(jamButton);
    body.addAndMakeVisible(modeStatusLabel);
    body.addAndMakeVisible(reductionLabel);
    body.addAndMakeVisible(reductionStatusButton);
    body.addAndMakeVisible(orchestrationLabel);
    body.addAndMakeVisible(orchestrationStatusButton);

    refreshFromSession();
    startTimer(UiConstants::workspaceModeTimerIntervalMs);
}

ModeWidget::~ModeWidget() {
    stopTimer();
}

auto ModeWidget::resized() -> void {
    WorkspaceWidget::resized();

    auto area = getContentComponent().getLocalBounds().reduced(
        UiConstants::workspaceModeContentPadding);

    auto modeRow = area.removeFromTop(UiConstants::workspaceModeRowHeight);
    editButton.setBounds(modeRow.removeFromLeft(UiConstants::workspaceModeButtonWidth));
    modeRow.removeFromLeft(UiConstants::workspaceModeRowGap);
    jamButton.setBounds(modeRow.removeFromLeft(UiConstants::workspaceModeButtonWidth));

    area.removeFromTop(UiConstants::workspaceModeRowGap);
    modeStatusLabel.setBounds(area.removeFromTop(UiConstants::workspaceModeRowHeight));

    area.removeFromTop(UiConstants::workspaceModeRowGap);
    auto reductionRow = area.removeFromTop(UiConstants::workspaceModeRowHeight);
    reductionLabel.setBounds(reductionRow.removeFromLeft(UiConstants::workspaceModeLabelWidth));
    reductionStatusButton.setBounds(
        reductionRow.removeFromLeft(UiConstants::workspaceModeStatusButtonWidth));

    area.removeFromTop(UiConstants::workspaceModeRowGap);
    auto orchRow = area.removeFromTop(UiConstants::workspaceModeRowHeight);
    orchestrationLabel.setBounds(orchRow.removeFromLeft(UiConstants::workspaceModeLabelWidth));
    orchestrationStatusButton.setBounds(
        orchRow.removeFromLeft(UiConstants::workspaceModeStatusButtonWidth));
}

auto ModeWidget::timerCallback() -> void {
    refreshFromSession();
}

auto ModeWidget::refreshFromSession() -> void {
    const bool isEdit = session.orchestrationTransformer.getMode() == OrchestrationMode::Edit;
    editButton.setToggleState(isEdit, juce::dontSendNotification);
    jamButton.setToggleState(! isEdit, juce::dontSendNotification);
    modeStatusLabel.setText(isEdit ? UiConstants::workspaceModeEditStatusText
                                   : UiConstants::workspaceModeJamStatusText,
                            juce::dontSendNotification);

    const bool reductionPaused = session.activeReduction().paused.get();
    reductionStatusButton.setButtonText(reductionPaused ? UiConstants::workspaceModePausedText
                                                        : UiConstants::workspaceModeRunningText);

    const bool orchPaused = session.orchestrationTransformer.paused.get();
    orchestrationStatusButton.setButtonText(orchPaused ? UiConstants::workspaceModePausedText
                                                       : UiConstants::workspaceModeRunningText);
}
