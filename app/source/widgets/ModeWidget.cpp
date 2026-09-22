#include "VirtualOrch/widgets/ModeWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/OrchLoopProfile.h"
#include "VirtualOrch/ui/UiConstants.h"

namespace {

auto padLeft(const juce::String &s, int width) -> juce::String {
    const int n = width - s.length();
    if (n <= 0)
        return s;
    return juce::String::repeatedString(" ", n) + s;
}

auto padRight(const juce::String &s, int width) -> juce::String {
    const int n = width - s.length();
    if (n <= 0)
        return s;
    return s + juce::String::repeatedString(" ", n);
}

auto formatAlignedMs(float ms) -> juce::String {
    const int decimals = ms >= 100.0f ? 0 : 1;
    return padLeft(juce::String(ms, decimals), 6) + UiConstants::workspaceModeLoopMsSuffix;
}

auto formatAlignedStep(const juce::String &name, float ms, float maxMs) -> juce::String {
    return padRight(name, 8) + formatAlignedMs(ms) + UiConstants::workspaceModeMaxPrefix
           + formatAlignedMs(maxMs);
}

auto formatReductionProfile(const ReductionLoopProfile &profile) -> juce::String {
    const auto &c = profile.current;
    const auto &m = profile.maxMs;
    juce::String text;
    text << formatAlignedStep("drain", c.drain, m.drain) << "\n";
    text << formatAlignedStep("prep", c.prep, m.prep) << "\n";
    text << formatAlignedStep("onnx", c.onnx, m.onnx) << "\n";
    text << formatAlignedStep("logits", c.logits, m.logits) << "\n";
    text << formatAlignedStep("mask", c.mask, m.mask) << "\n";
    text << formatAlignedStep("sample", c.sample, m.sample) << "\n";
    text << formatAlignedStep("push", c.push, m.push) << "\n";
    text << "ctx " << c.contextTokens << "  x" << c.onnxCalls;
    return text;
}

auto formatOrchProfile(const OrchLoopProfile &profile) -> juce::String {
    const auto &c = profile.current;
    const auto &m = profile.maxMs;
    juce::String text;
    text << formatAlignedStep("drain", c.drain, m.drain) << "\n";
    text << formatAlignedStep("snap", c.snap, m.snap) << "\n";
    text << formatAlignedStep("hist", c.hist, m.hist) << "\n";
    text << formatAlignedStep("model", c.model, m.model) << "\n";
    text << formatAlignedStep("prep", c.prep, m.prep) << "\n";
    text << formatAlignedStep("onnx", c.onnx, m.onnx) << "\n";
    text << formatAlignedStep("logits", c.logits, m.logits) << "\n";
    text << formatAlignedStep("mask", c.mask, m.mask) << "\n";
    text << formatAlignedStep("sample", c.sample, m.sample) << "\n";
    text << formatAlignedStep("decode", c.decode, m.decode) << "\n";
    text << formatAlignedStep("pack", c.pack, m.pack) << "\n";
    text << formatAlignedStep("upd", c.upd, m.upd) << "\n";
    text << "in " << c.midiIn << "/" << c.reductionIn << "  ctx " << c.contextTokens;
    if (c.getOutputCalls > 0)
        text << "  x" << c.getOutputCalls;
    return text;
}

auto formatVocsepProfile(const VocsepLoopProfile &profile) -> juce::String {
    const auto &c = profile.current;
    const auto &m = profile.maxMs;
    juce::String text;
    text << formatAlignedStep("graph", c.graph, m.graph) << "\n";
    text << formatAlignedStep("onnx", c.onnx, m.onnx) << "\n";
    text << formatAlignedStep("hungar", c.hungarian, m.hungarian) << "\n";
    text << formatAlignedStep("total", c.total, m.total) << "\n";
    text << "n " << c.nodes << "  e " << c.edges;
    return text;
}

auto profileFont() -> juce::Font {
    return {juce::Font::getDefaultMonospacedFontName(),
            UiConstants::workspaceModeProfileFontHeight, juce::Font::plain};
}

auto uiFont() -> juce::Font {
    return {UiConstants::workspaceModeUiFontHeight};
}

} // namespace

ModeWidget::ModeWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceModeWidgetTitle,
                      UiConstants::workspaceWidgetTypeMode),
      session(sessionIn),
      editButton(UiConstants::workspaceModeEditButtonText),
      jamButton(UiConstants::workspaceModeJamButtonText),
      playbackOtButton(UiConstants::workspaceModePlaybackOtButtonText),
      playbackRtButton(UiConstants::workspaceModePlaybackRtButtonText),
      reductionStatusButton(UiConstants::workspaceModeRunningText),
      orchestrationStatusButton(UiConstants::workspaceModeRunningText) {
    auto &body = getContentComponent();
    const auto labelFont = uiFont();
    const auto mono = profileFont();

    editButton.setClickingTogglesState(true);
    jamButton.setClickingTogglesState(true);
    editButton.setRadioGroupId(1);
    jamButton.setRadioGroupId(1);

    playbackOtButton.setClickingTogglesState(true);
    playbackRtButton.setClickingTogglesState(true);
    playbackOtButton.setRadioGroupId(2);
    playbackRtButton.setRadioGroupId(2);
    playbackRtButton.setTooltip(UiConstants::workspaceModePlaybackRtTooltip);

    editButton.onClick = [this] {
        session.orchestrationTransformer.setMode(OrchestrationMode::Edit);
        refreshFromSession();
    };
    jamButton.onClick = [this] {
        session.orchestrationTransformer.setMode(OrchestrationMode::Jam);
        refreshFromSession();
    };
    playbackOtButton.onClick = [this] {
        session.setPlaybackSource(PlaybackSource::Orchestration);
        refreshFromSession();
    };
    playbackRtButton.onClick = [this] {
        session.setPlaybackSource(PlaybackSource::Reduction);
        refreshFromSession();
    };

    reductionLabel.setText(UiConstants::workspaceModeReductionLabel, juce::dontSendNotification);
    reductionLabel.setFont(labelFont);
    reductionLabel.setColour(juce::Label::textColourId, UiConstants::workspaceWidgetTitleTextColour);
    orchestrationLabel.setText(UiConstants::workspaceModeOrchestrationLabel,
                               juce::dontSendNotification);
    orchestrationLabel.setFont(labelFont);
    orchestrationLabel.setColour(juce::Label::textColourId,
                                 UiConstants::workspaceWidgetTitleTextColour);
    vocsepLabel.setText(UiConstants::workspaceModeVocsepLabel, juce::dontSendNotification);
    vocsepLabel.setFont(labelFont);
    vocsepLabel.setColour(juce::Label::textColourId, UiConstants::workspaceWidgetTitleTextColour);
    modeStatusLabel.setFont(labelFont);
    modeStatusLabel.setColour(juce::Label::textColourId, UiConstants::workspaceWidgetTitleTextColour);
    modeStatusLabel.setJustificationType(juce::Justification::centredLeft);

    reductionLoopMsLabel.setFont(mono);
    reductionLoopMsLabel.setColour(juce::Label::textColourId,
                                   UiConstants::workspaceWidgetTitleTextColour);
    reductionLoopMsLabel.setJustificationType(juce::Justification::centredLeft);
    orchestrationLoopMsLabel.setFont(mono);
    orchestrationLoopMsLabel.setColour(juce::Label::textColourId,
                                       UiConstants::workspaceWidgetTitleTextColour);
    orchestrationLoopMsLabel.setJustificationType(juce::Justification::centredLeft);
    vocsepLoopMsLabel.setFont(mono);
    vocsepLoopMsLabel.setColour(juce::Label::textColourId,
                                UiConstants::workspaceWidgetTitleTextColour);
    vocsepLoopMsLabel.setJustificationType(juce::Justification::centredLeft);

    reductionProfileLabel.setFont(mono);
    reductionProfileLabel.setColour(juce::Label::textColourId,
                                    UiConstants::workspaceWidgetTitleTextColour);
    reductionProfileLabel.setJustificationType(juce::Justification::topLeft);
    reductionProfileLabel.setMinimumHorizontalScale(1.0f);
    orchestrationProfileLabel.setFont(mono);
    orchestrationProfileLabel.setColour(juce::Label::textColourId,
                                        UiConstants::workspaceWidgetTitleTextColour);
    orchestrationProfileLabel.setJustificationType(juce::Justification::topLeft);
    orchestrationProfileLabel.setMinimumHorizontalScale(1.0f);
    vocsepProfileLabel.setFont(mono);
    vocsepProfileLabel.setColour(juce::Label::textColourId,
                                 UiConstants::workspaceWidgetTitleTextColour);
    vocsepProfileLabel.setJustificationType(juce::Justification::topLeft);
    vocsepProfileLabel.setMinimumHorizontalScale(1.0f);

    samplingAlertLabel.setFont(labelFont);
    samplingAlertLabel.setColour(juce::Label::textColourId, UiConstants::workspaceModeAlertColour);
    samplingAlertLabel.setJustificationType(juce::Justification::topLeft);
    samplingAlertLabel.setMinimumHorizontalScale(1.0f);
    samplingAlertLabel.setTooltip(UiConstants::workspaceModeAlertTooltip);
    samplingAlertLabel.setInterceptsMouseClicks(true, false);
    samplingAlertLabel.addMouseListener(this, false);

    reductionStatusButton.onClick = [this] {
        auto &reduction = session.activeReduction();
        reduction.setGenerationPause(! reduction.generationPause.get());
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
    body.addAndMakeVisible(playbackOtButton);
    body.addAndMakeVisible(playbackRtButton);
    body.addAndMakeVisible(modeStatusLabel);
    body.addAndMakeVisible(reductionLabel);
    body.addAndMakeVisible(reductionStatusButton);
    body.addAndMakeVisible(reductionLoopMsLabel);
    body.addAndMakeVisible(reductionProfileLabel);
    body.addAndMakeVisible(orchestrationLabel);
    body.addAndMakeVisible(orchestrationStatusButton);
    body.addAndMakeVisible(orchestrationLoopMsLabel);
    body.addAndMakeVisible(orchestrationProfileLabel);
    body.addAndMakeVisible(vocsepLabel);
    body.addAndMakeVisible(vocsepLoopMsLabel);
    body.addAndMakeVisible(vocsepProfileLabel);
    body.addAndMakeVisible(samplingAlertLabel);

    refreshFromSession();
    startTimer(UiConstants::workspaceModeTimerIntervalMs);
}

ModeWidget::~ModeWidget() {
    samplingAlertLabel.removeMouseListener(this);
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
    modeRow.removeFromLeft(UiConstants::workspaceModeRowGap * 2);
    playbackOtButton.setBounds(
        modeRow.removeFromLeft(UiConstants::workspaceModePlaybackButtonWidth));
    modeRow.removeFromLeft(UiConstants::workspaceModeRowGap);
    playbackRtButton.setBounds(
        modeRow.removeFromLeft(UiConstants::workspaceModePlaybackButtonWidth));

    area.removeFromTop(UiConstants::workspaceModeRowGap);
    modeStatusLabel.setBounds(area.removeFromTop(UiConstants::workspaceModeRowHeight));

    area.removeFromTop(UiConstants::workspaceModeRowGap);
    samplingAlertLabel.setBounds(area.removeFromBottom(UiConstants::workspaceModeAlertHeight));
    area.removeFromBottom(UiConstants::workspaceModeRowGap);

    const int pauseBlockHeight =
        UiConstants::workspaceModeRowHeight * 3 + UiConstants::workspaceModeRowGap * 2;
    auto pauseArea = area.removeFromTop(pauseBlockHeight);
    const int halfGap = UiConstants::workspaceModeRowGap / 2;
    auto reductionCol = pauseArea.removeFromLeft(pauseArea.getWidth() / 2).withTrimmedRight(halfGap);
    auto orchCol = pauseArea.withTrimmedLeft(halfGap);

    reductionLabel.setBounds(reductionCol.removeFromTop(UiConstants::workspaceModeRowHeight));
    reductionCol.removeFromTop(UiConstants::workspaceModeRowGap);
    reductionStatusButton.setBounds(
        reductionCol.removeFromTop(UiConstants::workspaceModeRowHeight)
            .withWidth(UiConstants::workspaceModeStatusButtonWidth));
    reductionCol.removeFromTop(UiConstants::workspaceModeRowGap);
    reductionLoopMsLabel.setBounds(reductionCol.removeFromTop(UiConstants::workspaceModeRowHeight));

    orchestrationLabel.setBounds(orchCol.removeFromTop(UiConstants::workspaceModeRowHeight));
    orchCol.removeFromTop(UiConstants::workspaceModeRowGap);
    orchestrationStatusButton.setBounds(
        orchCol.removeFromTop(UiConstants::workspaceModeRowHeight)
            .withWidth(UiConstants::workspaceModeStatusButtonWidth));
    orchCol.removeFromTop(UiConstants::workspaceModeRowGap);
    orchestrationLoopMsLabel.setBounds(orchCol.removeFromTop(UiConstants::workspaceModeRowHeight));

    area.removeFromTop(UiConstants::workspaceModeRowGap);
    const int colGap = UiConstants::workspaceModeRowGap / 2;
    const int third = area.getWidth() / 3;
    auto reductionProfileCol = area.removeFromLeft(third).withTrimmedRight(colGap);
    auto orchProfileCol = area.removeFromLeft(third).withTrimmedRight(colGap);
    auto vocsepProfileCol = area.withTrimmedLeft(colGap);

    reductionProfileLabel.setBounds(reductionProfileCol);
    orchestrationProfileLabel.setBounds(orchProfileCol);

    vocsepLabel.setBounds(vocsepProfileCol.removeFromTop(UiConstants::workspaceModeRowHeight));
    vocsepProfileCol.removeFromTop(UiConstants::workspaceModeRowGap);
    vocsepLoopMsLabel.setBounds(
        vocsepProfileCol.removeFromTop(UiConstants::workspaceModeRowHeight));
    vocsepProfileCol.removeFromTop(UiConstants::workspaceModeRowGap);
    vocsepProfileLabel.setBounds(vocsepProfileCol);
}

auto ModeWidget::mouseDown(const juce::MouseEvent &event) -> void {
    if (event.eventComponent == &samplingAlertLabel
        && session.modelSamplingAlert.snapshot().isNotEmpty()) {
        session.modelSamplingAlert.clear();
        lastAlertSequence = session.modelSamplingAlert.getSequence();
        samplingAlertLabel.setText({}, juce::dontSendNotification);
        return;
    }

    WorkspaceWidget::mouseDown(event);
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

    const bool playbackOt = session.getPlaybackSource() == PlaybackSource::Orchestration;
    playbackOtButton.setToggleState(playbackOt, juce::dontSendNotification);
    playbackRtButton.setToggleState(! playbackOt, juce::dontSendNotification);

    const bool reductionPaused = session.activeReduction().generationPause.get();
    reductionStatusButton.setButtonText(reductionPaused ? UiConstants::workspaceModePausedText
                                                        : UiConstants::workspaceModeRunningText);

    const bool orchPaused = session.orchestrationTransformer.paused.get();
    orchestrationStatusButton.setButtonText(orchPaused ? UiConstants::workspaceModePausedText
                                                       : UiConstants::workspaceModeRunningText);

    const auto reductionProfile = session.activeReduction().getLastReductionProfile();
    reductionLoopMsLabel.setText(
        formatAlignedStep("total", reductionProfile.current.total, reductionProfile.maxMs.total),
        juce::dontSendNotification);
    reductionProfileLabel.setText(formatReductionProfile(reductionProfile),
                                  juce::dontSendNotification);

    const auto orchProfile = session.orchestrationTransformer.getLastOrchProfile();
    orchestrationLoopMsLabel.setText(
        formatAlignedStep("total", orchProfile.current.total, orchProfile.maxMs.total),
        juce::dontSendNotification);
    orchestrationProfileLabel.setText(formatOrchProfile(orchProfile), juce::dontSendNotification);

    const auto vocsepProfile = session.voiceSeparation.getLastProfile();
    vocsepLoopMsLabel.setText(
        formatAlignedStep("total", vocsepProfile.current.total, vocsepProfile.maxMs.total),
        juce::dontSendNotification);
    vocsepProfileLabel.setText(formatVocsepProfile(vocsepProfile), juce::dontSendNotification);

    const auto alertSeq = session.modelSamplingAlert.getSequence();
    if (alertSeq != lastAlertSequence) {
        lastAlertSequence = alertSeq;
        samplingAlertLabel.setText(session.modelSamplingAlert.snapshot(),
                                   juce::dontSendNotification);
    }
}
