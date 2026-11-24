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
      accompanimentButton(UiConstants::workspaceModeAccompanimentButtonText),
      playbackOtButton(UiConstants::workspaceModePlaybackOtButtonText),
      playbackRtButton(UiConstants::workspaceModePlaybackRtButtonText),
      reductionStatusButton(UiConstants::workspaceModeRunningText),
      orchestrationStatusButton(UiConstants::workspaceModeRunningText) {
    viewport.setViewedComponent(&body, false);
    viewport.setScrollBarThickness(UiConstants::workspaceModeContentPadding);
    viewport.getVerticalScrollBar().setColour(juce::ScrollBar::thumbColourId,
                                               UiConstants::workspaceWidgetBorderColour);
    viewport.getHorizontalScrollBar().setColour(juce::ScrollBar::thumbColourId,
                                                 UiConstants::workspaceWidgetBorderColour);
    getContentComponent().addAndMakeVisible(viewport);
    const auto labelFont = uiFont();
    const auto mono = profileFont();

    editButton.setClickingTogglesState(true);
    jamButton.setClickingTogglesState(true);
    accompanimentButton.setClickingTogglesState(true);
    editButton.setRadioGroupId(1);
    jamButton.setRadioGroupId(1);
    accompanimentButton.setRadioGroupId(1);

    playbackOtButton.setClickingTogglesState(true);
    playbackRtButton.setClickingTogglesState(true);
    playbackOtButton.setRadioGroupId(2);
    playbackRtButton.setRadioGroupId(2);
    playbackRtButton.setTooltip(UiConstants::workspaceModePlaybackRtTooltip);

    editButton.onClick = [this] {
        session.setOrchestrationMode(OrchestrationMode::Edit);
        refreshFromSession();
    };
    jamButton.onClick = [this] {
        session.setOrchestrationMode(OrchestrationMode::Jam);
        refreshFromSession();
    };
    accompanimentButton.onClick = [this] {
        session.setOrchestrationMode(OrchestrationMode::Accompaniment);
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
    body.addAndMakeVisible(accompanimentButton);
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

    constexpr int padding = UiConstants::workspaceModeContentPadding;
    constexpr int rowHeight = UiConstants::workspaceModeRowHeight;
    constexpr int gap = UiConstants::workspaceModeRowGap;
    constexpr int maxProfileLines = 13;
    const auto labelBorder = orchestrationProfileLabel.getBorderSize();
    const int profileHeight = static_cast<int>(std::ceil(profileFont().getHeight() * maxProfileLines))
                              + labelBorder.getTopAndBottom();
    const float timingWidth = UiConstants::measureTextWidth(
        profileFont(), formatAlignedStep("sample", 0.0f, 0.0f));
    const int profileWidth = static_cast<int>(std::ceil(timingWidth))
                             + labelBorder.getLeftAndRight() + 1;
    const int alertHeight = samplingAlertLabel.getText().isEmpty() ? 0
        : UiConstants::workspaceModeAlertHeight + gap;
    const int minHeight = 2 * padding + 5 * (rowHeight + gap) + profileHeight + alertHeight;
    const int minWidth = 2 * padding + 3 * profileWidth + 2 * gap;
    viewport.setBounds(getContentComponent().getLocalBounds());
    const int scrollbar = viewport.getScrollBarThickness();
    const bool needsHorizontalScroll = minWidth > viewport.getWidth()
        - (minHeight > viewport.getHeight() ? scrollbar : 0);
    const int visibleHeight = juce::jmax(0, viewport.getHeight() - (needsHorizontalScroll ? scrollbar : 0));
    const int visibleWidth = juce::jmax(0, viewport.getWidth() - (minHeight > visibleHeight ? scrollbar : 0));
    body.setSize(juce::jmax(minWidth, visibleWidth), juce::jmax(minHeight, visibleHeight));
    auto area = body.getLocalBounds().reduced(padding);

    auto modeRow = area.removeFromTop(rowHeight);
    editButton.setBounds(modeRow.removeFromLeft(UiConstants::workspaceModeButtonWidth));
    modeRow.removeFromLeft(gap);
    jamButton.setBounds(modeRow.removeFromLeft(UiConstants::workspaceModeButtonWidth));
    modeRow.removeFromLeft(gap);
    accompanimentButton.setBounds(modeRow.removeFromLeft(UiConstants::workspaceModeButtonWidth));
    modeRow.removeFromLeft(gap * 2);
    playbackOtButton.setBounds(modeRow.removeFromLeft(UiConstants::workspaceModePlaybackButtonWidth));
    modeRow.removeFromLeft(gap);
    playbackRtButton.setBounds(modeRow.removeFromLeft(UiConstants::workspaceModePlaybackButtonWidth));

    area.removeFromTop(gap);
    modeStatusLabel.setBounds(area.removeFromTop(rowHeight));
    area.removeFromTop(gap);
    samplingAlertLabel.setBounds(area.removeFromBottom(alertHeight).withTrimmedTop(gap));

    const int columnWidth = (area.getWidth() - 2 * gap) / 3;
    const auto placeProfile = [rowHeight, gap](juce::Rectangle<int> column, juce::Label &title,
                                               juce::TextButton *pause, juce::Label &total,
                                               juce::Label &profile) {
        title.setBounds(column.removeFromTop(rowHeight));
        column.removeFromTop(gap);
        auto pauseRow = column.removeFromTop(rowHeight);
        if (pause != nullptr)
            pause->setBounds(pauseRow.withWidth(UiConstants::workspaceModeStatusButtonWidth));
        column.removeFromTop(gap);
        total.setBounds(column.removeFromTop(rowHeight));
        column.removeFromTop(gap);
        profile.setBounds(column);
    };
    placeProfile(area.removeFromLeft(columnWidth), reductionLabel, &reductionStatusButton,
                 reductionLoopMsLabel, reductionProfileLabel);
    area.removeFromLeft(gap);
    placeProfile(area.removeFromLeft(columnWidth), orchestrationLabel, &orchestrationStatusButton,
                 orchestrationLoopMsLabel, orchestrationProfileLabel);
    area.removeFromLeft(gap);
    placeProfile(area, vocsepLabel, nullptr, vocsepLoopMsLabel, vocsepProfileLabel);
}

auto ModeWidget::mouseDown(const juce::MouseEvent &event) -> void {
    if (event.eventComponent == &samplingAlertLabel
        && session.modelSamplingAlert.snapshot().isNotEmpty()) {
        session.modelSamplingAlert.clear();
        lastAlertSequence = session.modelSamplingAlert.getSequence();
        samplingAlertLabel.setText({}, juce::dontSendNotification);
        resized();
        return;
    }

    WorkspaceWidget::mouseDown(event);
}

auto ModeWidget::timerCallback() -> void {
    refreshFromSession();
}

auto ModeWidget::refreshFromSession() -> void {
    const auto mode = session.orchestrationTransformer.getMode();
    editButton.setToggleState(mode == OrchestrationMode::Edit, juce::dontSendNotification);
    jamButton.setToggleState(mode == OrchestrationMode::Jam, juce::dontSendNotification);
    accompanimentButton.setToggleState(mode == OrchestrationMode::Accompaniment,
                                       juce::dontSendNotification);
    modeStatusLabel.setText(mode == OrchestrationMode::Edit
                                ? UiConstants::workspaceModeEditStatusText
                                : (mode == OrchestrationMode::Jam
                                       ? UiConstants::workspaceModeJamStatusText
                                       : UiConstants::workspaceModeAccompanimentStatusText),
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
        const auto alert = session.modelSamplingAlert.snapshot();
        const bool visibilityChanged = alert.isEmpty() != samplingAlertLabel.getText().isEmpty();
        samplingAlertLabel.setText(alert, juce::dontSendNotification);
        if (visibilityChanged)
            resized();
    }
}
