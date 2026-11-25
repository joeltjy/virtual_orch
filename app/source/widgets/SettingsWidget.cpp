#include "VirtualOrch/widgets/SettingsWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/UserSettings.h"
#include "VirtualOrch/reduction/RtLogitAdjustments.h"
#include "VirtualOrch/reduction/TdOnsetBias.h"
#include "VirtualOrch/ui/UiConstants.h"

namespace {

auto header(juce::Label &l, const juce::String &text) -> void {
    l.setText(text, juce::dontSendNotification);
    l.setFont(juce::Font(15.0f, juce::Font::bold));
    l.setColour(juce::Label::textColourId, UiConstants::workspaceWidgetTitleTextColour);
}

auto plain(juce::Label &l, const juce::String &text) -> void {
    l.setText(text, juce::dontSendNotification);
    l.setFont(juce::Font(13.0f));
    l.setColour(juce::Label::textColourId, UiConstants::workspaceWidgetTitleTextColour.withAlpha(0.9f));
}

auto onOff(bool on) -> juce::String { return on ? "on" : "off"; }

} // namespace

SettingsWidget::SettingsWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceSettingsTitle, UiConstants::workspaceWidgetTypeSettings),
      session(sessionIn) {
    auto &c = getContentComponent();
    header(reductionHeader, "Reduction: logit adjustments");
    header(orchestrationHeader, "Orchestration (OT)");
    header(scoreHeader, "Score viewer");
    plain(modelLabel, "");
    for (auto *l: {&reductionHeader, &orchestrationHeader, &scoreHeader, &modelLabel})
        c.addAndMakeVisible(*l);

    const char *rowNames[kRows] = {"Pitch", "Duration", "Onset gap", "LogitTD (temporal density)"};
    for (int r = 0; r < kRows; ++r) {
        plain(rowLabels[static_cast<size_t>(r)],
              juce::String(rowNames[r]) + "  (default " + onOff(kAdjustmentDefault) + ")");
        c.addAndMakeVisible(rowLabels[static_cast<size_t>(r)]);
    }
    plain(sideLabels[Continuation], "Continuation");
    plain(sideLabels[Simultaneous], "Simultaneous");
    for (auto &l: sideLabels) {
        l.setJustificationType(juce::Justification::centred);
        c.addAndMakeVisible(l);
    }
    for (int r = 0; r < kRows; ++r)
        for (int s = 0; s < kSides; ++s) {
            auto &t = toggles[static_cast<size_t>(r)][static_cast<size_t>(s)];
            t.onClick = [this, r, s, &t] {
                if (! updating)
                    apply(static_cast<Row>(r), static_cast<Side>(s), t.getToggleState());
            };
            c.addAndMakeVisible(t);
        }

    plain(strengthLabel, "LogitTD strength  (default " + juce::String(TdOnsetBias::kDefaultStrength, 1) + ")");
    c.addAndMakeVisible(strengthLabel);
    strengthSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    strengthSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 20);
    strengthSlider.setRange(0.0, static_cast<double>(TdOnsetBias::kMaxStrength), 0.5);
    strengthSlider.onValueChange = [this] {
        if (! updating)
            TdOnsetBias::setStrength(static_cast<float>(strengthSlider.getValue()));
    };
    c.addAndMakeVisible(strengthSlider);

    outerVoicesToggle.setButtonText("Outer voices only shift outward  (default "
                                    + onOff(UserSettings::kOuterVoicesOutwardDefault) + ")");
    outerVoicesToggle.onClick = [this] {
        if (updating)
            return;
        UserSettings::setOuterVoicesOutward(outerVoicesToggle.getToggleState());
        session.saveUserSettings();
    };
    oneInstrumentToggle.setButtonText("One instrument per octave shift  (default "
                                      + onOff(UserSettings::kOneInstrumentPerOctaveDefault) + ")");
    oneInstrumentToggle.onClick = [this] {
        if (updating)
            return;
        UserSettings::setOneInstrumentPerOctave(oneInstrumentToggle.getToggleState());
        session.saveUserSettings();
    };
    keySignatureToggle.setButtonText("Show key signature  (default "
                                     + onOff(UserSettings::kShowKeySignatureDefault) + ")");
    keySignatureToggle.onClick = [this] {
        if (updating)
            return;
        UserSettings::setShowKeySignature(keySignatureToggle.getToggleState());
        session.saveUserSettings();
    };
    for (auto *t: {&outerVoicesToggle, &oneInstrumentToggle, &keySignatureToggle}) {
        t->setColour(juce::ToggleButton::textColourId, UiConstants::workspaceWidgetTitleTextColour);
        c.addAndMakeVisible(*t);
    }

    resetButton.onClick = [this] { resetToDefaults(); };
    c.addAndMakeVisible(resetButton);

    refresh();
    startTimerHz(10);
}

SettingsWidget::~SettingsWidget() {
    stopTimer();
}

auto SettingsWidget::current(Row row, Side side) const -> std::optional<bool> {
    auto *m = session.activeAnticipated();
    if (m == nullptr)
        return std::nullopt;
    if (side == Continuation) {
        if (! m->hasSilentSide())
            return std::nullopt;
        switch (row) {
            case Pitch: return m->hybridV1PitchBias.load();
            case Duration: return m->hybridV1DurationBias.load();
            case OnsetGap: return m->hybridV1OnsetGapBias.load();
            case LogitTd: return m->hybridV1LogitTd.load();
            default: return std::nullopt;
        }
    }
    switch (row) {
        case Pitch: return m->isPitchBiasEnabled();
        case Duration: return m->isDurationBiasEnabled();
        case OnsetGap: return m->isOnsetGapBiasEnabled();
        case LogitTd: return m->isLogitTdEnabled();
        default: return std::nullopt;
    }
}

auto SettingsWidget::apply(Row row, Side side, bool on) -> void {
    auto *m = session.activeAnticipated();
    if (m == nullptr)
        return;
    if (side == Continuation) {
        if (! m->hasSilentSide())
            return;
        switch (row) {
            case Pitch: m->hybridV1PitchBias.store(on); break;
            case Duration: m->hybridV1DurationBias.store(on); break;
            case OnsetGap: m->hybridV1OnsetGapBias.store(on); break;
            case LogitTd: m->hybridV1LogitTd.store(on); break;
            default: break;
        }
        return;
    }
    switch (row) {
        case Pitch: m->setPitchBiasEnabled(on); break;
        case Duration: m->setDurationBiasEnabled(on); break;
        case OnsetGap: m->setOnsetGapBiasEnabled(on); break;
        case LogitTd: m->setLogitTdEnabled(on); break;
        default: break;
    }
}

auto SettingsWidget::refresh() -> void {
    updating = true;
    auto *m = session.activeAnticipated();
    if (m == nullptr)
        modelLabel.setText("The active reduction has no switches here (pick a Hybrid / informative-WAIT model).",
                           juce::dontSendNotification);
    else
        modelLabel.setText(m->hasSilentSide() ? "Hybrid: continuation = model 1 (piano silent), simultaneous = model 2"
                                              : "One model: its switches are in the Simultaneous column",
                           juce::dontSendNotification);
    for (int r = 0; r < kRows; ++r)
        for (int s = 0; s < kSides; ++s) {
            auto &t = toggles[static_cast<size_t>(r)][static_cast<size_t>(s)];
            const auto v = current(static_cast<Row>(r), static_cast<Side>(s));
            t.setEnabled(v.has_value());
            t.setToggleState(v.value_or(false), juce::dontSendNotification);
        }
    if (! strengthSlider.isMouseButtonDown())
        strengthSlider.setValue(static_cast<double>(TdOnsetBias::strength()), juce::dontSendNotification);
    outerVoicesToggle.setToggleState(UserSettings::outerVoicesOutward(), juce::dontSendNotification);
    oneInstrumentToggle.setToggleState(UserSettings::oneInstrumentPerOctave(), juce::dontSendNotification);
    keySignatureToggle.setToggleState(UserSettings::showKeySignature(), juce::dontSendNotification);
    updating = false;
}

auto SettingsWidget::timerCallback() -> void {
    refresh();
}

auto SettingsWidget::resetToDefaults() -> void {
    for (int r = 0; r < kRows; ++r)
        for (int s = 0; s < kSides; ++s)
            apply(static_cast<Row>(r), static_cast<Side>(s), kAdjustmentDefault);
    TdOnsetBias::setStrength(TdOnsetBias::kDefaultStrength);
    UserSettings::resetToDefaults();
    session.saveUserSettings();
    refresh();
}

auto SettingsWidget::resized() -> void {
    WorkspaceWidget::resized();
    auto area = getContentComponent().getLocalBounds().reduced(10);
    const int rowH = 24;
    const int sideW = 100;
    reductionHeader.setBounds(area.removeFromTop(rowH));
    modelLabel.setBounds(area.removeFromTop(rowH - 4));
    {
        auto head = area.removeFromTop(rowH);
        head.removeFromLeft(head.getWidth() - 2 * sideW);
        sideLabels[Continuation].setBounds(head.removeFromLeft(sideW));
        sideLabels[Simultaneous].setBounds(head.removeFromLeft(sideW));
    }
    for (int r = 0; r < kRows; ++r) {
        auto line = area.removeFromTop(rowH);
        auto right = line.removeFromRight(2 * sideW);
        rowLabels[static_cast<size_t>(r)].setBounds(line);
        for (int s = 0; s < kSides; ++s)
            toggles[static_cast<size_t>(r)][static_cast<size_t>(s)].setBounds(
                right.removeFromLeft(sideW).withSizeKeepingCentre(24, 22));
    }
    {
        auto line = area.removeFromTop(rowH);
        strengthLabel.setBounds(line.removeFromLeft(line.getWidth() / 2));
        strengthSlider.setBounds(line);
    }
    area.removeFromTop(8);
    orchestrationHeader.setBounds(area.removeFromTop(rowH));
    outerVoicesToggle.setBounds(area.removeFromTop(rowH));
    oneInstrumentToggle.setBounds(area.removeFromTop(rowH));
    area.removeFromTop(8);
    scoreHeader.setBounds(area.removeFromTop(rowH));
    keySignatureToggle.setBounds(area.removeFromTop(rowH));
    area.removeFromTop(8);
    resetButton.setBounds(area.removeFromTop(rowH + 4).removeFromLeft(160));
}
