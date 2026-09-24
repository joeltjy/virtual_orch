#include "VirtualOrch/widgets/LogitAdjustmentsWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/orchestration-models/IodPretrained.h"
#include "VirtualOrch/orchestration-models/IodPretrainedTypes.h"
#include "VirtualOrch/reduction/DenseDurationBias.h"
#include "VirtualOrch/reduction/DenseOnsetGapBias.h"
#include "VirtualOrch/reduction/DensePitchBias.h"
#include "VirtualOrch/reduction/ReductionTransformer.h"
#include "VirtualOrch/ui/UiConstants.h"

namespace {

constexpr int kRowH = 26;
constexpr int kGap = 3;
constexpr int kLabelW = 72;
constexpr double kTauMin = 0.0;
constexpr double kTauMax = 4.0;
constexpr double kTauStep = 0.05;

auto formatPair(float a, float b) -> juce::String {
    return "live tau=" + juce::String(a, 2) + "  tau'=" + juce::String(b, 2);
}

auto configureTauSlider(juce::Slider &slider, juce::Label &label, const juce::String &name)
    -> void {
    label.setText(name, juce::dontSendNotification);
    label.setJustificationType(juce::Justification::centredLeft);
    label.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    label.setFont(12.0f);

    slider.setSliderStyle(juce::Slider::LinearHorizontal);
    slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 48, 20);
    slider.setRange(kTauMin, kTauMax, kTauStep);
    slider.setNumDecimalPlacesToDisplay(2);
}

} // namespace

LogitAdjustmentsWidget::LogitAdjustmentsWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceLogitAdjustmentsTitle,
                      UiConstants::workspaceWidgetTypeLogitAdjustments),
      session(sessionIn) {
    auto &content = getContentComponent();

    const auto styleLive = [](juce::Label &label) {
        label.setJustificationType(juce::Justification::centredLeft);
        label.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
        label.setFont(12.0f);
    };

    content.addAndMakeVisible(statusLabel);
    styleLive(statusLabel);

    for (auto *toggle:
         {&pitchToggle, &durationToggle, &onsetGapToggle, &proportionToggle}) {
        content.addAndMakeVisible(*toggle);
        toggle->setClickingTogglesState(true);
    }
    for (auto *label:
         {&pitchLiveLabel, &durationLiveLabel, &onsetGapLiveLabel, &groupsLiveLabel}) {
        content.addAndMakeVisible(*label);
        styleLive(*label);
    }

    configureTauSlider(pitchTauBaseSlider, pitchTauBaseLabel, "tau base");
    configureTauSlider(pitchTauPrimeBaseSlider, pitchTauPrimeBaseLabel, "tau' base");
    configureTauSlider(durationTauBaseSlider, durationTauBaseLabel, "td base");
    configureTauSlider(durationTauPrimeBaseSlider, durationTauPrimeBaseLabel, "td' base");
    configureTauSlider(onsetGapTauBaseSlider, onsetGapTauBaseLabel, "tg base");
    configureTauSlider(groupsTauBaseSlider, groupsTauBaseLabel, "groups tau");

    for (auto *c: {static_cast<juce::Component *>(&pitchTauBaseLabel),
                   static_cast<juce::Component *>(&pitchTauPrimeBaseLabel),
                   static_cast<juce::Component *>(&durationTauBaseLabel),
                   static_cast<juce::Component *>(&durationTauPrimeBaseLabel),
                   static_cast<juce::Component *>(&onsetGapTauBaseLabel),
                   static_cast<juce::Component *>(&groupsTauBaseLabel),
                   static_cast<juce::Component *>(&pitchTauBaseSlider),
                   static_cast<juce::Component *>(&pitchTauPrimeBaseSlider),
                   static_cast<juce::Component *>(&durationTauBaseSlider),
                   static_cast<juce::Component *>(&durationTauPrimeBaseSlider),
                   static_cast<juce::Component *>(&onsetGapTauBaseSlider),
                   static_cast<juce::Component *>(&groupsTauBaseSlider)}) {
        content.addAndMakeVisible(*c);
    }

    pitchToggle.onClick = [this] {
        if (! ignoreToggleCallbacks)
            applyPitchBiasEnabled(pitchToggle.getToggleState());
    };
    durationToggle.onClick = [this] {
        if (! ignoreToggleCallbacks)
            applyDurationBiasEnabled(durationToggle.getToggleState());
    };
    onsetGapToggle.onClick = [this] {
        if (! ignoreToggleCallbacks)
            applyOnsetGapBiasEnabled(onsetGapToggle.getToggleState());
    };
    proportionToggle.onClick = [this] {
        if (! ignoreToggleCallbacks)
            applyProportionBiasEnabled(proportionToggle.getToggleState());
    };

    const auto onBaseChanged = [this] {
        if (! ignoreSliderCallbacks)
            applyBaseTausFromSliders();
    };
    pitchTauBaseSlider.onValueChange = onBaseChanged;
    pitchTauPrimeBaseSlider.onValueChange = onBaseChanged;
    durationTauBaseSlider.onValueChange = onBaseChanged;
    durationTauPrimeBaseSlider.onValueChange = onBaseChanged;
    onsetGapTauBaseSlider.onValueChange = onBaseChanged;
    groupsTauBaseSlider.onValueChange = onBaseChanged;

    syncRtBiasTogglesFromModels();
    syncBaseSlidersFromModels();
    proportionToggle.setToggleState(
        session.orchestrationTransformer.proportionBiasEnabled.get(),
        juce::dontSendNotification);

    startTimer(UiConstants::workspaceModeTimerIntervalMs);
    refreshFromSession();
}

LogitAdjustmentsWidget::~LogitAdjustmentsWidget() {
    stopTimer();
}

auto LogitAdjustmentsWidget::resized() -> void {
    WorkspaceWidget::resized();
    auto area = getContentComponent().getLocalBounds().reduced(8, 4);

    statusLabel.setBounds(area.removeFromTop(16));
    area.removeFromTop(kGap);

    const auto placeToggleRow = [&](juce::ToggleButton &toggle, juce::Label &live) {
        auto row = area.removeFromTop(kRowH);
        toggle.setBounds(row.removeFromLeft(juce::jmin(190, row.getWidth() / 2)));
        live.setBounds(row);
        area.removeFromTop(kGap);
    };

    const auto placeSliderRow = [&](juce::Label &label, juce::Slider &slider) {
        auto row = area.removeFromTop(kRowH);
        label.setBounds(row.removeFromLeft(kLabelW));
        slider.setBounds(row);
        area.removeFromTop(kGap);
    };

    placeToggleRow(pitchToggle, pitchLiveLabel);
    placeSliderRow(pitchTauBaseLabel, pitchTauBaseSlider);
    placeSliderRow(pitchTauPrimeBaseLabel, pitchTauPrimeBaseSlider);

    placeToggleRow(durationToggle, durationLiveLabel);
    placeSliderRow(durationTauBaseLabel, durationTauBaseSlider);
    placeSliderRow(durationTauPrimeBaseLabel, durationTauPrimeBaseSlider);

    placeToggleRow(onsetGapToggle, onsetGapLiveLabel);
    placeSliderRow(onsetGapTauBaseLabel, onsetGapTauBaseSlider);

    placeToggleRow(proportionToggle, groupsLiveLabel);
    placeSliderRow(groupsTauBaseLabel, groupsTauBaseSlider);
}

auto LogitAdjustmentsWidget::timerCallback() -> void {
    refreshFromSession();
}

auto LogitAdjustmentsWidget::syncRtBiasTogglesFromModels() -> void {
    ignoreToggleCallbacks = true;
    const bool dense =
        session.musicModelArch == MusicModelArch::DenseV2
        || session.musicModelArch == MusicModelArch::DensePianoReduction;
    if (session.musicModelArch == MusicModelArch::DensePianoReduction) {
        pitchToggle.setToggleState(
            session.reductionTransformerPianoReduction.isPitchBiasEnabled(),
            juce::dontSendNotification);
        durationToggle.setToggleState(
            session.reductionTransformerPianoReduction.isDurationBiasEnabled(),
            juce::dontSendNotification);
        onsetGapToggle.setToggleState(
            session.reductionTransformerPianoReduction.isOnsetGapBiasEnabled(),
            juce::dontSendNotification);
    } else {
        pitchToggle.setToggleState(session.reductionTransformerV2.isPitchBiasEnabled(),
                                   juce::dontSendNotification);
        durationToggle.setToggleState(session.reductionTransformerV2.isDurationBiasEnabled(),
                                      juce::dontSendNotification);
        onsetGapToggle.setToggleState(session.reductionTransformerV2.isOnsetGapBiasEnabled(),
                                      juce::dontSendNotification);
    }
    pitchToggle.setEnabled(dense);
    durationToggle.setEnabled(dense);
    onsetGapToggle.setEnabled(dense);
    pitchTauBaseSlider.setEnabled(dense);
    pitchTauPrimeBaseSlider.setEnabled(dense);
    durationTauBaseSlider.setEnabled(dense);
    durationTauPrimeBaseSlider.setEnabled(dense);
    onsetGapTauBaseSlider.setEnabled(dense);
    ignoreToggleCallbacks = false;
}

auto LogitAdjustmentsWidget::syncBaseSlidersFromModels() -> void {
    ignoreSliderCallbacks = true;
    const bool piano = session.musicModelArch == MusicModelArch::DensePianoReduction;

    if (piano) {
        pitchTauBaseSlider.setValue(session.reductionTransformerPianoReduction.getPitchTauBase(),
                                    juce::dontSendNotification);
        pitchTauPrimeBaseSlider.setValue(
            session.reductionTransformerPianoReduction.getPitchTauPrimeBase(),
            juce::dontSendNotification);
        durationTauBaseSlider.setValue(
            session.reductionTransformerPianoReduction.getDurationTauBase(),
            juce::dontSendNotification);
        durationTauPrimeBaseSlider.setValue(
            session.reductionTransformerPianoReduction.getDurationTauPrimeBase(),
            juce::dontSendNotification);
        onsetGapTauBaseSlider.setValue(
            session.reductionTransformerPianoReduction.getOnsetGapTauBase(),
            juce::dontSendNotification);
    } else {
        pitchTauBaseSlider.setValue(session.reductionTransformerV2.getPitchTauBase(),
                                    juce::dontSendNotification);
        pitchTauPrimeBaseSlider.setValue(session.reductionTransformerV2.getPitchTauPrimeBase(),
                                         juce::dontSendNotification);
        durationTauBaseSlider.setValue(session.reductionTransformerV2.getDurationTauBase(),
                                       juce::dontSendNotification);
        durationTauPrimeBaseSlider.setValue(
            session.reductionTransformerV2.getDurationTauPrimeBase(),
            juce::dontSendNotification);
        onsetGapTauBaseSlider.setValue(session.reductionTransformerV2.getOnsetGapTauBase(),
                                       juce::dontSendNotification);
    }

    float groupsTau = IodPretrainedTypes::groupsBiasTau;
    if (auto *iod = dynamic_cast<IodPretrained *>(session.orchestrationModel.get()))
        groupsTau = iod->getGroupsBiasTau();
    groupsTauBaseSlider.setValue(groupsTau, juce::dontSendNotification);
    ignoreSliderCallbacks = false;
}

auto LogitAdjustmentsWidget::applyPitchBiasEnabled(bool enabled) -> void {
    session.reductionTransformerV2.setPitchBiasEnabled(enabled);
    session.reductionTransformerPianoReduction.setPitchBiasEnabled(enabled);
}

auto LogitAdjustmentsWidget::applyDurationBiasEnabled(bool enabled) -> void {
    session.reductionTransformerV2.setDurationBiasEnabled(enabled);
    session.reductionTransformerPianoReduction.setDurationBiasEnabled(enabled);
}

auto LogitAdjustmentsWidget::applyOnsetGapBiasEnabled(bool enabled) -> void {
    session.reductionTransformerV2.setOnsetGapBiasEnabled(enabled);
    session.reductionTransformerPianoReduction.setOnsetGapBiasEnabled(enabled);
}

auto LogitAdjustmentsWidget::applyProportionBiasEnabled(bool enabled) -> void {
    session.orchestrationTransformer.proportionBiasEnabled.set(enabled);
    auto orch = session.presetStore.settings.getOrCreateChildWithName("orchestration", nullptr);
    orch.setProperty("proportionBias", enabled, nullptr);
    session.presetStore.saveSettings();
}

auto LogitAdjustmentsWidget::applyBaseTausFromSliders() -> void {
    const float pitchBase = static_cast<float>(pitchTauBaseSlider.getValue());
    const float pitchPrime = static_cast<float>(pitchTauPrimeBaseSlider.getValue());
    const float durBase = static_cast<float>(durationTauBaseSlider.getValue());
    const float durPrime = static_cast<float>(durationTauPrimeBaseSlider.getValue());
    const float gapBase = static_cast<float>(onsetGapTauBaseSlider.getValue());
    const float groupsTau = static_cast<float>(groupsTauBaseSlider.getValue());

    session.reductionTransformerV2.setPitchTauBase(pitchBase);
    session.reductionTransformerV2.setPitchTauPrimeBase(pitchPrime);
    session.reductionTransformerV2.setDurationTauBase(durBase);
    session.reductionTransformerV2.setDurationTauPrimeBase(durPrime);
    session.reductionTransformerV2.setOnsetGapTauBase(gapBase);

    session.reductionTransformerPianoReduction.setPitchTauBase(pitchBase);
    session.reductionTransformerPianoReduction.setPitchTauPrimeBase(pitchPrime);
    session.reductionTransformerPianoReduction.setDurationTauBase(durBase);
    session.reductionTransformerPianoReduction.setDurationTauPrimeBase(durPrime);
    session.reductionTransformerPianoReduction.setOnsetGapTauBase(gapBase);

    if (auto *iod = dynamic_cast<IodPretrained *>(session.orchestrationModel.get()))
        iod->setGroupsBiasTau(groupsTau);
}

auto LogitAdjustmentsWidget::refreshFromSession() -> void {
    syncRtBiasTogglesFromModels();

    ignoreToggleCallbacks = true;
    proportionToggle.setToggleState(
        session.orchestrationTransformer.proportionBiasEnabled.get(),
        juce::dontSendNotification);
    ignoreToggleCallbacks = false;

    // Keep slider values unless the user is dragging (timer shouldn't fight).
    if (! pitchTauBaseSlider.isMouseButtonDown() && ! pitchTauPrimeBaseSlider.isMouseButtonDown()
        && ! durationTauBaseSlider.isMouseButtonDown()
        && ! durationTauPrimeBaseSlider.isMouseButtonDown()
        && ! onsetGapTauBaseSlider.isMouseButtonDown()
        && ! groupsTauBaseSlider.isMouseButtonDown()) {
        syncBaseSlidersFromModels();
    }

    float pitchTau = session.reductionTransformerV2.getPitchTauBase();
    float pitchTauPrime = session.reductionTransformerV2.getPitchTauPrimeBase();
    float durationTau = session.reductionTransformerV2.getDurationTauBase();
    float durationTauPrime = session.reductionTransformerV2.getDurationTauPrimeBase();
    float onsetGapTau = session.reductionTransformerV2.getOnsetGapTauBase();
    juce::String archLabel = "AMT / no RT bias";

    if (session.musicModelArch == MusicModelArch::DenseV2) {
        archLabel = "Reduction V2";
        session.reductionTransformerV2.refreshPitchTauSchedule();
        session.reductionTransformerV2.refreshDurationTauSchedule();
        session.reductionTransformerV2.refreshOnsetGapTauSchedule();
        pitchTau = session.reductionTransformerV2.getPitchTau();
        pitchTauPrime = session.reductionTransformerV2.getPitchTauPrime();
        durationTau = session.reductionTransformerV2.getDurationTau();
        durationTauPrime = session.reductionTransformerV2.getDurationTauPrime();
        onsetGapTau = session.reductionTransformerV2.getOnsetGapTau();
    } else if (session.musicModelArch == MusicModelArch::DensePianoReduction) {
        archLabel = "Piano Reduction";
        session.reductionTransformerPianoReduction.refreshPitchTauSchedule();
        session.reductionTransformerPianoReduction.refreshDurationTauSchedule();
        session.reductionTransformerPianoReduction.refreshOnsetGapTauSchedule();
        pitchTau = session.reductionTransformerPianoReduction.getPitchTau();
        pitchTauPrime = session.reductionTransformerPianoReduction.getPitchTauPrime();
        durationTau = session.reductionTransformerPianoReduction.getDurationTau();
        durationTauPrime = session.reductionTransformerPianoReduction.getDurationTauPrime();
        onsetGapTau = session.reductionTransformerPianoReduction.getOnsetGapTau();
    }

    statusLabel.setText(archLabel, juce::dontSendNotification);
    pitchLiveLabel.setText(formatPair(pitchTau, pitchTauPrime), juce::dontSendNotification);
    durationLiveLabel.setText(formatPair(durationTau, durationTauPrime),
                              juce::dontSendNotification);
    onsetGapLiveLabel.setText("live tg=" + juce::String(onsetGapTau, 2),
                              juce::dontSendNotification);

    float groupsTau = IodPretrainedTypes::groupsBiasTau;
    if (auto *iod = dynamic_cast<IodPretrained *>(session.orchestrationModel.get()))
        groupsTau = iod->getGroupsBiasTau();
    const bool groupsOn = session.orchestrationTransformer.proportionBiasEnabled.get();
    groupsLiveLabel.setText("tau=" + juce::String(groupsTau, 2) + (groupsOn ? "  (on)" : "  (off)"),
                            juce::dontSendNotification);
}
