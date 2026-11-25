#pragma once

#include <JuceHeader.h>

#include <array>
#include <optional>

#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/**
 * Settings (Nov 25 2026, Joel): the choices that used to be fixed defaults, in one place, each with its default.
 *
 *  Reduction (the active transformer's own switches, as the Logit Adjustments widget; saved in presets):
 *    pitch / duration / onset-gap adjustments and LogitTD, for the continuation model (Hybrid model 1, "hybridV1")
 *    and the simultaneous model (model 2, or the only model of a non-hybrid transformer, "model");
 *    the LogitTD strength.
 *  Orchestration (UserSettings, saved in settings.xml):
 *    outer voices only shift outward; one instrument per octave shift.
 *  Score viewer (UserSettings): show the key signature.
 *
 * Every control follows the live state (10 Hz), so changes made elsewhere (a preset, the Logit Adjustments widget,
 * the Score widget's own toggle) show here. "Reset to defaults" puts all of them back.
 */
class SettingsWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit SettingsWidget(AppSession &session);
    ~SettingsWidget() override;

    auto resized() -> void override;

    /** The reduction adjustments' defaults (v1.1 / v1.2 presets): all on, LogitTD strength 6. */
    static constexpr bool kAdjustmentDefault = true;

private:
    enum Row { Pitch = 0, Duration, OnsetGap, LogitTd, kRows };
    enum Side { Continuation = 0, Simultaneous, kSides };

    auto timerCallback() -> void override;
    auto refresh() -> void;
    auto apply(Row row, Side side, bool on) -> void;
    [[nodiscard]] auto current(Row row, Side side) const -> std::optional<bool>;
    auto resetToDefaults() -> void;

    AppSession &session;

    juce::Label reductionHeader, orchestrationHeader, scoreHeader, modelLabel;
    std::array<juce::Label, kRows> rowLabels;
    std::array<juce::Label, kSides> sideLabels;
    std::array<std::array<juce::ToggleButton, kSides>, kRows> toggles;
    juce::Label strengthLabel;
    juce::Slider strengthSlider;
    juce::ToggleButton outerVoicesToggle, oneInstrumentToggle, keySignatureToggle;
    juce::TextButton resetButton{"Reset to defaults"};
    bool updating = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SettingsWidget)
};
