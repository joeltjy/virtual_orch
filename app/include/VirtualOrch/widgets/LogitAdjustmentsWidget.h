#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;
class TwoInstrumentAnticipatedReductionTransformer;

/**
 * Live τ readouts + base-τ sliders for RT pitch / duration / onset-gap and IOD
 * groups bias, with toggles to enable/disable each logit adjustment.
 */
class LogitAdjustmentsWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit LogitAdjustmentsWidget(AppSession &session);

    ~LogitAdjustmentsWidget() override;

    auto resized() -> void override;

private:
    auto timerCallback() -> void override;

    auto refreshFromSession() -> void;

    auto syncRtBiasTogglesFromModels() -> void;

    auto syncBaseSlidersFromModels() -> void;

    auto applyPitchBiasEnabled(bool enabled) -> void;
    auto applyDurationBiasEnabled(bool enabled) -> void;
    auto applyOnsetGapBiasEnabled(bool enabled) -> void;
    auto applyProportionBiasEnabled(bool enabled) -> void;

    auto applyBaseTausFromSliders() -> void;

    /** The active T3'' / W2-W3 / Hybrid transformer (their own adjustment settings), else nullptr. */
    [[nodiscard]] auto anticipatedModel() const -> TwoInstrumentAnticipatedReductionTransformer *;
    /** Hybrid with "v1" chosen: the switches show and set the v1 side (hybridV1*). */
    [[nodiscard]] auto showsHybridV1() const -> bool;

    AppSession &session;

    juce::Label statusLabel;
    /** Hybrid only: which side the switches below show (on = v1, the default; off = W2/W3). */
    juce::ToggleButton hybridSideToggle{"Show v1's switches (off: W2/W3's)"};

    /** Global (BiasSettings): biases on/off, and if on, only while the piano is silent > 1 s. */
    juce::ToggleButton biasesOnToggle{"Logit biases on"};
    juce::ToggleButton biasGateToggle{"Only when the piano is silent > 1 s"};
    /** TdOnsetBias: shorter onset gaps for higher Launchpad TD (independent of the toggle above). */
    juce::ToggleButton logitTdToggle{"LogitTD (shorter onset gaps for higher TD)"};
    /** T3'' / W2-W3 / Hybrid only (the older models always use it): the 1..8 cs penalty. */
    juce::ToggleButton shortGapToggle{"Short-gap penalty (1-8 cs)"};
    /** W2/W3 and Hybrid: play when P(play) >= this (threshold rule, 0.30 recommended). */
    juce::Label waitMinPlayLabel;
    juce::Slider waitMinPlaySlider;
    juce::Label logitTdStrengthLabel;
    juce::Slider logitTdStrengthSlider; // TdOnsetBias::strength (exponent at TD 64), saved

    juce::ToggleButton pitchToggle{"Pitch bias"};
    juce::ToggleButton durationToggle{"Duration bias"};
    juce::ToggleButton onsetGapToggle{"Onset-gap bias"};
    juce::ToggleButton proportionToggle{"OT proportion / IOD groups"};

    juce::Label pitchLiveLabel;
    juce::Label durationLiveLabel;
    juce::Label onsetGapLiveLabel;
    juce::Label groupsLiveLabel;

    juce::Label pitchTauBaseLabel;
    juce::Label pitchTauPrimeBaseLabel;
    juce::Label durationTauBaseLabel;
    juce::Label durationTauPrimeBaseLabel;
    juce::Label onsetGapTauBaseLabel;
    juce::Label groupsTauBaseLabel;

    juce::Slider pitchTauBaseSlider;
    juce::Slider pitchTauPrimeBaseSlider;
    juce::Slider durationTauBaseSlider;
    juce::Slider durationTauPrimeBaseSlider;
    juce::Slider onsetGapTauBaseSlider;
    juce::Slider groupsTauBaseSlider;

    bool ignoreToggleCallbacks = false;
    bool ignoreSliderCallbacks = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LogitAdjustmentsWidget)
};
