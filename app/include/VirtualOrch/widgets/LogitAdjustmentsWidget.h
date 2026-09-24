#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

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

    AppSession &session;

    juce::Label statusLabel;

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
