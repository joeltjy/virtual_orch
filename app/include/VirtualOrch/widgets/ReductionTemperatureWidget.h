#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/**
 * Dense RT per-field sampling temperatures, plus the add/remove proportion sliders of the
 * vocsep ensemble (the global on/off switch lives in MainComponent).
 */
class ReductionTemperatureWidget : public WorkspaceWidget {
public:
    explicit ReductionTemperatureWidget(AppSession &session);

    ~ReductionTemperatureWidget() override = default;

    auto resized() -> void override;

private:
    auto syncSlidersFromSession() -> void;
    auto syncEnsembleFromSettings() -> void;

    AppSession &session;
    juce::Label onsetLabel;
    juce::Label durationLabel;
    juce::Label noteLabel;
    juce::Label velocityLabel;
    juce::Slider onsetSlider;
    juce::Slider durationSlider;
    juce::Slider noteSlider;
    juce::Slider velocitySlider;

    juce::Label addPLabel;
    juce::Label removePLabel;
    juce::Slider addPSlider;
    juce::Slider removePSlider;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ReductionTemperatureWidget)
};
