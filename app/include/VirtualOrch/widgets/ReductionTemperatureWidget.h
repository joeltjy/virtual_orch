#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/** Sliders for dense RT per-field sampling temperatures (onset / duration / note / velocity). */
class ReductionTemperatureWidget : public WorkspaceWidget {
public:
    explicit ReductionTemperatureWidget(AppSession &session);

    ~ReductionTemperatureWidget() override = default;

    auto resized() -> void override;

private:
    auto syncSlidersFromSession() -> void;

    AppSession &session;
    juce::Label onsetLabel;
    juce::Label durationLabel;
    juce::Label noteLabel;
    juce::Label velocityLabel;
    juce::Slider onsetSlider;
    juce::Slider durationSlider;
    juce::Slider noteSlider;
    juce::Slider velocitySlider;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ReductionTemperatureWidget)
};
