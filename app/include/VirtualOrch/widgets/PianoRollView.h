#pragma once

#include <JuceHeader.h>

#include <vector>

#include "VirtualOrch/Clock.h"
#include "VirtualOrch/MusicTransformer.h"

/**
 * Fixed nowbar at 75% of width. Timeline scrolls so clock time stays under it.
 */
class PianoRollView : public juce::Component, private juce::Timer {
public:
    explicit PianoRollView(Clock &clock);

    ~PianoRollView() override;

    auto setNotes(std::vector<Token> notes) -> void;

    auto setPitchRange(int32_t low, int32_t high) -> void;

    auto paint(juce::Graphics &g) -> void override;

private:
    auto timerCallback() -> void override;

    [[nodiscard]] auto timeToX(int32_t time, int32_t nowTime, float width) const -> float;

    [[nodiscard]] auto pitchToY(int32_t pitch, float height) const -> float;

    [[nodiscard]] auto nowBarX(float width) const -> float;

    Clock &clock;
    std::vector<Token> notes;
    int32_t pitchLow = 36;
    int32_t pitchHigh = 96;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PianoRollView)
};
