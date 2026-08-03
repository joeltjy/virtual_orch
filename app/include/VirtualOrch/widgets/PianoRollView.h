#pragma once

#include <JuceHeader.h>

#include <vector>

#include "VirtualOrch/Clock.h"
#include "VirtualOrch/MusicTransformer.h"

class PianoRollView : public juce::Component, private juce::Timer {
public:
    explicit PianoRollView(Clock &clock);

    ~PianoRollView() override;

    /** History / committed notes (drawn first). */
    auto setNotes(std::vector<Token> notes) -> void;

    /** Just-drained / pending notes (drawn on top of history). */
    auto setPendingNotes(std::vector<Token> notes) -> void;

    auto setNoteColours(juce::Colour history, juce::Colour pending) -> void;

    auto setPitchRange(int32_t low, int32_t high) -> void;

    auto paint(juce::Graphics &g) -> void override;

private:
    auto timerCallback() -> void override;

    [[nodiscard]] auto timeToX(int32_t time, int32_t nowTime, float width) const -> float;

    [[nodiscard]] auto pitchToY(int32_t pitch, float height) const -> float;

    [[nodiscard]] auto nowBarX(float width) const -> float;

    auto paintNotes(juce::Graphics &g,
                    const std::vector<Token> &tokens,
                    juce::Colour colour,
                    int32_t nowTime,
                    float width,
                    float height,
                    float noteHeight) const -> void;

    Clock &clock;
    std::vector<Token> notes;
    std::vector<Token> pendingNotes;
    juce::Colour historyNoteColour;
    juce::Colour pendingNoteColour;
    int32_t pitchLow = 36;
    int32_t pitchHigh = 96;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PianoRollView)
};
