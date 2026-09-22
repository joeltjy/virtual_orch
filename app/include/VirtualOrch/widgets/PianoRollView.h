#pragma once

#include <JuceHeader.h>

#include <vector>

#include "VirtualOrch/Clock.h"
#include "VirtualOrch/MusicTransformer.h"

class PianoRollView : public juce::Component, private juce::Timer {
public:
    explicit PianoRollView(Clock &clock);

    ~PianoRollView() override;

    /** History / committed notes (drawn first). Optional per-note colours (same size). */
    auto setNotes(std::vector<Token> notes, std::vector<juce::Colour> colours = {}) -> void;

    /** Just-drained / pending notes (drawn on top of history). Optional per-note colours. */
    auto setPendingNotes(std::vector<Token> notes, std::vector<juce::Colour> colours = {}) -> void;

    auto setNoteColours(juce::Colour history, juce::Colour pending) -> void;

    auto setPitchRange(int32_t low, int32_t high) -> void;

    auto paint(juce::Graphics &g) -> void override;

private:
    struct TimeWindow {
        int32_t leftTime = 0;
        int32_t rightTime = 1;
        int32_t nowTime = 0;
    };

    auto timerCallback() -> void override;

    [[nodiscard]] auto isDrawableNote(const Token &token) const -> bool;

    [[nodiscard]] auto computeTimeWindow() const -> TimeWindow;

    [[nodiscard]] auto timeToX(int32_t time, const TimeWindow &window, float width) const -> float;

    [[nodiscard]] auto pitchToY(int32_t pitch, float height) const -> float;

    [[nodiscard]] auto nowBarX(const TimeWindow &window, float width) const -> float;

    auto paintNotes(juce::Graphics &g,
                    const std::vector<Token> &tokens,
                    const std::vector<juce::Colour> &colours,
                    juce::Colour fallbackColour,
                    const TimeWindow &window,
                    float width,
                    float height,
                    float noteHeight) const -> void;

    Clock &clock;
    std::vector<Token> notes;
    std::vector<Token> pendingNotes;
    std::vector<juce::Colour> noteColours;
    std::vector<juce::Colour> pendingNoteColours;
    juce::Colour historyNoteColour;
    juce::Colour pendingNoteColour;
    int32_t pitchLow = 36;
    int32_t pitchHigh = 96;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PianoRollView)
};
