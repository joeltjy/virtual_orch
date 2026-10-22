#pragma once

#include <JuceHeader.h>

#include <map>
#include <string>
#include <vector>

#include "VirtualOrch/ScoreLayout.h"
#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/**
 * Rolling grand staff (treble + bass) with a nowbar in the middle: noteheads only, no stems or
 * beams. Shows the MELODY and BASS of the reduction's generated notes (ScoreLayout::
 * skylineAndBass: onsets within 30 ms snapped, then ~/orchestration's skyline_and_bass) from
 * kWindowCs before the clock to kWindowCs after it, and each chord's name above the treble staff.
 */
class ScoreView : public juce::Component {
public:
    static constexpr int32_t kWindowCs = 200; // 2 s each side of the nowbar

    struct DrawNote {
        int32_t onsetCs = 0;
        int pitch = 60;
        bool flats = false; // spelled with flats (the chord's root is flat)
        bool bass = false;  // the bass line (else the melody)
    };
    struct DrawName {
        int32_t onsetCs = 0;
        std::string name;
    };

    ScoreView();

    auto setFrame(int32_t nowCs, std::vector<DrawNote> notes, std::vector<DrawName> names) -> void;
    /** Larger chord names for an audience (the staff makes room above itself). */
    auto setTextScale(float scale) -> void { textScale = scale; repaint(); }
    auto paint(juce::Graphics &g) -> void override;

private:
    /** A sharp centred on the note's line / space, a flat with its bowl on it, left of the head. */
    auto drawAccidental(juce::Graphics &g, bool sharp, float headLeftX, float cy, float s) -> void;

    juce::Font musicFont;
    juce::Path sharpSign, flatSign; // at height 1
    int32_t nowCs = 0;
    float textScale = 1.0f;
    std::vector<DrawNote> notes;
    std::vector<DrawName> names;
};

class ScoreWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit ScoreWidget(AppSession &session);
    ~ScoreWidget() override;

    auto resized() -> void override;

protected:
    auto textScaleChanged() -> void override { view.setTextScale(getTextScale()); }

private:
    auto timerCallback() -> void override;
    [[nodiscard]] auto nameFor(const std::vector<int> &pitches) -> const std::string &;

    AppSession &session;
    ScoreView view;
    std::map<std::vector<int>, std::string> nameCache;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ScoreWidget)
};
