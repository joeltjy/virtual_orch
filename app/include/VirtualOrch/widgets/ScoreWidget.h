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
 * beams. Shows the reduction's generated notes from kWindowCs before the clock to kWindowCs
 * after it (generation runs ahead, so notes past the nowbar are already there). Every note of a
 * chord is drawn; the skyline (highest) note is highlighted, and each chord's name is written
 * above the treble staff. Layout logic lives in ScoreLayout.h.
 */
class ScoreView : public juce::Component {
public:
    static constexpr int32_t kWindowCs = 400;

    struct DrawChord {
        ScoreLayout::Chord chord;
        std::string name; // empty: no name
        bool flats = false;
    };

    ScoreView();

    auto setFrame(int32_t nowCs, std::vector<DrawChord> chords) -> void;
    auto paint(juce::Graphics &g) -> void override;

private:
    juce::Font musicFont;
    int32_t nowCs = 0;
    std::vector<DrawChord> chords;
};

class ScoreWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit ScoreWidget(AppSession &session);
    ~ScoreWidget() override;

    auto resized() -> void override;

private:
    auto timerCallback() -> void override;
    [[nodiscard]] auto nameFor(const std::vector<int> &pitches) -> const std::string &;

    AppSession &session;
    ScoreView view;
    std::map<std::vector<int>, std::string> nameCache;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ScoreWidget)
};
