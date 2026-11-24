#pragma once

#include <JuceHeader.h>

#include <map>
#include <string>
#include <vector>

#include "VirtualOrch/KeyDetector.h"
#include "VirtualOrch/ScoreLayout.h"
#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/**
 * Rolling grand staff (treble + bass) with a nowbar in the middle: noteheads only, no stems or
 * beams. Shows the MELODY and BASS of the reduction's generated notes (ScoreLayout::
 * skylineAndBass: onsets within 30 ms snapped, per onset group the top / bottom note unless a
 * note sounding with it lies above / below) from kWindowCs before the clock to kWindowCs after
 * it, the chord names above the treble staff (ScoreHarmony: shown when they change, N/C for a
 * single line), and the key (KeyDetector), optionally as a key signature.
 */
class ScoreView : public juce::Component {
public:
    static constexpr int32_t kWindowCs = 200; // 2 s each side of the nowbar

    struct DrawNote {
        int32_t onsetCs = 0;
        int pitch = 60;
        int step = 35;      // staff position (Spelling::staffStep)
        int accidental = 0; // of its spelling: -2 .. +2
        bool bass = false;  // the bass line (else the melody)
    };
    struct DrawName {
        int32_t onsetCs = 0;
        std::string name;
    };
    struct DrawKey {
        std::string name;     // "C major", "" before a key is known
        std::string runnerUp; // a close second, or ""
        bool uncertain = true;
        int signature = 0;    // sharps (> 0) / flats (< 0)
    };

    ScoreView();

    auto setFrame(int32_t nowCs, std::vector<DrawNote> notes, std::vector<DrawName> names, DrawKey key) -> void;
    auto setFrame(int32_t nowCs, std::vector<DrawNote> notes, std::vector<DrawName> names) -> void;
    /** Draw the key signature after the clefs; accidentals then only where a note departs from it. */
    auto setShowKeySignature(bool show) -> void { showKeySignature = show; repaint(); }
    /** Larger chord names for an audience (the staff makes room above itself). */
    auto setTextScale(float scale) -> void { textScale = scale; repaint(); }
    auto paint(juce::Graphics &g) -> void override;

private:
    /**
     * An accidental (-2 double flat, -1 flat, 0 natural, 1 sharp, 2 double sharp) left of the
     * head: sharps, naturals and double sharps centred on the note's line / space, flats with
     * their bowl on it.
     */
    auto drawAccidental(juce::Graphics &g, int accidental, float headLeftX, float cy, float s) -> void;

    juce::Font musicFont;
    juce::Path sharpSign, flatSign, naturalSign, doubleSharpSign, doubleFlatSign; // at height 1
    juce::Path trebleClef, bassClef; // the clef outlines as the font draws them (scaled to the staff per frame)
    int32_t nowCs = 0;
    float textScale = 1.0f;
    bool showKeySignature = false;
    std::vector<DrawNote> notes;
    std::vector<DrawName> names;
    DrawKey key;
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
    juce::ToggleButton keySignatureToggle{"Key signature"};
    KeyDetector keyDetector;
    std::map<std::vector<int>, std::string> nameCache;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ScoreWidget)
};
