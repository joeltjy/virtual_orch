#include "VirtualOrch/widgets/ScoreWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ChordNaming.h"
#include "VirtualOrch/ui/UiConstants.h"
#include "VirtualOrchBinaryData.h"

namespace {

using namespace ScoreLayout;

const juce::Colour kInk{juce::Colours::lightgrey};
const juce::Colour kSkyline{0xffffb347}; // highlighted top note
const juce::Colour kNowbar{0xff4fc3f7};

constexpr juce::juce_wchar kGClef = 0x1D11E;
constexpr juce::juce_wchar kFClef = 0x1D122;
constexpr juce::juce_wchar kSharp = 0x266F;
constexpr juce::juce_wchar kFlat = 0x266D;

auto musicTypeface() -> juce::Typeface::Ptr {
    static const auto typeface = juce::Typeface::createSystemTypefaceFor(
        VirtualOrchBinaryData::NotoMusicRegular_ttf,
        static_cast<size_t>(VirtualOrchBinaryData::NotoMusicRegular_ttfSize));
    return typeface;
}

} // namespace

ScoreView::ScoreView() : musicFont(musicTypeface()) {
    setOpaque(true);
}

auto ScoreView::setFrame(int32_t now, std::vector<DrawChord> next) -> void {
    nowCs = now;
    chords = std::move(next);
    repaint();
}

auto ScoreView::paint(juce::Graphics &g) -> void {
    g.fillAll(UiConstants::pianoRollBackground);
    const float w = static_cast<float>(getWidth());
    const float h = static_cast<float>(getHeight());
    if (w < 40.0f || h < 40.0f)
        return;

    // Vertical layout in staff spaces s: name row, room for ledgers, treble, 4 s gap, bass.
    const float s = std::max(4.0f, h / 21.0f);
    const float trebleTop = s * 5.5f;          // F5 line
    const float bassTop = trebleTop + s * 8.0f; // A3 line
    const float gutter = s * 3.2f;              // clefs
    const float nowX = gutter + (w - gutter) * 0.5f;
    const float pxPerCs = (w - gutter) * 0.5f / static_cast<float>(kWindowCs);

    const auto yOf = [&](int step, bool treble) {
        return treble ? trebleTop + static_cast<float>(kTrebleTopStep - step) * s * 0.5f
                      : bassTop + static_cast<float>(kBassTopStep - step) * s * 0.5f;
    };

    // Staves and clefs.
    g.setColour(kInk.withAlpha(0.55f));
    for (int i = 0; i < 5; ++i) {
        g.drawHorizontalLine(juce::roundToInt(trebleTop + s * static_cast<float>(i)), 2.0f, w);
        g.drawHorizontalLine(juce::roundToInt(bassTop + s * static_cast<float>(i)), 2.0f, w);
    }
    g.setColour(kInk);
    g.setFont(musicFont.withHeight(s * 6.6f));
    g.drawText(juce::String::charToString(kGClef),
               juce::Rectangle<float>(2.0f, trebleTop - s * 2.4f, gutter, s * 8.0f),
               juce::Justification::centred, false);
    g.setFont(musicFont.withHeight(s * 3.6f));
    g.drawText(juce::String::charToString(kFClef),
               juce::Rectangle<float>(2.0f, bassTop - s * 0.9f, gutter, s * 4.4f),
               juce::Justification::centred, false);

    g.saveState();
    g.reduceClipRegion(juce::Rectangle<float>(gutter, 0.0f, w - gutter, h).toNearestInt());

    // Chords: noteheads, ledger lines, accidentals, names.
    const auto nameFont = juce::Font(s * 1.25f, juce::Font::bold);
    float lastNameRight = -1.0e9f;
    for (const auto &dc: chords) {
        const float x = xFor(dc.chord.onsetCs, nowCs, nowX, pxPerCs);
        if (x < gutter - s * 3.0f || x > w + s * 2.0f)
            continue;
        const bool past = dc.chord.onsetCs < nowCs;
        const int top = dc.chord.topPitch();

        for (const bool treble: {true, false}) {
            std::vector<int> pitches;
            for (const int p: dc.chord.pitches)
                if (onTreble(p) == treble)
                    pitches.push_back(p);
            std::vector<int> steps;
            std::vector<int> accidentals;
            for (const int p: pitches) {
                const auto sp = spell(p, dc.flats);
                steps.push_back(sp.step);
                accidentals.push_back(sp.accidental);
            }
            const auto shifted = secondShifts(steps);
            const int bottom = treble ? kTrebleBottomStep : kBassBottomStep;
            const int topLine = treble ? kTrebleTopStep : kBassTopStep;

            for (size_t i = 0; i < pitches.size(); ++i) {
                const float cx = x + (shifted[i] ? s * 1.15f : 0.0f);
                const float cy = yOf(steps[i], treble);
                g.setColour(kInk.withAlpha(past ? 0.45f : 0.8f));
                for (const int ls: ledgerSteps(steps[i], bottom, topLine))
                    g.drawLine(cx - s * 0.95f, yOf(ls, treble), cx + s * 0.95f, yOf(ls, treble), 1.2f);

                juce::Path head;
                head.addEllipse(-s * 0.62f, -s * 0.42f, s * 1.24f, s * 0.84f);
                head.applyTransform(juce::AffineTransform::rotation(-0.35f).translated(cx, cy));
                const auto colour = pitches[i] == top ? kSkyline : kInk.withAlpha(0.85f);
                g.setColour(colour.withMultipliedAlpha(past ? 0.55f : 1.0f));
                g.fillPath(head);

                if (accidentals[i] != 0) {
                    g.setFont(musicFont.withHeight(s * 3.4f));
                    g.drawText(juce::String::charToString(accidentals[i] > 0 ? kSharp : kFlat),
                               juce::Rectangle<float>(cx - s * 2.5f, cy - s * 2.2f, s * 1.6f, s * 4.0f),
                               juce::Justification::centred, false);
                }
            }
        }

        if (! dc.name.empty() && x - s * 0.5f > lastNameRight) {
            g.setFont(nameFont);
            const auto text = juce::String(dc.name);
            const float textW = nameFont.getStringWidthFloat(text) + s * 0.6f;
            g.setColour(kInk.withMultipliedAlpha(past ? 0.6f : 1.0f));
            g.drawText(text, juce::Rectangle<float>(x - s * 0.5f, s * 0.6f, textW, s * 1.8f),
                       juce::Justification::centredLeft, false);
            lastNameRight = x - s * 0.5f + textW;
        }
    }
    g.restoreState();

    // Nowbar.
    g.setColour(kNowbar.withAlpha(0.8f));
    g.fillRect(juce::Rectangle<float>(nowX - 1.0f, trebleTop - s * 2.5f, 2.0f, bassTop + s * 6.5f - (trebleTop - s * 2.5f)));
}

ScoreWidget::ScoreWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceScoreTitle, UiConstants::workspaceWidgetTypeScore),
      session(sessionIn) {
    getContentComponent().addAndMakeVisible(view);
    startTimerHz(30);
}

ScoreWidget::~ScoreWidget() {
    stopTimer();
}

auto ScoreWidget::resized() -> void {
    WorkspaceWidget::resized();
    view.setBounds(getContentComponent().getLocalBounds());
}

auto ScoreWidget::nameFor(const std::vector<int> &pitches) -> const std::string & {
    if (nameCache.size() > 1024)
        nameCache.clear();
    const auto it = nameCache.find(pitches);
    if (it != nameCache.end())
        return it->second;
    return nameCache.emplace(pitches, ChordNaming::nameFor(pitches)).first->second;
}

auto ScoreWidget::timerCallback() -> void {
    const auto now = static_cast<int32_t>(session.clock.getTime());
    std::vector<ScoreLayout::Note> notes;
    for (const auto &t: session.activeReduction().getOutputHistorySince(now - ScoreView::kWindowCs)) {
        if (t.note < static_cast<int32_t>(Vocab::NoteOffset) || t.note >= static_cast<int32_t>(Vocab::Rest))
            continue; // ClearQueue and other markers
        if (t.time < now - ScoreView::kWindowCs - 100 || t.time > now + ScoreView::kWindowCs + 100)
            continue;
        notes.push_back({t.time, t.getRealDuration(), t.getPitch()});
    }
    std::vector<ScoreView::DrawChord> chords;
    for (auto &c: ScoreLayout::groupChords(std::move(notes))) {
        ScoreView::DrawChord dc;
        dc.name = nameFor(c.pitches);
        dc.flats = ChordNaming::rootIsFlat(dc.name);
        dc.chord = std::move(c);
        chords.push_back(std::move(dc));
    }
    view.setFrame(now, std::move(chords));
}
