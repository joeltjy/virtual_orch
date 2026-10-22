#include "VirtualOrch/widgets/ScoreWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ChordNaming.h"
#include "VirtualOrch/ui/UiConstants.h"
#include "VirtualOrchBinaryData.h"

namespace {

using namespace ScoreLayout;

const juce::Colour kInk{juce::Colours::lightgrey};
const juce::Colour kSkyline{0xffffb347}; // melody notes
const juce::Colour kBassLine{0xff81c784}; // bass notes
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
    // The sign outlines at height 1, built once (scaled per frame).
    const auto outline = [&](juce::juce_wchar c) {
        juce::GlyphArrangement glyphs;
        glyphs.addLineOfText(musicFont.withHeight(100.0f), juce::String::charToString(c), 0.0f, 0.0f);
        juce::Path path;
        glyphs.createPath(path);
        path.applyTransform(juce::AffineTransform::scale(0.01f));
        return path;
    };
    sharpSign = outline(kSharp);
    flatSign = outline(kFlat);
}

auto ScoreView::setFrame(int32_t now, std::vector<DrawNote> nextNotes, std::vector<DrawName> nextNames) -> void {
    nowCs = now;
    notes = std::move(nextNotes);
    names = std::move(nextNames);
    repaint();
}

auto ScoreView::drawAccidental(juce::Graphics &g, bool sharp, float headLeftX, float cy, float s) -> void {
    // Place by the glyph's own outline, not its text box: the music font's boxes are not centred
    // on the sign, which left sharps and flats off the note's line.
    juce::Path sign = sharp ? sharpSign : flatSign;
    sign.applyTransform(juce::AffineTransform::scale(s * 3.4f));
    const auto b = sign.getBounds();
    if (b.isEmpty())
        return;
    const float targetRight = headLeftX - s * 0.25f;
    const float dy = sharp ? cy - b.getCentreY()          // the two bars straddle the note
                           : cy + s * 0.45f - b.getBottom(); // the bowl sits on the note
    sign.applyTransform(juce::AffineTransform::translation(targetRight - b.getRight(), dy));
    g.fillPath(sign);
}

auto ScoreView::paint(juce::Graphics &g) -> void {
    g.fillAll(UiConstants::pianoRollBackground);
    const float w = static_cast<float>(getWidth());
    const float h = static_cast<float>(getHeight());
    if (w < 40.0f || h < 40.0f)
        return;

    // Vertical layout in staff spaces s: name row, room for ledgers, treble, 4 s gap, bass.
    const float nameRows = 1.8f * textScale; // the chord-name row, in staff spaces
    const float s = std::max(4.0f, h / (19.2f + nameRows));
    const float trebleTop = s * (3.7f + nameRows); // F5 line
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

    // Skyline notes: one notehead each, with ledger lines and accidentals.
    for (const auto &n: notes) {
        const float x = xFor(n.onsetCs, nowCs, nowX, pxPerCs);
        if (x < gutter - s * 3.0f || x > w + s * 2.0f)
            continue;
        const bool past = n.onsetCs < nowCs;
        const bool treble = onTreble(n.pitch);
        const auto sp = spell(n.pitch, n.flats);
        const float cy = yOf(sp.step, treble);
        g.setColour(kInk.withAlpha(past ? 0.45f : 0.8f));
        for (const int ls: ledgerSteps(sp.step, treble ? kTrebleBottomStep : kBassBottomStep,
                                       treble ? kTrebleTopStep : kBassTopStep))
            g.drawLine(x - s * 0.95f, yOf(ls, treble), x + s * 0.95f, yOf(ls, treble), 1.2f);

        juce::Path head;
        head.addEllipse(-s * 0.62f, -s * 0.42f, s * 1.24f, s * 0.84f);
        head.applyTransform(juce::AffineTransform::rotation(-0.35f).translated(x, cy));
        g.setColour((n.bass ? kBassLine : kSkyline).withMultipliedAlpha(past ? 0.55f : 1.0f));
        g.fillPath(head);

        if (sp.accidental != 0) {
            g.setColour(kInk.withMultipliedAlpha(past ? 0.55f : 1.0f));
            drawAccidental(g, sp.accidental > 0, x - s * 0.62f, cy, s);
        }
    }

    // Chord names above the treble staff.
    const auto nameFont = juce::Font(s * 1.25f * textScale, juce::Font::bold);
    float lastNameRight = -1.0e9f;
    for (const auto &nm: names) {
        const float x = xFor(nm.onsetCs, nowCs, nowX, pxPerCs);
        if (nm.name.empty() || x < gutter - s * 3.0f || x > w + s * 2.0f || x - s * 0.5f <= lastNameRight)
            continue;
        g.setFont(nameFont);
        const auto text = juce::String(nm.name);
        const float textW = nameFont.getStringWidthFloat(text) + s * 0.6f;
        g.setColour(kInk.withMultipliedAlpha(nm.onsetCs < nowCs ? 0.6f : 1.0f));
        g.drawText(text, juce::Rectangle<float>(x - s * 0.5f, s * 0.6f, textW, s * nameRows),
                   juce::Justification::centredLeft, false);
        lastNameRight = x - s * 0.5f + textW;
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
    std::vector<ScoreLayout::Note> generated;
    for (const auto &t: session.activeReduction().getOutputHistorySince(now - ScoreView::kWindowCs)) {
        if (t.note < static_cast<int32_t>(Vocab::NoteOffset) || t.note >= static_cast<int32_t>(Vocab::Rest))
            continue; // ClearQueue and other markers
        if (t.time < now - ScoreView::kWindowCs - 100 || t.time > now + ScoreView::kWindowCs + 100)
            continue;
        generated.push_back({t.time, t.getRealDuration(), t.getPitch()});
    }
    // Names (and their flat / sharp spelling) from the chords after the 30 ms snap.
    std::vector<ScoreView::DrawName> names;
    std::map<int32_t, bool> flatsAt;
    for (auto &c: ScoreLayout::groupChords(ScoreLayout::snapOnsets(generated))) {
        const auto &name = nameFor(c.pitches);
        flatsAt[c.onsetCs] = ChordNaming::rootIsFlat(name);
        names.push_back({c.onsetCs, ChordNaming::basicChord(name)}); // root + quality only
    }
    std::vector<ScoreView::DrawNote> notes;
    const auto lines = ScoreLayout::skylineAndBass(std::move(generated));
    for (const bool bass: {true, false})
        for (const auto &n: bass ? lines.bass : lines.melody) {
            // the chord whose onset is the nearest at or before the note's
            auto it = flatsAt.upper_bound(n.onsetCs);
            const bool flats = it != flatsAt.begin() && std::prev(it)->second;
            notes.push_back({n.onsetCs, n.pitch, flats, bass});
        }
    view.setFrame(now, std::move(notes), std::move(names));
}
