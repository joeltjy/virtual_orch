#include "VirtualOrch/widgets/ScoreWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ChordNaming.h"
#include "VirtualOrch/ScoreHarmony.h"
#include "VirtualOrch/Spelling.h"
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
constexpr juce::juce_wchar kNatural = 0x266E;
constexpr juce::juce_wchar kDoubleSharp = 0x1D12A;
constexpr juce::juce_wchar kDoubleFlat = 0x1D12B;

/** Key-signature positions (staff steps) on the treble staff; the bass staff is 14 steps lower. */
constexpr int kSharpSteps[7] = {45, 42, 46, 43, 40, 44, 41}; // F5 C5 G5 D5 A4 E5 B4
constexpr int kFlatSteps[7] = {41, 44, 40, 43, 39, 42, 38};  // B4 E5 A4 D5 G4 C5 F4

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
    naturalSign = outline(kNatural);
    doubleSharpSign = outline(kDoubleSharp);
    doubleFlatSign = outline(kDoubleFlat);
}

auto ScoreView::setFrame(int32_t now, std::vector<DrawNote> nextNotes, std::vector<DrawName> nextNames, DrawKey nextKey)
    -> void {
    nowCs = now;
    notes = std::move(nextNotes);
    names = std::move(nextNames);
    key = std::move(nextKey);
    repaint();
}

auto ScoreView::setFrame(int32_t now, std::vector<DrawNote> nextNotes, std::vector<DrawName> nextNames) -> void {
    setFrame(now, std::move(nextNotes), std::move(nextNames), DrawKey{});
}

auto ScoreView::drawAccidental(juce::Graphics &g, int accidental, float headLeftX, float cy, float s) -> void {
    // Place by the glyph's own outline, not its text box: the music font's boxes are not centred
    // on the sign, which left sharps and flats off the note's line.
    const bool flat = accidental < 0;
    juce::Path sign = accidental == -2 ? doubleFlatSign
                      : accidental == -1 ? flatSign
                      : accidental == 0  ? naturalSign
                      : accidental == 1  ? sharpSign
                                         : doubleSharpSign;
    sign.applyTransform(juce::AffineTransform::scale(s * (accidental == 2 ? 4.4f : 3.4f)));
    const auto b = sign.getBounds();
    if (b.isEmpty())
        return;
    const float targetRight = headLeftX - s * 0.25f;
    const float dy = flat ? cy + s * 0.45f - b.getBottom() // the bowl sits on the note
                          : cy - b.getCentreY();           // centred on the note
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
    const int sigCount = showKeySignature ? std::clamp(key.signature, -7, 7) : 0;
    const float clefGutter = s * 3.2f;
    const float gutter = clefGutter + (sigCount != 0 ? s * (0.4f + 0.9f * static_cast<float>(std::abs(sigCount))) : 0.0f);
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
               juce::Rectangle<float>(2.0f, trebleTop - s * 2.4f, clefGutter, s * 8.0f),
               juce::Justification::centred, false);
    g.setFont(musicFont.withHeight(s * 3.6f));
    g.drawText(juce::String::charToString(kFClef),
               juce::Rectangle<float>(2.0f, bassTop - s * 0.9f, clefGutter, s * 4.4f),
               juce::Justification::centred, false);

    // Key signature after the clefs.
    for (int i = 0; i < std::abs(sigCount); ++i) {
        const int trebleStep = sigCount > 0 ? kSharpSteps[i] : kFlatSteps[i];
        const float x = clefGutter + s * (1.1f + 0.9f * static_cast<float>(i));
        g.setColour(kInk);
        drawAccidental(g, sigCount > 0 ? 1 : -1, x, yOf(trebleStep, true), s);
        drawAccidental(g, sigCount > 0 ? 1 : -1, x, yOf(trebleStep - 14, false), s);
    }

    g.saveState();
    g.reduceClipRegion(juce::Rectangle<float>(gutter, 0.0f, w - gutter, h).toNearestInt());

    // Skyline notes: one notehead each, with ledger lines and accidentals.
    for (const auto &n: notes) {
        const float x = xFor(n.onsetCs, nowCs, nowX, pxPerCs);
        if (x < gutter - s * 3.0f || x > w + s * 2.0f)
            continue;
        const bool past = n.onsetCs < nowCs;
        const bool treble = onTreble(n.pitch);
        const float cy = yOf(n.step, treble);
        g.setColour(kInk.withAlpha(past ? 0.45f : 0.8f));
        for (const int ls: ledgerSteps(n.step, treble ? kTrebleBottomStep : kBassBottomStep,
                                       treble ? kTrebleTopStep : kBassTopStep))
            g.drawLine(x - s * 0.95f, yOf(ls, treble), x + s * 0.95f, yOf(ls, treble), 1.2f);

        juce::Path head;
        head.addEllipse(-s * 0.62f, -s * 0.42f, s * 1.24f, s * 0.84f);
        head.applyTransform(juce::AffineTransform::rotation(-0.35f).translated(x, cy));
        g.setColour((n.bass ? kBassLine : kSkyline).withMultipliedAlpha(past ? 0.55f : 1.0f));
        g.fillPath(head);

        // With a key signature, only where the note departs from it (a natural cancels it).
        const int fromSignature = showKeySignature ? Spelling::signatureAccidental(Spelling::mod(n.step, 7), sigCount) : 0;
        if (n.accidental != fromSignature) {
            g.setColour(kInk.withMultipliedAlpha(past ? 0.55f : 1.0f));
            drawAccidental(g, n.accidental, x - s * 0.62f, cy, s);
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

    // The key, top left (greyed while uncertain; a close second in brackets).
    if (! key.name.empty()) {
        const auto keyFont = juce::Font(s * 1.1f * textScale);
        auto text = juce::String(key.name);
        if (! key.runnerUp.empty())
            text << " (" << juce::String(key.runnerUp) << ")";
        const float textW = keyFont.getStringWidthFloat(text) + s * 0.8f;
        const juce::Rectangle<float> box(2.0f, s * 0.4f, textW, s * nameRows);
        g.setColour(UiConstants::pianoRollBackground.withAlpha(0.85f));
        g.fillRect(box);
        g.setFont(keyFont);
        g.setColour(kInk.withAlpha(key.uncertain ? 0.45f : 0.95f));
        g.drawText(text, box.reduced(s * 0.3f, 0.0f), juce::Justification::centredLeft, false);
    }

    // Nowbar.
    g.setColour(kNowbar.withAlpha(0.8f));
    g.fillRect(juce::Rectangle<float>(nowX - 1.0f, trebleTop - s * 2.5f, 2.0f, bassTop + s * 6.5f - (trebleTop - s * 2.5f)));
}

ScoreWidget::ScoreWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceScoreTitle, UiConstants::workspaceWidgetTypeScore),
      session(sessionIn) {
    getContentComponent().addAndMakeVisible(view);
    keySignatureToggle.setToggleState(false, juce::dontSendNotification);
    keySignatureToggle.onClick = [this] { view.setShowKeySignature(keySignatureToggle.getToggleState()); };
    getContentComponent().addAndMakeVisible(keySignatureToggle);
    startTimerHz(30);
}

ScoreWidget::~ScoreWidget() {
    stopTimer();
}

auto ScoreWidget::resized() -> void {
    WorkspaceWidget::resized();
    const auto area = getContentComponent().getLocalBounds();
    view.setBounds(area);
    keySignatureToggle.setBounds(area.getRight() - 130, area.getY() + 2, 126, 22);
    keySignatureToggle.toFront(false);
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
    // Chord labels show only changes, so they are worked out from kLookBackCs before the window:
    // a name stays put while earlier chords scroll out.
    constexpr int32_t kLookBackCs = 600;
    const auto now = static_cast<int32_t>(session.clock.getTime());
    const int32_t from = now - ScoreView::kWindowCs - kLookBackCs;
    std::vector<ScoreLayout::Note> generated;
    for (const auto &t: session.activeReduction().getOutputHistorySince(from)) {
        if (t.note < static_cast<int32_t>(Vocab::NoteOffset) || t.note >= static_cast<int32_t>(Vocab::Rest))
            continue; // ClearQueue and other markers
        if (t.time < from || t.time > now + ScoreView::kWindowCs + 100)
            continue;
        generated.push_back({t.time, t.getRealDuration(), t.getPitch()});
    }

    // The key, from what has started by now: the orchestra's notes and the piano's.
    std::vector<ScoreLayout::Note> heard;
    for (const auto &n: generated)
        if (n.onsetCs <= now)
            heard.push_back(n);
    {
        const auto snap = session.orchestrationTransformer.getDebugSnapshotSince(now - KeyDetector::kMaxAgeCs - 300);
        for (const auto *tokens: {&snap.midiHistory, &snap.midiPending})
            for (const auto &t: *tokens)
                if (t.time <= now && t.note >= static_cast<int32_t>(Vocab::NoteOffset)
                    && t.note < static_cast<int32_t>(Vocab::Rest))
                    heard.push_back({t.time, std::max(1, t.getRealDuration()), t.getPitch()});
    }
    heard = ScoreLayout::prepare(std::move(heard));
    const auto lowest = ScoreLayout::lowestAtOnset(heard);
    std::vector<KeyDetector::Note> keyNotes;
    keyNotes.reserve(heard.size());
    for (size_t i = 0; i < heard.size(); ++i)
        keyNotes.push_back({heard[i].onsetCs, heard[i].durationCs, heard[i].pitch, lowest[i]});
    const auto &found = keyDetector.update(keyNotes, now);
    const Spelling::Key key = found.key.value_or(Spelling::Key{}); // C major until one is known

    // Chords and their labels, then the melody and bass spelled in the key.
    const auto prepared = ScoreLayout::prepare(generated);
    const auto chords = ScoreHarmony::chords(prepared, key, [this](const std::vector<int> &p) { return nameFor(p); });
    std::vector<ScoreView::DrawName> names;
    for (const auto &l: ScoreHarmony::labels(chords))
        names.push_back({l.onsetCs, l.text});
    std::vector<ScoreView::DrawNote> notes;
    const auto lines = ScoreLayout::skylineAndBass(generated);
    for (const bool bass: {true, false}) {
        const auto &line = bass ? lines.bass : lines.melody;
        const auto spelled = ScoreHarmony::spellLine(line, chords, key);
        for (size_t i = 0; i < line.size(); ++i)
            notes.push_back({line[i].onsetCs, line[i].pitch, Spelling::staffStep(line[i].pitch, spelled[i]),
                             Spelling::accidentalOf(spelled[i]), bass});
    }
    ScoreView::DrawKey drawKey;
    if (found.key.has_value()) {
        drawKey.name = KeyDetector::name(*found.key);
        drawKey.signature = Spelling::signature(*found.key);
    }
    if (found.runnerUp.has_value())
        drawKey.runnerUp = KeyDetector::name(*found.runnerUp);
    drawKey.uncertain = found.uncertain;
    view.setFrame(now, std::move(notes), std::move(names), std::move(drawKey));
}
