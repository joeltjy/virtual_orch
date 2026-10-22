#include <gtest/gtest.h>

#include <cstdlib>

#include <VirtualOrch/ScoreLayout.h>
#include <VirtualOrch/widgets/ScoreWidget.h>

namespace virtual_orch_test {

using namespace ScoreLayout;

TEST(ScoreLayout, chordsAreNotesWithin50msChainedByOnset) {
    const auto chords = groupChords({{100, 50, 60}, {104, 50, 64}, {108, 50, 67}, // chained
                                     {120, 50, 72},                               // 12 cs later
                                     {200, 50, 48}, {200, 50, 48}});              // duplicate
    ASSERT_EQ(chords.size(), 3u);
    EXPECT_EQ(chords[0].onsetCs, 100);
    EXPECT_EQ(chords[0].pitches, (std::vector<int>{60, 64, 67}));
    EXPECT_EQ(chords[0].topPitch(), 67) << "skyline note";
    EXPECT_EQ(chords[1].pitches, (std::vector<int>{72}));
    EXPECT_EQ(chords[2].pitches, (std::vector<int>{48}));
}

TEST(ScoreLayout, stepsMatchTheStaffLines) {
    EXPECT_EQ(spell(64, false).step, kTrebleBottomStep) << "E4";
    EXPECT_EQ(spell(77, false).step, kTrebleTopStep) << "F5";
    EXPECT_EQ(spell(43, false).step, kBassBottomStep) << "G2";
    EXPECT_EQ(spell(57, false).step, kBassTopStep) << "A3";
    EXPECT_TRUE(onTreble(60));
    EXPECT_FALSE(onTreble(59));
}

TEST(ScoreLayout, sharpsOrFlats) {
    EXPECT_EQ(spell(61, false).step, spell(60, false).step) << "C# on the C step";
    EXPECT_EQ(spell(61, false).accidental, 1);
    EXPECT_EQ(spell(61, true).step, spell(62, true).step) << "Db on the D step";
    EXPECT_EQ(spell(61, true).accidental, -1);
    EXPECT_EQ(spell(62, true).accidental, 0);
}

TEST(ScoreLayout, ledgerLines) {
    EXPECT_EQ(ledgerSteps(spell(60, false).step, kTrebleBottomStep, kTrebleTopStep),
              (std::vector<int>{35})) << "middle C: one ledger below the treble staff";
    EXPECT_EQ(ledgerSteps(spell(84, false).step, kTrebleBottomStep, kTrebleTopStep),
              (std::vector<int>{47, 49})) << "C6: two above";
    EXPECT_TRUE(ledgerSteps(spell(67, false).step, kTrebleBottomStep, kTrebleTopStep).empty());
    EXPECT_EQ(ledgerSteps(spell(40, false).step, kBassBottomStep, kBassTopStep),
              (std::vector<int>{23})) << "E2: one below the bass staff";
}

TEST(ScoreLayout, secondsAreShiftedAlternately) {
    // C D E (steps 35 36 37): D shifted, E not (its lower neighbour D is shifted).
    EXPECT_EQ(secondShifts({35, 36, 37}), (std::vector<bool>{false, true, false}));
    EXPECT_EQ(secondShifts({35, 37, 39}), (std::vector<bool>{false, false, false}));
}

TEST(ScoreLayout, nowbarMapping) {
    EXPECT_FLOAT_EQ(xFor(1000, 1000, 300.0f, 0.5f), 300.0f);
    EXPECT_FLOAT_EQ(xFor(1200, 1000, 300.0f, 0.5f), 400.0f) << "ahead of the clock: right";
    EXPECT_FLOAT_EQ(xFor(800, 1000, 300.0f, 0.5f), 200.0f);
}

TEST(ScoreLayout, onsetsWithin30msAreSnappedToTheFirst) {
    const auto snapped = snapOnsets({{100, 50, 60}, {102, 50, 64}, {105, 50, 67}, // chained, <= 3 cs apart
                                     {109, 50, 72}});                             // 4 cs after: its own
    ASSERT_EQ(snapped.size(), 4u);
    EXPECT_EQ(snapped[0].onsetCs, 100);
    EXPECT_EQ(snapped[1].onsetCs, 100);
    EXPECT_EQ(snapped[2].onsetCs, 100);
    EXPECT_EQ(snapped[2].durationCs, 55) << "keeps its end time";
    EXPECT_EQ(snapped[3].onsetCs, 109);
}

TEST(ScoreLayout, melodyIsEveryNoteUncoveredAboveForMoreThan70Percent) {
    // A melody note 2 cs after its chord's bass joins the chord (the snap); the top note is melody.
    const auto top = skyline({{100, 100, 48}, {102, 98, 72}, {101, 99, 64}, // chord, top 72
                              {200, 50, 74},                                 // melody
                              {200, 200, 50}});                              // 74 covers only 25%
    ASSERT_EQ(top.size(), 3u);
    EXPECT_EQ(top[0].pitch, 72);
    EXPECT_EQ(top[1].pitch, 50) << "nothing above it for 75% of its length";
    EXPECT_EQ(top[2].pitch, 74);
    // 29% covered: melody (71% uncovered); exactly 30%: not (70% is not more than 70%).
    EXPECT_EQ(skyline({{0, 100, 60}, {71, 29, 72}}).size(), 2u);
    EXPECT_EQ(skyline({{0, 100, 60}, {70, 30, 72}}).size(), 1u);
    const auto covered = skyline({{0, 100, 60}, {60, 40, 72}});
    ASSERT_EQ(covered.size(), 1u);
    EXPECT_EQ(covered[0].pitch, 72);
}

TEST(ScoreLayout, theBassIsUncoveredBelowAndNotAlreadyMelody) {
    // An inner voice covered above and below is in neither line.
    const auto lines = skylineAndBass({{0, 100, 48}, {0, 100, 60}, {0, 100, 72}});
    ASSERT_EQ(lines.melody.size(), 1u);
    ASSERT_EQ(lines.bass.size(), 1u);
    EXPECT_EQ(lines.melody[0].pitch, 72);
    EXPECT_EQ(lines.bass[0].pitch, 48);
    // A lone note is melody only.
    const auto lone = skylineAndBass({{0, 100, 60}});
    EXPECT_EQ(lone.melody.size(), 1u);
    EXPECT_TRUE(lone.bass.empty());
}

TEST(ScoreLayout, theBassLineIsKeptApartFromTheMelody) {
    const auto lines = skylineAndBass({{100, 50, 48}, {101, 50, 64}, {102, 50, 72}, // C E C
                                       {200, 50, 43}, {200, 50, 74}});             // G D
    ASSERT_EQ(lines.melody.size(), 2u);
    ASSERT_EQ(lines.bass.size(), 2u);
    EXPECT_EQ(lines.melody[0].pitch, 72);
    EXPECT_EQ(lines.bass[0].pitch, 48);
    EXPECT_EQ(lines.bass[0].onsetCs, 100) << "snapped";
    EXPECT_EQ(lines.bass[1].pitch, 43);
}

TEST(ScoreLayout, aHeldTopNoteIsShownOnce) {
    // C5 held over a lower note's onset: the skyline keeps it, and it is drawn once.
    const auto top = skyline({{100, 300, 72}, {200, 50, 60}});
    ASSERT_EQ(top.size(), 1u);
    EXPECT_EQ(top[0].pitch, 72);
}

// Draws a sample frame; with VIRTUAL_ORCH_SCORE_PNG=<path> it is also saved for a visual check.
TEST(ScoreView, rendersASampleFrame) {
    const juce::ScopedJuceInitialiser_GUI gui;
    ScoreView view;
    view.setBounds(0, 0, 960, 260);
    std::vector<ScoreLayout::Note> generated;
    std::vector<ScoreView::DrawName> names;
    const auto add = [&](int32_t onset, std::vector<int> pitches, std::string name) {
        for (const int p: pitches)
            generated.push_back({onset, 30, p}); // short, so no top note is held into the next chord
        names.push_back({onset, std::move(name)});
    };
    add(840, {48, 64, 67, 72}, "C");
    add(900, {55, 59, 62, 65, 71}, "G7");
    add(960, {60, 61, 73}, "");                 // C#5 on top
    add(1000, {46, 62, 65, 70}, "Bb");          // at the nowbar
    add(1080, {51, 63, 67, 70, 75}, "Ebmaj7");  // Eb5 on top, drawn flat
    add(1160, {40, 84, 86}, "");                // ledger lines above
    std::vector<ScoreView::DrawNote> notes;
    const auto lines = skylineAndBass(generated);
    for (const auto &n: lines.melody)
        notes.push_back({n.onsetCs, n.pitch, n.onsetCs == 1000 || n.onsetCs == 1080, false});
    for (const auto &n: lines.bass)
        notes.push_back({n.onsetCs, n.pitch, n.onsetCs == 1000 || n.onsetCs == 1080, true});
    EXPECT_EQ(lines.melody.size(), 6u) << "one melody note per chord";
    view.setFrame(1000, notes, names);
    const auto image = view.createComponentSnapshot(view.getLocalBounds(), false);
    ASSERT_EQ(image.getWidth(), 960);

    if (const char *path = std::getenv("VIRTUAL_ORCH_SCORE_PNG")) {
        juce::File file(path);
        file.deleteFile();
        juce::FileOutputStream out(file);
        juce::PNGImageFormat().writeImageToStream(image, out);
    }
}

} // namespace virtual_orch_test
