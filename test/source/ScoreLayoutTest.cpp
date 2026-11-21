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

TEST(ScoreLayout, notesSoundTogetherOnlyIfTheyOverlapEnough) {
    // overlap >= min(150 ms, half the shorter note)
    EXPECT_FALSE(soundTogether({0, 50, 60}, {40, 50, 62})) << "legato: 10 cs of 50 cs notes";
    EXPECT_FALSE(soundTogether({0, 60, 60}, {50, 60, 62})) << "100 ms hand-over";
    EXPECT_TRUE(soundTogether({0, 60, 60}, {30, 60, 62})) << "half of the shorter note";
    EXPECT_FALSE(soundTogether({0, 8, 60}, {5, 8, 62})) << "fast run: 30 ms of 80 ms";
    EXPECT_TRUE(soundTogether({0, 8, 60}, {4, 8, 62})) << "40 ms of 80 ms";
    EXPECT_FALSE(soundTogether({0, 200, 60}, {186, 200, 62})) << "under 150 ms of long notes";
    EXPECT_TRUE(soundTogether({0, 200, 60}, {185, 200, 62})) << "150 ms";
    EXPECT_TRUE(soundTogether({0, 300, 48}, {100, 6, 72})) << "a short note over a pedal";
    EXPECT_TRUE(soundTogether({0, 10, 60}, {0, 50, 64})) << "starting together";
    EXPECT_FALSE(soundTogether({0, 10, 60}, {10, 10, 64})) << "touching";
}

TEST(ScoreLayout, melodyIsTheTopOfEachOnsetGroupWithNoMinimumLength) {
    // A note 2 cs after its chord's bass joins the chord (the snap); its top note is melody.
    const auto top = skyline({{100, 100, 48}, {102, 98, 72}, {101, 99, 64}, // chord, top 72
                              {200, 50, 74},                                 // melody
                              {205, 5, 76}});                                // 50 ms, still melody
    ASSERT_EQ(top.size(), 3u);
    EXPECT_EQ(top[0].pitch, 72);
    EXPECT_EQ(top[1].pitch, 74);
    EXPECT_EQ(top[2].pitch, 76) << "above the held 74: the top at its onset";
    // short notes of a fast run are all melody
    EXPECT_EQ(skyline({{0, 8, 60}, {8, 8, 62}, {16, 8, 64}}).size(), 3u);
    // a legato line: overlaps under 150 ms do not hide the next note
    EXPECT_EQ(skyline({{0, 60, 64}, {50, 60, 62}, {100, 60, 60}}).size(), 3u);
}

TEST(ScoreLayout, aNoteUnderAHeldHigherNoteIsNotMelody) {
    // C5 held over a lower note's onset: the melody keeps it, drawn once; the lower note is bass.
    const auto lines = skylineAndBass({{100, 300, 72}, {200, 50, 60}});
    ASSERT_EQ(lines.melody.size(), 1u);
    EXPECT_EQ(lines.melody[0].pitch, 72);
    ASSERT_EQ(lines.bass.size(), 1u);
    EXPECT_EQ(lines.bass[0].pitch, 60);
}

TEST(ScoreLayout, theBassIsTheBottomOfEachGroupUnlessAHeldNoteIsBelow) {
    // An inner voice is in neither line.
    const auto lines = skylineAndBass({{0, 100, 48}, {0, 100, 60}, {0, 100, 72}});
    ASSERT_EQ(lines.melody.size(), 1u);
    ASSERT_EQ(lines.bass.size(), 1u);
    EXPECT_EQ(lines.melody[0].pitch, 72);
    EXPECT_EQ(lines.bass[0].pitch, 48);
    // A lone note is melody only.
    const auto lone = skylineAndBass({{0, 100, 60}});
    EXPECT_EQ(lone.melody.size(), 1u);
    EXPECT_TRUE(lone.bass.empty());
    // A held bass under new upper notes: the upper group has no bass of its own.
    const auto pedal = skylineAndBass({{0, 400, 36}, {0, 400, 60}, {100, 50, 64}, {100, 50, 67}});
    ASSERT_EQ(pedal.bass.size(), 1u);
    EXPECT_EQ(pedal.bass[0].pitch, 36);
    ASSERT_EQ(pedal.melody.size(), 2u);
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
    const auto addLine = [&](const std::vector<ScoreLayout::Note> &line, bool bass) {
        for (const auto &n: line) {
            const auto sp = spell(n.pitch, n.onsetCs == 1000 || n.onsetCs == 1080);
            notes.push_back({n.onsetCs, n.pitch, sp.step, sp.accidental, bass});
        }
    };
    addLine(lines.melody, false);
    addLine(lines.bass, true);
    notes.push_back({1120, 67, 38, 2, false}); // G4 spelled F double sharp: on the F4 space
    EXPECT_EQ(lines.melody.size(), 6u) << "one melody note per chord";
    view.setShowKeySignature(true);
    view.setFrame(1000, notes, names, {"Bb major", "G minor", false, -2});
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
