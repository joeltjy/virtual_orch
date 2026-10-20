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

// Draws a sample frame; with VIRTUAL_ORCH_SCORE_PNG=<path> it is also saved for a visual check.
TEST(ScoreView, rendersASampleFrame) {
    const juce::ScopedJuceInitialiser_GUI gui;
    ScoreView view;
    view.setBounds(0, 0, 960, 260);
    std::vector<ScoreView::DrawChord> chords;
    const auto add = [&](int32_t onset, std::vector<int> pitches, std::string name, bool flats) {
        ScoreView::DrawChord dc;
        dc.chord = {onset, std::move(pitches)};
        dc.name = std::move(name);
        dc.flats = flats;
        chords.push_back(dc);
    };
    add(700, {48, 64, 67, 72}, "C", false);
    add(820, {55, 59, 62, 65, 71}, "G7", false);
    add(960, {60, 61, 76}, "", false);          // a second (C, C#) and E5
    add(1000, {46, 62, 65, 70}, "Bb", true);    // at the nowbar
    add(1150, {51, 63, 67, 70, 74}, "Ebmaj7", true);
    add(1300, {40, 84, 86}, "", false);         // ledger lines both ways
    view.setFrame(1000, chords);
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
