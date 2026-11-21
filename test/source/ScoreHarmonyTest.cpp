#include <gtest/gtest.h>

#include <VirtualOrch/KeyDetector.h>
#include <VirtualOrch/ScoreHarmony.h>
#include <VirtualOrch/Spelling.h>

#include <vector>

namespace virtual_orch_test {

namespace {

using Spelling::Key;

const Key kC{0, false};
const Key kF{-1, false};
const Key kAMinor{3, true};
const Key kGSharpMinor{8, true};
const Key kDb{-5, false};

auto names(const std::vector<ScoreHarmony::Label> &labels) -> std::vector<std::string> {
    std::vector<std::string> out;
    for (const auto &l: labels)
        out.push_back(l.text);
    return out;
}

auto labelsOf(std::vector<ScoreLayout::Note> notes, const Key &key = kC) -> std::vector<std::string> {
    return names(ScoreHarmony::labels(ScoreHarmony::chords(ScoreLayout::prepare(std::move(notes)), key)));
}

/** Notes of a scale, `stepCs` apart, each `durCs` long. */
auto line(std::vector<int> pitches, int32_t start, int32_t stepCs, int32_t durCs) -> std::vector<KeyDetector::Note> {
    std::vector<KeyDetector::Note> out;
    for (const int p: pitches) {
        out.push_back({start, durCs, p, false});
        start += stepCs;
    }
    return out;
}

} // namespace

// ---- Spelling

TEST(Spelling, namesOnTheLineOfFifths) {
    EXPECT_EQ(Spelling::nameOf(0), "C");
    EXPECT_EQ(Spelling::nameOf(6), "F#");
    EXPECT_EQ(Spelling::nameOf(-2), "Bb");
    EXPECT_EQ(Spelling::nameOf(13), "Fx");
    EXPECT_EQ(Spelling::nameOf(-10), "Ebb");
}

TEST(Spelling, theKeysWindowSpellsEveryPitchClass) {
    // C major: Db .. F#
    EXPECT_EQ(Spelling::nameOf(Spelling::inKey(1, kC)), "Db");
    EXPECT_EQ(Spelling::nameOf(Spelling::inKey(6, kC)), "F#");
    EXPECT_EQ(Spelling::nameOf(Spelling::inKey(8, kC)), "Ab");
    // A minor: Bb .. D#
    EXPECT_EQ(Spelling::nameOf(Spelling::inKey(10, kAMinor)), "Bb");
    EXPECT_EQ(Spelling::nameOf(Spelling::inKey(8, kAMinor)), "G#");
    EXPECT_EQ(Spelling::nameOf(Spelling::inKey(3, kAMinor)), "D#");
    // G# minor: the leading tone is F double sharp
    EXPECT_EQ(Spelling::nameOf(Spelling::inKey(7, kGSharpMinor)), "Fx");
    // Db major: Gb, not F#
    EXPECT_EQ(Spelling::nameOf(Spelling::inKey(6, kDb)), "Gb");
}

TEST(Spelling, aDiminishedChordOutsideTheKeyLeadsUp) {
    EXPECT_EQ(Spelling::nameOf(Spelling::root(1, kC, true)), "C#") << "C#dim7 -> D";
    EXPECT_EQ(Spelling::nameOf(Spelling::root(3, kC, true)), "D#") << "D#dim -> E";
    EXPECT_EQ(Spelling::nameOf(Spelling::root(1, kC, false)), "Db") << "Db major (Neapolitan)";
}

TEST(Spelling, chordTonesBySpelledInterval) {
    const Spelling::Quality plain{};
    EXPECT_EQ(Spelling::nameOf(*Spelling::chordTone(8, 4, plain)), "G#") << "the third of E";
    EXPECT_EQ(Spelling::nameOf(*Spelling::chordTone(10, 1, plain)), "Bb") << "the third of G minor";
    EXPECT_EQ(Spelling::nameOf(*Spelling::chordTone(7, 9, plain)), "Fx") << "the third of D#";
    const auto dim7 = Spelling::qualityOf("dim7");
    EXPECT_EQ(Spelling::nameOf(*Spelling::chordTone(10, 7, dim7)), "Bb") << "C#dim7's seventh";
    EXPECT_EQ(Spelling::nameOf(*Spelling::chordTone(7, 7, dim7)), "G");
    EXPECT_EQ(Spelling::nameOf(*Spelling::chordTone(8, 0, Spelling::qualityOf("aug"))), "G#");
}

TEST(Spelling, chromaticStepsFollowTheirDirection) {
    EXPECT_EQ(Spelling::nameOf(Spelling::passing(61, 60, kC)), "C#") << "rising";
    EXPECT_EQ(Spelling::nameOf(Spelling::passing(61, 62, kC)), "Db") << "falling";
    EXPECT_EQ(Spelling::nameOf(Spelling::passing(63, 62, kC)), "D#");
    EXPECT_EQ(Spelling::nameOf(Spelling::passing(63, 64, kC)), "Eb");
    EXPECT_EQ(Spelling::nameOf(Spelling::passing(64, 63, kC)), "E") << "in the key";
    EXPECT_EQ(Spelling::nameOf(Spelling::passing(66, std::nullopt, kC)), "F#") << "no step: the key";
}

TEST(Spelling, staffPositionsAndSignatures) {
    EXPECT_EQ(Spelling::staffStep(67, 13), 38) << "F double sharp 4 sits on F4";
    EXPECT_EQ(Spelling::staffStep(60, 12), 34) << "B#3";
    EXPECT_EQ(Spelling::staffStep(59, -7), 35) << "Cb4";
    EXPECT_EQ(Spelling::signature(kGSharpMinor), 5);
    EXPECT_EQ(Spelling::signature(Key{-2, false}), -2);
    EXPECT_EQ(Spelling::signatureAccidental(3, 5), 1) << "F# in five sharps";
    EXPECT_EQ(Spelling::signatureAccidental(6, 5), 0) << "B natural in five sharps";
    EXPECT_EQ(Spelling::signatureAccidental(6, -2), -1) << "Bb in two flats";
    EXPECT_EQ(Spelling::signatureAccidental(5, -2), 0) << "A natural in two flats";
}

// ---- KeyDetector

TEST(KeyDetector, findsMajorAndHarmonicMinor) {
    const auto cMajor = KeyDetector::window(line({60, 62, 64, 65, 67, 69, 71, 72, 67, 64, 60, 48}, 0, 25, 25), 300);
    const auto bestC = KeyDetector::rank(cMajor, 300).front();
    EXPECT_EQ(bestC.tonicPc, 0);
    EXPECT_FALSE(bestC.minor);

    auto aMinor = line({57, 60, 64, 68, 69, 71, 72, 74, 76, 68, 69, 57}, 0, 25, 25);
    aMinor.back().bass = true;
    const auto bestA = KeyDetector::rank(KeyDetector::window(aMinor, 300), 300).front();
    EXPECT_EQ(bestA.tonicPc, 9);
    EXPECT_TRUE(bestA.minor) << "G# and a tonic bass: A minor, not C major";
}

TEST(KeyDetector, needsEnoughNotesAndKeepsItsKeyThroughAFewOthers) {
    KeyDetector detector;
    auto notes = line({60, 62, 64}, 0, 25, 25);
    EXPECT_FALSE(detector.update(notes, 80).key.has_value()) << "fewer than 8 notes";

    // the C scale alone is also A minor's notes: a C bass decides it
    notes = line({48, 62, 64, 65, 55, 69, 71, 72, 64, 48}, 0, 25, 25);
    for (auto &n: notes)
        n.bass = n.pitch < 60;
    ASSERT_TRUE(detector.update(notes, 250).key.has_value());
    EXPECT_EQ(KeyDetector::name(*detector.current().key), "C major");

    // G major arrives note by note: two F#s are not enough, a run of G-major notes is.
    auto gMajor = line({43, 66, 67, 71, 50, 66, 67, 71, 43, 78, 79, 66, 43, 71, 74, 66}, 250, 25, 25);
    for (auto &n: gMajor)
        n.bass = n.pitch < 60;
    int32_t now = 250;
    for (size_t i = 0; i < gMajor.size(); ++i) {
        notes.push_back(gMajor[i]);
        now = gMajor[i].onsetCs + 5;
        detector.update(notes, now);
        if (i == 1)
            EXPECT_EQ(KeyDetector::name(*detector.current().key), "C major") << "after two notes";
    }
    EXPECT_EQ(KeyDetector::name(*detector.current().key), "G major");
}

TEST(KeyDetector, tiedKeysFollowThePreviousKey) {
    EXPECT_EQ(KeyDetector::lofForTonic(6, false, std::nullopt), 6) << "F# major by default";
    EXPECT_EQ(KeyDetector::lofForTonic(6, false, kDb), -6) << "Gb major after Db";
    EXPECT_EQ(KeyDetector::lofForTonic(3, true, std::nullopt), -3) << "Eb minor by default";
    EXPECT_EQ(KeyDetector::lofForTonic(3, true, kGSharpMinor), 9) << "D# minor after G# minor";
    EXPECT_EQ(KeyDetector::lofForTonic(1, false, std::nullopt), -5) << "Db major, not C#";
    EXPECT_EQ(KeyDetector::lofForTonic(8, true, std::nullopt), 8) << "G# minor, not Ab";
}

// ---- ScoreHarmony

TEST(ScoreHarmony, aRepeatedChordIsNamedOnce) {
    EXPECT_EQ(labelsOf({{0, 40, 48}, {0, 40, 64}, {0, 40, 67},
                        {50, 40, 48}, {50, 40, 64}, {50, 40, 67},
                        {100, 40, 43}, {100, 40, 62}, {100, 40, 67}}),
              (std::vector<std::string>{"C", "G"}));
}

TEST(ScoreHarmony, heldNotesCountAndAnArpeggioIsOneChord) {
    // C, E, G one after another, each held: one C
    EXPECT_EQ(labelsOf({{0, 200, 48}, {20, 180, 52}, {40, 160, 55}}), (std::vector<std::string>{"C"}));
}

TEST(ScoreHarmony, aSingleLineIsNoChord) {
    // a chord, then a melody alone: N/C at its first note
    const auto chords = ScoreHarmony::chords(
        ScoreLayout::prepare({{0, 40, 48}, {0, 40, 64}, {0, 40, 67}, {50, 20, 72}, {75, 20, 74}, {100, 20, 76}}), kC);
    const auto labels = ScoreHarmony::labels(chords);
    ASSERT_EQ(labels.size(), 2u);
    EXPECT_EQ(labels[1].text, "N/C");
    EXPECT_EQ(labels[1].onsetCs, 50);
    // one short lone note between two chords is not
    EXPECT_EQ(labelsOf({{0, 40, 48}, {0, 40, 64}, {0, 40, 67}, {45, 10, 72},
                        {60, 40, 41}, {60, 40, 65}, {60, 40, 69}}),
              (std::vector<std::string>{"C", "F"}));
    // octaves are one line
    EXPECT_EQ(labelsOf({{0, 30, 48}, {0, 30, 60}, {40, 30, 50}, {40, 30, 62}}), (std::vector<std::string>{"N/C"}));
}

TEST(ScoreHarmony, aPowerChordTakesItsThirdFromTheKey) {
    EXPECT_EQ(labelsOf({{0, 50, 45}, {0, 50, 52}}, kC), (std::vector<std::string>{"Am"}));
    EXPECT_EQ(labelsOf({{0, 50, 45}, {0, 50, 52}}, Key{2, false}), (std::vector<std::string>{"A"})) << "D major";
}

TEST(ScoreHarmony, rootsAndLineNotesAreSpelledInTheKey) {
    // G minor in F major: its Bb is a Bb
    const auto prepared = ScoreLayout::prepare({{0, 50, 55}, {0, 50, 58}, {0, 50, 62}});
    const auto chords = ScoreHarmony::chords(prepared, kF);
    ASSERT_EQ(chords.size(), 1u);
    EXPECT_EQ(chords[0].text, "Gm");
    const auto lofs = ScoreHarmony::spellLine({{0, 50, 58}}, chords, kF);
    EXPECT_EQ(Spelling::nameOf(lofs[0]), "Bb");
    // F# major chord in Db major is Gb
    EXPECT_EQ(labelsOf({{0, 50, 54}, {0, 50, 58}, {0, 50, 61}}, kDb), (std::vector<std::string>{"Gb"}));
    // the V of G# minor is D#, its third F double sharp
    const auto v = ScoreHarmony::chords(ScoreLayout::prepare({{0, 50, 51}, {0, 50, 55}, {0, 50, 58}}), kGSharpMinor);
    EXPECT_EQ(v[0].text, "D#");
    EXPECT_EQ(Spelling::nameOf(ScoreHarmony::spellLine({{0, 50, 55}}, v, kGSharpMinor)[0]), "Fx");
}

} // namespace virtual_orch_test
