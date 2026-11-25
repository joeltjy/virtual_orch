#include <gtest/gtest.h>

#include <algorithm>

#include <VirtualOrch/reduction/ChordTheory.h>

/**
 * Hybrid v1.2 CHORD BRANCHES (expressified_simultaneous.txt B3f): the MATCH RULE, the INFERENCE RULE steps 1-4 with
 * their ties, the key estimate and the k preference.
 */
namespace {

using namespace ChordTheory;

constexpr int kUnison = 0;
constexpr int kMaj = 7;
constexpr int kMin = 8;
constexpr int kDom7 = 11;
constexpr int kMin7 = 12;
constexpr int kMaj7 = 13;
constexpr int kDim7 = 16;

auto pcs(std::initializer_list<int> list) -> Pcs { return maskOf(std::vector<int32_t>(list)); }

/** Weights shaped like a key: the Krumhansl profile itself, rotated to the tonic. */
auto keyLike(int tonic, bool minor) -> std::array<double, 12> {
    static constexpr double kMajor[12] = {6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88};
    static constexpr double kMinor[12] = {6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17};
    std::array<double, 12> w{};
    for (int pc = 0; pc < 12; ++pc)
        w[static_cast<size_t>(pc)] = (minor ? kMinor : kMajor)[((pc - tonic) % 12 + 12) % 12];
    return w;
}

const std::array<double, 12> kNoWeights{};

} // namespace

TEST(ChordTheory, NamesChordsAsWait3Does) {
    EXPECT_EQ(typeOf(pcs({0, 4, 7})), (TypeRoot{kMaj, 0}));
    EXPECT_EQ(typeOf(pcs({4, 7, 12})), (TypeRoot{kMaj, 0})); // any inversion / octave
    EXPECT_EQ(typeOf(pcs({2, 5, 9, 0})), (TypeRoot{kMin7, 2}));
    EXPECT_EQ(typeOf(pcs({0, 4, 8})), (TypeRoot{10, 0}));           // aug: the lowest pitch class
    EXPECT_EQ(typeOf(pcs({11, 2, 5, 8})), (TypeRoot{kDim7, 2}));    // dim7: the lowest pitch class
    EXPECT_EQ(typeOf(pcs({0, 9})), (TypeRoot{3, 9}));               // A-C: ic3, the lower note of the closer interval
    EXPECT_EQ(typeOf(pcs({5})), (TypeRoot{kUnison, 5}));
    EXPECT_FALSE(typeOf(pcs({0, 2, 4})).has_value());
    EXPECT_FALSE(typeOf(pcs({0, 2, 4, 5, 7})).has_value());
}

TEST(ChordTheory, MatchRule) {
    EXPECT_TRUE(recognized(pcs({0, 4, 7})));      // triad
    EXPECT_TRUE(recognized(pcs({0, 4, 7, 11})));  // seventh
    EXPECT_TRUE(recognized(pcs({0, 2})));         // < 3 pitch classes
    EXPECT_TRUE(recognized(pcs({3})));
    EXPECT_FALSE(recognized(0));
    EXPECT_FALSE(recognized(pcs({0, 2, 4})));
    EXPECT_FALSE(recognized(pcs({0, 4, 7, 9, 2})));
    // piano E with a C major candidate: still C major -> matches; piano D with it: C D E G -> no match
    EXPECT_TRUE(recognized(static_cast<Pcs>(pcs({64}) | chordMask(kMaj, 0))));
    EXPECT_FALSE(recognized(static_cast<Pcs>(pcs({62}) | chordMask(kMaj, 0))));
    // piano B with C major: C E G B = Cmaj7 -> matches
    EXPECT_TRUE(recognized(static_cast<Pcs>(pcs({71}) | chordMask(kMaj, 0))));
}

TEST(ChordTheory, KeyEstimate) {
    EXPECT_EQ(estimateKey(keyLike(0, false)), (Key{0, false}));
    EXPECT_EQ(estimateKey(keyLike(2, true)), (Key{2, true}));
    EXPECT_EQ(estimateKey(keyLike(10, false)), (Key{10, false}));
    // the notes of an A minor stretch weighted by duration: A, C, E long, B, D short, G# (leading tone)
    std::array<double, 12> w{};
    w[9] = 400;
    w[0] = 200;
    w[4] = 300;
    w[11] = 50;
    w[2] = 60;
    w[8] = 80;
    EXPECT_EQ(estimateKey(w), (Key{9, true}));
    EXPECT_EQ(estimateKey(kNoWeights), (Key{0, false}));
}

TEST(ChordTheory, CanonicalChordsOfMajorAndMinorKeys) {
    const auto has = [](const std::vector<Canonical> &list, TypeRoot c) {
        return std::any_of(list.begin(), list.end(), [&](const Canonical &e) { return e.chord == c; });
    };
    const auto cMajor = canonicalChords(Key{0, false});
    EXPECT_EQ(cMajor.size(), 14u);
    EXPECT_TRUE(has(cMajor, {kMaj, 0}));
    EXPECT_TRUE(has(cMajor, {kDom7, 7}));  // V7
    EXPECT_TRUE(has(cMajor, {15, 11}));    // viiø7
    EXPECT_FALSE(has(cMajor, {kDom7, 0}));
    const auto aMinor = canonicalChords(Key{9, true});
    EXPECT_TRUE(has(aMinor, {kMin, 9}));   // i
    EXPECT_TRUE(has(aMinor, {kMin, 4}));   // v (natural)
    EXPECT_TRUE(has(aMinor, {kMaj, 4}));   // V (harmonic)
    EXPECT_TRUE(has(aMinor, {kDom7, 4}));  // V7
    EXPECT_TRUE(has(aMinor, *typeOf(chordMask(kDim7, 8)))); // vii°7 (G# dim7, named by its lowest pc)
    EXPECT_EQ(degreeRank(0), 0); // I
    EXPECT_EQ(degreeRank(4), 1); // V
    EXPECT_EQ(degreeRank(3), 2); // IV
}

TEST(ChordTheory, Step1UnisonOrDyadAndItsExtensionToK) {
    auto r = infer(pcs({60, 64}), 60, 2, kNoWeights);
    EXPECT_EQ(r.step, 1);
    EXPECT_EQ(r.chord, (TypeRoot{4, 0})); // C-E: ic4
    r = infer(pcs({67}), 67, 1, kNoWeights);
    EXPECT_EQ(r.chord, (TypeRoot{kUnison, 7}));
    // k = 3: the dyad C-E extended to a triad; canonical in C major and the most common degree -> C major
    r = infer(pcs({60, 64}), 60, 3, keyLike(0, false));
    EXPECT_EQ(r.step, 1);
    EXPECT_EQ(r.chord, (TypeRoot{kMaj, 0}));
    // ... in A minor the same dyad is A minor (i)
    r = infer(pcs({60, 64}), 60, 3, keyLike(9, true));
    EXPECT_EQ(r.chord, (TypeRoot{kMin, 9}));
    // k = 1 with a dyad: a mismatch is accepted (the dyad; C-G is a fourth G-C: ic5 on G)
    r = infer(pcs({60, 67}), 60, 1, kNoWeights);
    EXPECT_EQ(r.chord, (TypeRoot{5, 7}));
}

TEST(ChordTheory, Step2FitsATriadOrSeventhContainingThePianoNotes) {
    // the spec's example: C Eb F -> F7 (F7 and Fm7 both contain it with one note missing; dom7 before min7)
    auto r = infer(pcs({60, 63, 65}), 60, 4, kNoWeights);
    EXPECT_EQ(r.step, 2);
    EXPECT_EQ(r.chord, (TypeRoot{kDom7, 5}));
    // k = 3 (no triad contains it): the seventh is still taken rather than failing
    r = infer(pcs({60, 63, 65}), 60, 3, kNoWeights);
    EXPECT_EQ(r.chord, (TypeRoot{kDom7, 5}));
    // canonical in the key beats dom7-before-min7: in Eb major Fm7 is ii7
    r = infer(pcs({60, 63, 65}), 60, 4, keyLike(3, false));
    EXPECT_EQ(r.chord, (TypeRoot{kMin7, 5}));
    // exact fit first
    r = infer(pcs({60, 64, 67}), 60, 3, kNoWeights);
    EXPECT_EQ(r.chord, (TypeRoot{kMaj, 0}));
    // k preference: k = 4 extends the exact triad to the seventh canonical in C major (Imaj7, not C7)
    r = infer(pcs({60, 64, 67}), 60, 4, keyLike(0, false));
    EXPECT_EQ(r.chord, (TypeRoot{kMaj7, 0}));
    // ... and in F major C7 is V7
    r = infer(pcs({60, 64, 67}), 60, 4, keyLike(5, false));
    EXPECT_EQ(r.chord, (TypeRoot{kDom7, 0}));
}

TEST(ChordTheory, Step3ACanonicalChordOfTheKeyOneNoteAway) {
    // C E G A D: no triad / seventh contains it; Am7 (vi7) is it minus one extra note (D)
    auto r = infer(pcs({60, 64, 67, 69, 62}), 60, 4, keyLike(0, false));
    EXPECT_EQ(r.step, 3);
    EXPECT_EQ(r.key, (Key{0, false}));
    EXPECT_EQ(r.chord, (TypeRoot{kMin7, 9}));
    // C C# E G: k = 3 -> C major (one extra note); k = 4 -> Cmaj7 (C# swapped for B; ties Am7 on differences, I first)
    r = infer(pcs({60, 61, 64, 67}), 60, 3, keyLike(0, false));
    EXPECT_EQ(r.step, 3);
    EXPECT_EQ(r.chord, (TypeRoot{kMaj, 0}));
    r = infer(pcs({60, 61, 64, 67}), 60, 4, keyLike(0, false));
    EXPECT_EQ(r.chord, (TypeRoot{kMaj7, 0}));
}

TEST(ChordTheory, Step4ATriadOnTheLowestNote) {
    // C# D D# E F: nothing near -> on the lowest note C# (61); its minor third E is there -> C# minor
    auto r = infer(pcs({61, 62, 63, 64, 65}), 61, 3, keyLike(0, false));
    EXPECT_EQ(r.step, 4);
    EXPECT_EQ(r.chord, (TypeRoot{kMin, 1}));
    // lowest D (62), no F: D major; k = 4: the seventh chord on it
    r = infer(pcs({62, 63, 64, 61, 60}), 62, 3, keyLike(0, false));
    EXPECT_EQ(r.step, 4);
    EXPECT_EQ(r.chord, (TypeRoot{kMaj, 2}));
    r = infer(pcs({62, 63, 64, 61, 60}), 62, 4, keyLike(0, false));
    EXPECT_EQ(r.chord, (TypeRoot{kDom7, 2}));
}
