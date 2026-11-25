#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

/**
 * Hybrid v1.2 chord branches (orchestration/docs/model_specs/expressified_simultaneous.txt B3f, "CHORD BRANCHES"):
 * the pitch-class rules the engine applies at a fire. Types are VoicingBank::kTypeNames / wait3.TYPES indices
 * (0 unison, 1..6 icN, 7 maj, 8 min, 9 dim, 10 aug, 11 dom7, 12 min7, 13 maj7, 14 minmaj7, 15 hdim7, 16 dim7,
 * 17 augmaj7). A pitch-class set is a 12-bit mask (bit p = pitch class p). "Notes" are pitch classes throughout.
 */
namespace ChordTheory {

using Pcs = uint16_t;

[[nodiscard]] auto count(Pcs s) -> int;
[[nodiscard]] auto maskOf(const std::vector<int32_t> &pitches) -> Pcs;
/** The pitch classes of (type, root). */
[[nodiscard]] auto chordMask(int type, int root) -> Pcs;

struct TypeRoot {
    int type = 0;
    int root = 0;
    friend auto operator==(const TypeRoot &a, const TypeRoot &b) -> bool { return a.type == b.type && a.root == b.root; }
};
/**
 * wait3.chord_type: 1 pc unison; 2 pcs the interval class, root = the lower note of the closer interval; 3-4 pcs an
 * exact triad / seventh (any inversion; aug / dim7 root = the lowest pitch class); otherwise none.
 */
[[nodiscard]] auto typeOf(Pcs s) -> std::optional<TypeRoot>;
/** MATCH RULE: a triad, a seventh chord, or fewer than 3 pitch classes (and not empty). */
[[nodiscard]] auto recognized(Pcs s) -> bool;

struct Key {
    int tonic = 0;
    bool minor = false;
    friend auto operator==(const Key &a, const Key &b) -> bool { return a.tonic == b.tonic && a.minor == b.minor; }
};
/** Krumhansl-Kessler major / minor profiles, the best Pearson correlation with the weights (C major if all 0). */
[[nodiscard]] auto estimateKey(const std::array<double, 12> &weights) -> Key;

struct Canonical {
    TypeRoot chord;
    int degree = 0; // 0..6, scale degree of the root (I = 0)
};
/**
 * Major: the diatonic triads and sevenths. Minor: natural minor (triads and sevenths) plus V, V7 and vii°7 from
 * harmonic minor.
 */
[[nodiscard]] auto canonicalChords(Key key) -> std::vector<Canonical>;
/** The rank of a degree in "the most common degree" order I, V, IV, ii, vi, iii, vii (0 = most common). */
[[nodiscard]] auto degreeRank(int degree) -> int;

struct Inferred {
    TypeRoot chord;
    int step = 0; // the INFERENCE RULE step that gave it: 1..4
    Key key;      // the estimated key (steps 1-3 use it for ties)
};
/**
 * INFERENCE RULE, from the pitch classes `piano` of the onset's piano notes (lowest MIDI pitch `lowestPitch`), the
 * K token's k (pitch classes) and the key weights (the last ~8 s of piano + orchestra, by duration):
 *   1. < 3 pitch classes: that unison / dyad;
 *   2. else a triad / seventh containing them (ties: exact fit, fewest missing, canonical in the key, dom7 before min7);
 *   3. else a canonical chord of the estimated key that differs by one note (one extra, one missing or one swapped;
 *      ties: fewest differences, then the most common degree);
 *   4. else a major / minor triad on the lowest note (minor if its minor third is present).
 * At each step the chords with k pitch classes come first (a dyad is extended to a triad when k = 3, ...), but a
 * mismatch is accepted rather than failing.
 */
[[nodiscard]] auto infer(Pcs piano, int32_t lowestPitch, int k, const std::array<double, 12> &keyWeights) -> Inferred;

} // namespace ChordTheory
