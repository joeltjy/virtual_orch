#pragma once

#include <array>
#include <cstdlib>
#include <optional>
#include <string>

/**
 * Note spelling on the line of fifths, for the score: which letter and accidental a pitch class
 * gets in a key.
 *
 * Positions on the line of fifths (C = 0, G = 1, F = -1, F# = 6, Bb = -2, ...): any 12
 * consecutive positions hold every pitch class once, so a window of 12 positions decides the
 * spelling of every note. The window of a key is tonic - 5 .. tonic + 6 (C major: Db .. F#;
 * A minor: Bb .. D#). Chord tones are spelled by their interval above the chord's root
 * (E major in C: G#; G minor: Bb). Strict theory spelling: roots such as D# or Ebb are allowed,
 * and double sharps / flats are used where the key or the interval asks for them.
 */
namespace Spelling {

/** A key: its tonic's line-of-fifths position and mode. Default: C major. */
struct Key {
    int tonicLof = 0;
    bool minor = false;
    [[nodiscard]] auto operator==(const Key &o) const -> bool { return tonicLof == o.tonicLof && minor == o.minor; }
};

[[nodiscard]] inline auto floorDiv(int a, int b) -> int {
    const int q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

[[nodiscard]] inline auto mod(int a, int m) -> int { return ((a % m) + m) % m; }

/** Pitch class of a line-of-fifths position. */
[[nodiscard]] inline auto pcOf(int lof) -> int { return mod(lof * 7, 12); }

/** Letter of a position as a degree, C = 0 ... B = 6. */
[[nodiscard]] inline auto degreeOf(int lof) -> int {
    static constexpr std::array<int, 7> kFromF{3, 0, 4, 1, 5, 2, 6}; // F C G D A E B
    return kFromF[static_cast<size_t>(mod(lof + 1, 7))];
}

/** Accidental of a position: +1 sharp, -1 flat, +2 / -2 double. */
[[nodiscard]] inline auto accidentalOf(int lof) -> int { return floorDiv(lof + 1, 7); }

/** The position of a letter (degree, C = 0) with an accidental. */
[[nodiscard]] inline auto lofOf(int degree, int accidental) -> int {
    static constexpr std::array<int, 7> kNatural{0, 2, 4, -1, 1, 3, 5}; // C D E F G A B
    return kNatural[static_cast<size_t>(mod(degree, 7))] + 7 * accidental;
}

/** Natural pitch class of a degree (C = 0 ... B = 11). */
[[nodiscard]] inline auto naturalPc(int degree) -> int {
    static constexpr std::array<int, 7> kPc{0, 2, 4, 5, 7, 9, 11};
    return kPc[static_cast<size_t>(mod(degree, 7))];
}

/** The position with this pitch class in the window lo .. lo + 11. */
[[nodiscard]] inline auto lofInWindow(int pc, int lo) -> int {
    for (int p = lo; p < lo + 12; ++p)
        if (pcOf(p) == mod(pc, 12))
            return p;
    return lo;
}

/** "C", "F#", "Bb", "Fx" (double sharp), "Ebb". */
[[nodiscard]] inline auto nameOf(int lof) -> std::string {
    static constexpr const char *kLetters = "CDEFGAB";
    std::string s(1, kLetters[degreeOf(lof)]);
    const int acc = accidentalOf(lof);
    if (acc == 2)
        s += 'x';
    else if (acc > 0)
        s += std::string(static_cast<size_t>(acc), '#');
    else if (acc < 0)
        s += std::string(static_cast<size_t>(-acc), 'b');
    return s;
}

[[nodiscard]] inline auto tonicPc(const Key &key) -> int { return pcOf(key.tonicLof); }

/** First position of the key's spelling window (tonic - 5). */
[[nodiscard]] inline auto windowLow(const Key &key) -> int { return key.tonicLof - 5; }

/**
 * Whether a pitch class belongs to the key: the major scale, or for minor the natural minor
 * plus the raised 6th and 7th.
 */
[[nodiscard]] inline auto inScale(int pc, const Key &key) -> bool {
    static constexpr std::array<bool, 12> kMajor{true, false, true, false, true, true, false, true, false, true, false, true};
    static constexpr std::array<bool, 12> kMinor{true, false, true, true, false, true, false, true, true, true, true, true};
    const int rel = mod(pc - tonicPc(key), 12);
    return (key.minor ? kMinor : kMajor)[static_cast<size_t>(rel)];
}

/** The key's spelling of a pitch class (its window). */
[[nodiscard]] inline auto inKey(int pc, const Key &key) -> int { return lofInWindow(pc, windowLow(key)); }

/** A letter + accidental for `pc` on `degree`, or nullopt beyond a double sharp / flat. */
[[nodiscard]] inline auto onDegree(int pc, int degree) -> std::optional<int> {
    int acc = mod(pc - naturalPc(degree), 12);
    if (acc > 6)
        acc -= 12;
    if (std::abs(acc) > 2)
        return std::nullopt;
    return lofOf(degree, acc);
}

/**
 * A chord's root: the key's spelling, except a diminished chord (dim, dim7, m7b5) whose root is
 * outside the key, which usually leads up a semitone: it is spelled as the note below the scale
 * note above it (C#dim7 in C, not Dbdim7).
 */
[[nodiscard]] inline auto root(int pc, const Key &key, bool diminished) -> int {
    if (diminished && ! inScale(pc, key) && inScale(pc + 1, key))
        if (const auto p = onDegree(pc, degreeOf(inKey(pc + 1, key)) - 1))
            return *p;
    return inKey(pc, key);
}

/** What a chord's quality says about its ambiguous intervals. */
struct Quality {
    bool augmented = false;  // the 8-semitone interval is a raised 5th (else a minor 6th)
    bool sharp11 = false;    // the 6-semitone interval is a raised 4th / 11th (else a lowered 5th)
    bool dim7 = false;       // the 9-semitone interval is a diminished 7th (else a major 6th)
};

[[nodiscard]] inline auto qualityOf(const std::string &q) -> Quality {
    const auto has = [&](const char *s) { return q.find(s) != std::string::npos; };
    return {has("aug") || has("+") || has("#5"), has("#11"), has("dim7") || has("o7")};
}

/**
 * A chord tone spelled by its interval above the root (third = 2 letters up, fifth = 4,
 * seventh = 6, ninth = 1, eleventh = 3, thirteenth = 5), or nullopt if that needs more than a
 * double accidental.
 */
[[nodiscard]] inline auto chordTone(int pc, int rootLof, const Quality &quality) -> std::optional<int> {
    static constexpr std::array<int, 12> kSteps{0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6};
    const int interval = mod(pc - pcOf(rootLof), 12);
    int steps = kSteps[static_cast<size_t>(interval)];
    if (interval == 8 && quality.augmented)
        steps = 4;
    if (interval == 6 && quality.sharp11)
        steps = 3;
    if (interval == 9 && quality.dim7)
        steps = 6;
    return onDegree(pc, degreeOf(rootLof) + steps);
}

/**
 * A note outside the chord: a note outside the key a semitone from the previous note of its
 * line is a chromatic step (rising: the lower letter raised, C - C# - D; falling: the upper
 * letter lowered, D - Db - C); anything else takes the key's spelling.
 */
[[nodiscard]] inline auto passing(int pitch, std::optional<int> previousPitch, const Key &key) -> int {
    const int pc = mod(pitch, 12);
    if (previousPitch.has_value() && ! inScale(pc, key)) {
        const int step = pitch - *previousPitch;
        if (step == 1)
            if (const auto p = onDegree(pc, degreeOf(inKey(pc - 1, key))))
                return *p;
        if (step == -1)
            if (const auto p = onDegree(pc, degreeOf(inKey(pc + 1, key))))
                return *p;
    }
    return inKey(pc, key);
}

/** Staff position (diatonic step, 7 per MIDI octave as ScoreLayout::spell) of a spelled pitch. */
[[nodiscard]] inline auto staffStep(int pitch, int lof) -> int {
    const int natural = pitch - accidentalOf(lof);
    return floorDiv(natural, 12) * 7 + degreeOf(lof);
}

/** Key signature: sharps (> 0) or flats (< 0) of a key (minor: its relative major's). */
[[nodiscard]] inline auto signature(const Key &key) -> int { return key.minor ? key.tonicLof - 3 : key.tonicLof; }

/** The accidental the key signature gives a letter (degree). */
[[nodiscard]] inline auto signatureAccidental(int degree, int signatureCount) -> int {
    const int natural = lofOf(degree, 0); // F = -1 ... B = 5
    if (signatureCount > 0)
        return natural <= signatureCount - 2 ? 1 : 0; // sharps added F C G D A E B
    if (signatureCount < 0)
        return natural >= 6 + signatureCount ? -1 : 0; // flats added B E A D G C F
    return 0;
}

} // namespace Spelling
