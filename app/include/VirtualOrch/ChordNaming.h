#pragma once

#include <string>
#include <vector>

/**
 * Chord names for the ScoreWidget, from the vendored CppChordNamer (libs/chordnamer).
 *
 * The lowest pitch is the bass (slash chords are named against it). Every group of two or more
 * notes gets a name: three or more pitch classes go to the namer, and if it finds none, upper
 * notes are left out one at a time until it does; two pitch classes are named from their
 * interval (C5, Cm, C, Csus2, C7, Cmaj7, Cdim, F/C, Ab/C, Am/C, Cmaj7/B); one pitch class (octaves)
 * is its note name. A single note has no name. The root is spelled with sharps, except Db / Eb / Ab / Bb
 * for major-type chords; minor-type chords follow the minor keys (C#m, G#m, but Ebm, Bbm).
 * F# stays sharp. A slash bass is spelled from its interval above the root
 * (B/D#, E/G#, Eb/G), never as an enharmonic like B/Eb.
 */
namespace ChordNaming {

[[nodiscard]] auto nameFor(std::vector<int> midiPitches) -> std::string;

/** The name without its slash bass ("C/E" -> "C"): the score shows only the chord. */
[[nodiscard]] auto withoutBass(const std::string &name) -> std::string;

/**
 * Root and basic quality only, no extensions or slash bass: Cmaj7 / C7 / C9 / Cadd9 -> C,
 * Dm7 -> Dm, Bm7b5 / Bdim7 -> Bdim, Caug7 -> Caug, G7sus4 -> Gsus4, C5 stays. The score shows this.
 */
[[nodiscard]] auto basicChord(const std::string &name) -> std::string;

/** True when the chord's root is spelled flat (its notes are then drawn with flats). */
[[nodiscard]] auto rootIsFlat(const std::string &name) -> bool;

} // namespace ChordNaming
