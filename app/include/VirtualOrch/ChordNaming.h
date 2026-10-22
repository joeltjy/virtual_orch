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
 * is its note name. A single note has no name. The root is spelled with sharps, except Db / Eb / Ab / Bb, which
 * are written flat (F# stays sharp). A slash bass is spelled from its interval above the root
 * (B/D#, E/G#, Eb/G), never as an enharmonic like B/Eb.
 */
namespace ChordNaming {

[[nodiscard]] auto nameFor(std::vector<int> midiPitches) -> std::string;

/** True when the chord's root is spelled flat (its notes are then drawn with flats). */
[[nodiscard]] auto rootIsFlat(const std::string &name) -> bool;

} // namespace ChordNaming
