#pragma once

#include <string>
#include <vector>

/**
 * Chord names for the ScoreWidget, from the vendored CppChordNamer (libs/chordnamer).
 *
 * The lowest pitch is the bass (slash chords are named against it). Fewer than three distinct
 * pitch classes give no name. Names are spelled with sharps, except that a root or bass on
 * Db / Eb / Ab / Bb is written flat (F# stays sharp).
 */
namespace ChordNaming {

[[nodiscard]] auto nameFor(std::vector<int> midiPitches) -> std::string;

/** True when the chord's root is spelled flat (its notes are then drawn with flats). */
[[nodiscard]] auto rootIsFlat(const std::string &name) -> bool;

} // namespace ChordNaming
