# chordnamer (vendored)

Chord names for the ScoreWidget (`app/source/ChordNaming.cpp`).

- Source: https://github.com/lancelotblanchard/CppChordNamer (fork of
  https://github.com/junghau/ChordNamer), commit `aef76a8` ("Fix sus misrating", 2024-09-13).
- Copied: `include/{chord,interval,note}.h`, `src/{chord,interval,note}.cpp`. Not copied: the
  demo (`src/demo.cpp`, it has its own `main()`) and upstream's CMake / CI files. Built by
  `app/CMakeLists.txt` as the `chordnamer` static library.
- Upstream has no LICENSE file; check with the authors before redistributing outside this project.

## Local changes (2026-10-20)

1. `src/note.cpp`: the note letter is the FIRST letter of the name. The parser kept the last
   character in a-g, so the flat sign of "Eb" was read as the note B ("Eb" -> Bb, "C Eb G Bb"
   named "C57"). Also the upper-case range check is now `c <= 'G'` (was `'g'`).
2. `src/chord.cpp`: in a half-diminished chord the b5 is part of the m7b5 quality, not an added
   alteration. Before, it counted toward the ranking, so B D F A tied with (and lost to) "Dm/B".

Known limitation kept: C D G is named "G5/C" before "Csus2" (a ranking tie; preferring root
position on ties broke E G C -> "Em#5").
