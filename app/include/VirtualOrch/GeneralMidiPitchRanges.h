#pragma once

#include <array>
#include <cstdint>

/**
 * Pitch range (MIDI note numbers, inclusive) for each General MIDI program id [0, 127].
 * Fill in low/high per instrument; placeholders are full MIDI range.
 */
namespace GeneralMidiPitchRanges {

struct PitchRange {
    int32_t low = 0;
    int32_t high = 127;
};

inline constexpr int32_t kNumInstruments = 128;

/** Indexed by GM program number (0 = Acoustic Grand Piano … 127 = Gunshot). */
inline constexpr std::array<PitchRange, kNumInstruments> kPitchRanges{{
    /* 0   Acoustic Grand Piano   */ {.low = 21, .high = 108}, // A0–C8
    /* 1   Bright Acoustic Piano  */ {},
    /* 2   Electric Grand Piano   */ {},
    /* 3   Honky-tonk Piano       */ {},
    /* 4   Electric Piano 1       */ {},
    /* 5   Electric Piano 2       */ {},
    /* 6   Harpsichord            */ {},
    /* 7   Clavi                  */ {},
    /* 8   Celesta                */ {},
    /* 9   Glockenspiel           */ {},
    /* 10  Music Box              */ {},
    /* 11  Vibraphone             */ {},
    /* 12  Marimba                */ {},
    /* 13  Xylophone              */ {},
    /* 14  Tubular Bells          */ {},
    /* 15  Dulcimer               */ {},
    /* 16  Drawbar Organ          */ {},
    /* 17  Percussive Organ       */ {},
    /* 18  Rock Organ             */ {},
    /* 19  Church Organ           */ {},
    /* 20  Reed Organ             */ {},
    /* 21  Accordion              */ {},
    /* 22  Harmonica              */ {},
    /* 23  Tango Accordion        */ {},
    /* 24  Acoustic Guitar (nylon)*/ {},
    /* 25  Acoustic Guitar (steel)*/ {.low = 40, .high = 88}, // E2–E6
    /* 26  Electric Guitar (jazz) */ {},
    /* 27  Electric Guitar (clean)*/ {},
    /* 28  Electric Guitar (muted)*/ {},
    /* 29  Overdriven Guitar      */ {},
    /* 30  Distortion Guitar      */ {},
    /* 31  Guitar Harmonics       */ {},
    /* 32  Acoustic Bass          */ {},
    /* 33  Electric Bass (finger) */ {},
    /* 34  Electric Bass (pick)   */ {},
    /* 35  Fretless Bass          */ {},
    /* 36  Slap Bass 1            */ {},
    /* 37  Slap Bass 2            */ {},
    /* 38  Synth Bass 1           */ {},
    /* 39  Synth Bass 2           */ {},
    /* 40  Violin                 */ {.low = 55, .high = 100}, // G3–E7
    /* 41  Viola                  */ {.low = 48, .high = 88}, // C3–E6
    /* 42  Cello                  */ {.low = 36, .high = 76}, // C2–E5
    /* 43  Contrabass             */ {.low = 28, .high = 55}, // E1–G3
    /* 44  Tremolo Strings        */ {.low = 36, .high = 96}, // C2–C7 (section)
    /* 45  Pizzicato Strings      */ {.low = 36, .high = 96}, // C2–C7 (section)
    /* 46  Orchestral Harp        */ {.low = 23, .high = 103}, // B0–G7
    /* 47  Timpani                */ {.low = 36, .high = 60}, // C2–C4
    /* 48  String Ensemble 1      */ {.low = 36, .high = 96}, // C2–C7
    /* 49  String Ensemble 2      */ {},
    /* 50  Synth Strings 1        */ {},
    /* 51  Synth Strings 2        */ {},
    /* 52  Choir Aahs             */ {.low = 48, .high = 84}, // C3–C6
    /* 53  Voice Oohs             */ {},
    /* 54  Synth Voice            */ {},
    /* 55  Orchestra Hit          */ {},
    /* 56  Trumpet                */ {.low = 52, .high = 82}, // E3–A#5
    /* 57  Trombone               */ {.low = 40, .high = 72}, // E2–C5
    /* 58  Tuba                   */ {.low = 26, .high = 60}, // D1–C4
    /* 59  Muted Trumpet          */ {},
    /* 60  French Horn            */ {.low = 34, .high = 77}, // Bb1–F5
    /* 61  Brass Section          */ {},
    /* 62  Synth Brass 1          */ {},
    /* 63  Synth Brass 2          */ {},
    /* 64  Soprano Sax            */ {},
    /* 65  Alto Sax               */ {},
    /* 66  Tenor Sax              */ {},
    /* 67  Baritone Sax           */ {},
    /* 68  Oboe                   */ {},
    /* 69  English Horn           */ {},
    /* 70  Bassoon                */ {.low = 34, .high = 75}, // Bb1–Eb5
    /* 71  Clarinet               */ {.low = 50, .high = 91}, // D3–G6
    /* 72  Piccolo                */ {},
    /* 73  Flute                  */ {.low = 60, .high = 96}, // C4–C7

    /* 74  Recorder               */ {},
    /* 75  Pan Flute              */ {},
    /* 76  Blown Bottle           */ {},
    /* 77  Shakuhachi             */ {},
    /* 78  Whistle                */ {},
    /* 79  Ocarina                */ {},
    /* 80  Lead 1 (square)        */ {},
    /* 81  Lead 2 (sawtooth)      */ {},
    /* 82  Lead 3 (calliope)      */ {},
    /* 83  Lead 4 (chiff)         */ {},
    /* 84  Lead 5 (charang)       */ {},
    /* 85  Lead 6 (voice)         */ {},
    /* 86  Lead 7 (fifths)        */ {},
    /* 87  Lead 8 (bass + lead)   */ {},
    /* 88  Pad 1 (new age)        */ {},
    /* 89  Pad 2 (warm)           */ {},
    /* 90  Pad 3 (polysynth)      */ {},
    /* 91  Pad 4 (choir)          */ {},
    /* 92  Pad 5 (bowed)          */ {},
    /* 93  Pad 6 (metallic)       */ {},
    /* 94  Pad 7 (halo)           */ {},
    /* 95  Pad 8 (sweep)          */ {},
    /* 96  FX 1 (rain)            */ {},
    /* 97  FX 2 (soundtrack)      */ {},
    /* 98  FX 3 (crystal)         */ {},
    /* 99  FX 4 (atmosphere)      */ {},
    /* 100 FX 5 (brightness)      */ {},
    /* 101 FX 6 (goblins)         */ {},
    /* 102 FX 7 (echoes)          */ {},
    /* 103 FX 8 (sci-fi)          */ {},
    /* 104 Sitar                  */ {},
    /* 105 Banjo                  */ {},
    /* 106 Shamisen               */ {},
    /* 107 Koto                   */ {},
    /* 108 Kalimba                */ {},
    /* 109 Bag Pipe               */ {},
    /* 110 Fiddle                 */ {},
    /* 111 Shanai                 */ {},
    /* 112 Tinkle Bell            */ {},
    /* 113 Agogo                  */ {},
    /* 114 Steel Drums            */ {},
    /* 115 Woodblock              */ {},
    /* 116 Taiko Drum             */ {},
    /* 117 Melodic Tom            */ {},
    /* 118 Synth Drum             */ {},
    /* 119 Reverse Cymbal         */ {},
    /* 120 Guitar Fret Noise      */ {},
    /* 121 Breath Noise           */ {},
    /* 122 Seashore               */ {},
    /* 123 Bird Tweet             */ {},
    /* 124 Telephone Ring         */ {},
    /* 125 Helicopter             */ {},
    /* 126 Applause               */ {},
    /* 127 Gunshot                */ {},
}};

[[nodiscard]] inline auto pitchRangeForId(int32_t gmProgramId) -> PitchRange {
    if (gmProgramId < 0 || gmProgramId >= kNumInstruments)
        return {};
    return kPitchRanges[static_cast<size_t>(gmProgramId)];
}

} // namespace GeneralMidiPitchRanges
