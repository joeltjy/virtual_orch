#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "VirtualOrch/reduction/MelodyBassSkyline.h"

/**
 * Layout logic of the ScoreWidget (no drawing): a rolling grand staff of the reduction's
 * generated notes with a nowbar, noteheads only (no stems, beams or rests).
 *
 *  - Chords: notes whose onsets are within kChordToleranceCs of the previous note of the chord
 *    form one chord; its skyline note is the highest one.
 *  - Staff: MIDI 60 (middle C) and above on the treble staff, below it on the bass staff.
 *  - Step: diatonic position, 7 per octave (C = 0 … B = 6, octave = midi / 12), so a line or
 *    space is one step. Treble lines are E4 G4 B4 D5 F5, bass lines G2 B2 D3 F3 A3.
 *  - Spelling: sharps, or flats for a chord whose name's root is flat.
 *  - Melody + bass (what the score shows): onsets within kSnapCs (30 ms) of the previous one
 *    are snapped to the first onset of their cluster, then MelodyBass::skylineAndBass (the port
 *    of ~/orchestration melody_bass.py, skyline_and_bass) keeps the top and bottom lines; a note
 *    that continues after a cut is shown once, at its first slice.
 */
namespace ScoreLayout {

inline constexpr int32_t kChordToleranceCs = 5;
inline constexpr int32_t kSnapCs = 3; // 30 ms
inline constexpr int kMiddleC = 60;

/** Bottom / top line steps of each staff. */
inline constexpr int kTrebleBottomStep = 37; // E4
inline constexpr int kTrebleTopStep = 45;    // F5
inline constexpr int kBassBottomStep = 25;   // G2
inline constexpr int kBassTopStep = 33;      // A3

struct Note {
    int32_t onsetCs = 0;
    int32_t durationCs = 0;
    int pitch = 60;
};

struct Chord {
    int32_t onsetCs = 0;
    std::vector<int> pitches; // ascending, duplicates kept once
    [[nodiscard]] auto topPitch() const -> int { return pitches.empty() ? 0 : pitches.back(); }
};

/** Onsets chained within `toleranceCs` of the previous one move to their cluster's first onset. */
[[nodiscard]] inline auto snapOnsets(std::vector<Note> notes, int32_t toleranceCs = kSnapCs) -> std::vector<Note> {
    std::ranges::stable_sort(notes, [](const Note &a, const Note &b) { return a.onsetCs < b.onsetCs; });
    int32_t clusterStart = 0;
    int32_t previous = 0;
    for (size_t i = 0; i < notes.size(); ++i) {
        const int32_t onset = notes[i].onsetCs;
        if (i == 0 || onset - previous > toleranceCs)
            clusterStart = onset;
        previous = onset;
        const int32_t end = onset + notes[i].durationCs;
        notes[i].onsetCs = clusterStart;
        notes[i].durationCs = std::max(0, end - clusterStart); // the end time is kept
    }
    return notes;
}

/** The top (melody) and bottom (bass) lines of a passage, one entry per note. */
struct Lines {
    std::vector<Note> melody;
    std::vector<Note> bass;
};

/**
 * Melody and bass of `notes` (after snapOnsets): ~/orchestration's skyline_and_bass, one entry
 * per note at its onset, with its length in that line (a later cut shortens it). The two lines
 * never share a note.
 */
[[nodiscard]] inline auto skylineAndBass(std::vector<Note> notes, int32_t snapCs = kSnapCs) -> Lines {
    notes = snapOnsets(std::move(notes), snapCs);
    std::ranges::stable_sort(notes, [](const Note &a, const Note &b) {
        if (a.onsetCs != b.onsetCs)
            return a.onsetCs < b.onsetCs;
        if (a.pitch != b.pitch)
            return a.pitch < b.pitch;
        return a.durationCs < b.durationCs;
    });
    std::vector<MelodyBass::Note> input;
    for (size_t i = 0; i < notes.size(); ++i)
        if (notes[i].durationCs > 0)
            input.push_back({static_cast<int>(i), notes[i].onsetCs, notes[i].onsetCs + notes[i].durationCs,
                             notes[i].pitch, 64});
    const auto merged = [](const std::vector<MelodyBass::Segment> &segments) {
        std::vector<Note> out;
        int lastSource = -1;
        for (const auto &seg: segments) {
            if (seg.sourceIndex == lastSource && ! out.empty()) {
                out.back().durationCs = seg.end - out.back().onsetCs; // a continuation: same note
                continue;
            }
            out.push_back({seg.start, seg.end - seg.start, seg.pitch});
            lastSource = seg.sourceIndex;
        }
        return out;
    };
    const auto result = MelodyBass::skylineAndBass(input);
    return {merged(result.melody), merged(result.bass)};
}

/** The skyline (top line) only. */
[[nodiscard]] inline auto skyline(std::vector<Note> notes, int32_t snapCs = kSnapCs) -> std::vector<Note> {
    return skylineAndBass(std::move(notes), snapCs).melody;
}

/** Group notes into chords by onset (chained within kChordToleranceCs), in onset order. */
[[nodiscard]] inline auto groupChords(std::vector<Note> notes) -> std::vector<Chord> {
    std::ranges::stable_sort(notes, [](const Note &a, const Note &b) { return a.onsetCs < b.onsetCs; });
    std::vector<Chord> chords;
    int32_t lastOnset = 0;
    for (const auto &n: notes) {
        if (chords.empty() || n.onsetCs - lastOnset > kChordToleranceCs) {
            chords.push_back({n.onsetCs, {}});
        }
        auto &pitches = chords.back().pitches;
        if (std::ranges::find(pitches, n.pitch) == pitches.end())
            pitches.push_back(n.pitch);
        lastOnset = n.onsetCs;
    }
    for (auto &c: chords)
        std::ranges::sort(c.pitches);
    return chords;
}

struct Spelled {
    int step = 0;       // diatonic position (see above)
    int accidental = 0; // -1 flat, 0 natural, +1 sharp
};

[[nodiscard]] inline auto spell(int pitch, bool flats) -> Spelled {
    // pitch class -> (degree, accidental)
    static constexpr int sharpDegree[12] = {0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6};
    static constexpr int sharpAcc[12] = {0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0};
    static constexpr int flatDegree[12] = {0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6};
    static constexpr int flatAcc[12] = {0, -1, 0, -1, 0, 0, -1, 0, -1, 0, -1, 0};
    const int pc = ((pitch % 12) + 12) % 12;
    const int octave = (pitch - pc) / 12;
    const int degree = flats ? flatDegree[pc] : sharpDegree[pc];
    return {octave * 7 + degree, flats ? flatAcc[pc] : sharpAcc[pc]};
}

[[nodiscard]] inline auto onTreble(int pitch) -> bool { return pitch >= kMiddleC; }

/** Ledger-line steps needed for a note at `step` on a staff with these bottom / top lines. */
[[nodiscard]] inline auto ledgerSteps(int step, int bottomStep, int topStep) -> std::vector<int> {
    std::vector<int> out;
    for (int s = bottomStep - 2; s >= step; s -= 2)
        out.push_back(s);
    for (int s = topStep + 2; s <= step; s += 2)
        out.push_back(s);
    return out;
}

/**
 * Noteheads of a chord on one staff that sit a step apart (a second) cannot share a column:
 * walking up from the bottom, a note one step above an unshifted note moves to the right.
 * Returns, per entry of `steps` (ascending), whether that head is shifted.
 */
[[nodiscard]] inline auto secondShifts(const std::vector<int> &steps) -> std::vector<bool> {
    std::vector<bool> shifted(steps.size(), false);
    for (size_t i = 1; i < steps.size(); ++i)
        if (steps[i] - steps[i - 1] == 1 && ! shifted[i - 1])
            shifted[i] = true;
    return shifted;
}

/** x of an onset, with `nowX` at the clock and `pxPerCs` horizontal scale. */
[[nodiscard]] inline auto xFor(int32_t onsetCs, int32_t nowCs, float nowX, float pxPerCs) -> float {
    return nowX + static_cast<float>(onsetCs - nowCs) * pxPerCs;
}

} // namespace ScoreLayout
