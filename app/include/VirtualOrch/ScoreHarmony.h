#pragma once

#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "VirtualOrch/ChordNaming.h"
#include "VirtualOrch/ScoreLayout.h"
#include "VirtualOrch/Spelling.h"

/**
 * Harmony for the score, from notes prepared by ScoreLayout::prepare (snapped, sorted):
 *
 *  - A chord at every onset group, named from the notes sounding there: the group's notes and the
 *    earlier ones sounding together with them (ScoreLayout::soundTogether). Its root is spelled
 *    in the key (Spelling::root); a power chord ("5") gets its third from the key, else from a
 *    recent chord on the same root, else stays "5".
 *  - Labels: a chord's name is shown only when it differs from the last one shown. One pitch class
 *    sounding (one line, octaves included) for at least kNoChordOnsets onsets or kNoChordCs shows
 *    "N/C" once, at the first of those onsets.
 *  - Spelling of the melody and bass notes: a chord tone by its interval above the chord's root,
 *    anything else as a chromatic step or in the key (Spelling::passing).
 */
namespace ScoreHarmony {

inline constexpr int kNoChordOnsets = 2;
inline constexpr int32_t kNoChordCs = 50;    // 500 ms
inline constexpr int32_t kRecentRootCs = 300; // a "5" chord borrows the third of a chord this recent

using Note = ScoreLayout::Note;
using NameFn = std::function<std::string(const std::vector<int> &)>;

struct Chord {
    int32_t onsetCs = 0;
    int32_t endCs = 0;
    std::vector<int> pitches; // sounding, ascending, each pitch once
    bool mono = false;        // a single pitch class
    int rootLof = 0;
    std::string fullQuality;  // the namer's quality, without the slash bass ("m7", "dim7", "5")
    std::string quality;      // shown: root + basic quality ("", "m", "dim", "sus4", ...)
    std::string text;         // shown name, "" for a mono onset
};

struct Label {
    int32_t onsetCs = 0;
    std::string text;
};

/** Root pitch class of a chord name ("C#m7/E" -> 1), and the length of the root's spelling. */
[[nodiscard]] inline auto rootOf(const std::string &name) -> std::optional<std::pair<int, size_t>> {
    if (name.empty() || name[0] < 'A' || name[0] > 'G')
        return std::nullopt;
    static constexpr int kPc[7] = {9, 11, 0, 2, 4, 5, 7}; // A B C D E F G
    int pc = kPc[name[0] - 'A'];
    size_t len = 1;
    if (name.size() > 1 && (name[1] == '#' || name[1] == 'b')) {
        pc += name[1] == '#' ? 1 : -1;
        len = 2;
    }
    return std::pair{Spelling::mod(pc, 12), len};
}

/** The chord at every onset group of `notes` (prepared), spelled in `key`. */
[[nodiscard]] inline auto chords(const std::vector<Note> &notes, const Spelling::Key &key,
                                 const NameFn &nameFor = ChordNaming::nameFor) -> std::vector<Chord> {
    std::vector<Chord> out;
    for (const auto &g: ScoreLayout::onsetGroups(notes)) {
        Chord c;
        c.onsetCs = g.onsetCs;
        std::set<int> pitches, pcs;
        for (const size_t i: g.members) {
            pitches.insert(notes[i].pitch);
            c.endCs = std::max(c.endCs, notes[i].onsetCs + notes[i].durationCs);
        }
        for (const size_t i: g.held)
            pitches.insert(notes[i].pitch);
        for (const int p: pitches)
            pcs.insert(Spelling::mod(p, 12));
        c.pitches.assign(pitches.begin(), pitches.end());
        c.mono = pcs.size() <= 1;
        if (! c.mono) {
            const auto name = nameFor(c.pitches);
            if (const auto root = rootOf(name)) {
                const auto chordPart = ChordNaming::withoutBass(name);
                c.fullQuality = chordPart.substr(root->second);
                const auto basic = ChordNaming::basicChord(name);
                c.quality = basic.size() >= root->second ? basic.substr(root->second) : std::string();
                if (c.quality == "5") {
                    const Spelling::Key k = key;
                    const bool major = Spelling::inScale(root->first + 4, k);
                    const bool minor = Spelling::inScale(root->first + 3, k);
                    if (major != minor) {
                        c.quality = major ? "" : "m";
                    } else {
                        for (auto it = out.rbegin(); it != out.rend() && c.onsetCs - it->onsetCs <= kRecentRootCs; ++it)
                            if (! it->mono && Spelling::pcOf(it->rootLof) == root->first && it->quality != "5") {
                                c.quality = it->quality;
                                break;
                            }
                    }
                }
                c.rootLof = Spelling::root(root->first, key, c.quality.rfind("dim", 0) == 0);
                c.text = Spelling::nameOf(c.rootLof) + c.quality;
            } else {
                c.mono = true; // the namer found nothing: treat as no chord
            }
        }
        out.push_back(std::move(c));
    }
    return out;
}

/** The labels to show for these chords: changes only, and N/C for a single line (see top). */
[[nodiscard]] inline auto labels(const std::vector<Chord> &chordsIn) -> std::vector<Label> {
    std::vector<Label> out;
    std::string shown;
    std::optional<size_t> runStart;
    int runCount = 0;
    for (size_t i = 0; i < chordsIn.size(); ++i) {
        const auto &c = chordsIn[i];
        if (c.mono) {
            if (! runStart.has_value()) {
                runStart = i;
                runCount = 0;
            }
            ++runCount;
            const int32_t stretchEnd = i + 1 < chordsIn.size() ? chordsIn[i + 1].onsetCs : c.endCs;
            const int32_t start = chordsIn[*runStart].onsetCs;
            if (shown != "N/C" && (runCount >= kNoChordOnsets || stretchEnd - start >= kNoChordCs)) {
                out.push_back({start, "N/C"});
                shown = "N/C";
            }
            continue;
        }
        runStart.reset();
        if (c.text != shown) {
            out.push_back({c.onsetCs, c.text});
            shown = c.text;
        }
    }
    return out;
}

/** A line's notes (melody or bass, in onset order) spelled: line-of-fifths position per note. */
[[nodiscard]] inline auto spellLine(const std::vector<Note> &line, const std::vector<Chord> &chordsIn,
                                    const Spelling::Key &key) -> std::vector<int> {
    std::map<int32_t, const Chord *> at;
    for (const auto &c: chordsIn)
        at[c.onsetCs] = &c;
    std::vector<int> out;
    std::optional<int> previous;
    for (const auto &n: line) {
        const int pc = Spelling::mod(n.pitch, 12);
        std::optional<int> lof;
        const auto it = at.find(n.onsetCs);
        if (it != at.end() && ! it->second->mono) {
            const auto &c = *it->second;
            const bool tone = std::any_of(c.pitches.begin(), c.pitches.end(),
                                          [pc](int p) { return Spelling::mod(p, 12) == pc; });
            if (tone)
                lof = Spelling::chordTone(pc, c.rootLof, Spelling::qualityOf(c.fullQuality));
        }
        out.push_back(lof.value_or(Spelling::passing(n.pitch, previous, key)));
        previous = n.pitch;
    }
    return out;
}

} // namespace ScoreHarmony
