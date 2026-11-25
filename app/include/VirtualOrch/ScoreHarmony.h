#pragma once

#include <algorithm>
#include <array>
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
 *  - A chord at every onset group, from the notes sounding there: the group's notes and the
 *    earlier ones sounding together with them (ScoreLayout::soundTogether). In order (Joel,
 *    2026-11-25, docs/score_viewer_chords.md in the orchestration repo):
 *      1. If every pitch class sounding is a tone of the previous chord, it stays that chord
 *         (not after an N/C).
 *      2. Else a triad of the key (or the runner-up key) holding every pitch class sounding;
 *      3. else a seventh chord of either key (keyChords lists both);
 *         ties: root = lowest note, then I / IV / V, then the previous chord, then the detected key.
 *      4. Else the usual name (ChordNaming): its root is spelled in the key (Spelling::root), and
 *         a power chord ("5") gets its third from the key, else from a recent chord on the same
 *         root, else stays "5".
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
    std::set<int> tones;      // the chord's pitch classes (a key chord's full set, else those sounding)
};

/** A chord of a key: root, qualities, pitch classes. */
struct KeyChord {
    int rootPc = 0;
    int rootRel = 0;          // root above the tonic, semitones
    bool seventh = false;
    std::string fullQuality;  // "", "m", "dim", "aug", "maj7", "7", "m7", "m7b5", "dim7"
    std::string quality;      // shown ("", "m", "dim", "aug")
    std::set<int> tones;
};

/**
 * The chords of a key. Major: the scale's triads and sevenths (I, ii, iii, IV, V, vi, vii°).
 * Minor: natural minor's triads and sevenths, plus from the raised 7th the V and V7, vii° and
 * vii°7, and III+. No raised-6th chords and no sus chords (Joel).
 */
[[nodiscard]] inline auto keyChords(const Spelling::Key &key) -> std::vector<KeyChord> {
    struct Def {
        int rel;
        const char *full;
    };
    static const std::vector<Def> kMajor{{0, ""}, {2, "m"}, {4, "m"}, {5, ""}, {7, ""}, {9, "m"}, {11, "dim"},
                                         {0, "maj7"}, {2, "m7"}, {4, "m7"}, {5, "maj7"}, {7, "7"}, {9, "m7"}, {11, "m7b5"}};
    static const std::vector<Def> kMinor{{0, "m"}, {2, "dim"}, {3, ""}, {5, "m"}, {7, "m"}, {8, ""}, {10, ""},
                                         {7, ""}, {11, "dim"}, {3, "aug"},
                                         {0, "m7"}, {2, "m7b5"}, {3, "maj7"}, {5, "m7"}, {7, "m7"}, {8, "maj7"}, {10, "7"},
                                         {7, "7"}, {11, "dim7"}};
    static const std::map<std::string, std::pair<std::vector<int>, std::string>> kShape{
        {"", {{0, 4, 7}, ""}},          {"m", {{0, 3, 7}, "m"}},        {"dim", {{0, 3, 6}, "dim"}},
        {"aug", {{0, 4, 8}, "aug"}},    {"maj7", {{0, 4, 7, 11}, ""}},  {"7", {{0, 4, 7, 10}, ""}},
        {"m7", {{0, 3, 7, 10}, "m"}},   {"m7b5", {{0, 3, 6, 10}, "dim"}}, {"dim7", {{0, 3, 6, 9}, "dim"}}};
    std::vector<KeyChord> out;
    const int tonic = Spelling::tonicPc(key);
    for (const auto &d: key.minor ? kMinor : kMajor) {
        const auto &[intervals, shown] = kShape.at(d.full);
        KeyChord c;
        c.rootRel = d.rel;
        c.rootPc = Spelling::mod(tonic + d.rel, 12);
        c.seventh = intervals.size() == 4;
        c.fullQuality = d.full;
        c.quality = shown;
        for (const int i: intervals)
            c.tones.insert(Spelling::mod(c.rootPc + i, 12));
        out.push_back(std::move(c));
    }
    return out;
}

/**
 * The key chord for these pitch classes (steps 2 and 3 above), or nullopt. `keys` is the detected
 * key first, then the runner-up.
 */
[[nodiscard]] inline auto matchInKeys(const std::set<int> &pcs, int bassPc, const std::vector<Spelling::Key> &keys,
                                      const std::set<int> *previousTones)
    -> std::optional<std::pair<KeyChord, Spelling::Key>> {
    for (const bool seventh: {false, true}) {
        std::optional<std::pair<KeyChord, Spelling::Key>> best;
        std::array<int, 4> bestScore{};
        for (size_t k = 0; k < keys.size(); ++k)
            for (const auto &c: keyChords(keys[k])) {
                if (c.seventh != seventh || ! std::includes(c.tones.begin(), c.tones.end(), pcs.begin(), pcs.end()))
                    continue;
                const std::array<int, 4> score{c.rootPc == bassPc ? 1 : 0,
                                               (c.rootRel == 0 || c.rootRel == 5 || c.rootRel == 7) ? 1 : 0,
                                               (previousTones != nullptr && *previousTones == c.tones) ? 1 : 0,
                                               k == 0 ? 1 : 0};
                if (! best.has_value() || score > bestScore) {
                    best = std::pair{c, keys[k]};
                    bestScore = score;
                }
            }
        if (best.has_value())
            return best;
    }
    return std::nullopt;
}

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

/**
 * The chord at every onset group of `notes` (prepared), spelled in `key`. `runnerUp`: a close
 * second key, also matched. `matchInKey` false (no key detected yet): skip the key chords.
 */
[[nodiscard]] inline auto chords(const std::vector<Note> &notes, const Spelling::Key &key,
                                 const NameFn &nameFor = ChordNaming::nameFor,
                                 const std::optional<Spelling::Key> &runnerUp = std::nullopt,
                                 bool matchInKey = true) -> std::vector<Chord> {
    std::vector<Chord> out;
    std::vector<Spelling::Key> keys{key};
    if (runnerUp.has_value() && ! (*runnerUp == key))
        keys.push_back(*runnerUp);
    std::optional<size_t> previous;  // the last chord (not mono), unless an N/C came after it
    std::optional<int32_t> monoStart; // the first onset of the current run of mono onsets
    int monoCount = 0;
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
        if (c.mono) {
            if (! monoStart.has_value()) {
                monoStart = c.onsetCs;
                monoCount = 0;
            }
            ++monoCount;
            if (monoCount >= kNoChordOnsets || c.endCs - *monoStart >= kNoChordCs)
                previous.reset(); // an N/C: the next chord is named afresh
            out.push_back(std::move(c));
            continue;
        }
        monoStart.reset();
        const std::set<int> *previousTones = previous.has_value() ? &out[*previous].tones : nullptr;
        if (previousTones != nullptr && std::includes(previousTones->begin(), previousTones->end(), pcs.begin(), pcs.end())) {
            // 1. still the previous chord
            const auto &p = out[*previous];
            c.rootLof = p.rootLof;
            c.fullQuality = p.fullQuality;
            c.quality = p.quality;
            c.text = p.text;
            c.tones = p.tones;
        } else if (const auto match = matchInKey ? matchInKeys(pcs, Spelling::mod(c.pitches.front(), 12), keys, previousTones)
                                                 : std::nullopt) {
            // 2-3. a chord of the key
            const auto &[kc, inKey] = *match;
            c.rootLof = Spelling::root(kc.rootPc, inKey, kc.quality == "dim");
            c.fullQuality = kc.fullQuality;
            c.quality = kc.quality;
            c.text = Spelling::nameOf(c.rootLof) + c.quality;
            c.tones = kc.tones;
        } else {
            // 4. the usual name
            c.tones = pcs;
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
        if (! c.mono)
            previous = out.size();
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
