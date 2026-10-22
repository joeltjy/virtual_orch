#include "VirtualOrch/ChordNaming.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>

#include "chord.h"

namespace ChordNaming {

namespace {

constexpr std::array<const char *, 12> kSharpNames{
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};

/** Sharp spellings that read better flat, and their flat names. */
constexpr std::array<std::pair<const char *, const char *>, 4> kFlatRespell{{
    {"C#", "Db"}, {"D#", "Eb"}, {"G#", "Ab"}, {"A#", "Bb"}}};

/** Respell the note name at `pos` (letter + optional '#') if it is one of kFlatRespell. */
auto respellAt(std::string &name, size_t pos) -> void {
    if (pos + 1 >= name.size() || name[pos + 1] != '#')
        return;
    const std::string note = name.substr(pos, 2);
    for (const auto &[sharp, flat]: kFlatRespell)
        if (note == sharp) {
            name.replace(pos, 2, flat);
            return;
        }
}

constexpr std::array<int, 7> kLetterPc{0, 2, 4, 5, 7, 9, 11}; // C D E F G A B
constexpr const char *kLetters = "CDEFGAB";

/** The note name at `pos` (letter + optional '#' / 'b'): letter index and pitch class, or false. */
auto parseNote(const std::string &name, size_t pos, int &letter, int &pc, size_t &length) -> bool {
    if (pos >= name.size())
        return false;
    const char *found = std::strchr(kLetters, name[pos]);
    if (found == nullptr || *found == '\0')
        return false;
    letter = static_cast<int>(found - kLetters);
    pc = kLetterPc[static_cast<size_t>(letter)];
    length = 1;
    if (pos + 1 < name.size() && (name[pos + 1] == '#' || name[pos + 1] == 'b')) {
        pc += name[pos + 1] == '#' ? 1 : -1;
        length = 2;
    }
    pc = ((pc % 12) + 12) % 12;
    return true;
}

/**
 * Spell the slash bass from its interval above the root, so it agrees with the root's key:
 * B/D# (a third above B is a D), E/G#, Eb/G, Db/F. Ambiguous intervals take the usual chord
 * spelling (b5, b6, b7), except #5 in an augmented chord and #4 next to a #11. A bass that would
 * need a double sharp / flat keeps the plain spelling (sharp, or Db / Eb / Ab / Bb).
 */
auto spellBassAt(std::string &name, size_t pos) -> void {
    int rootLetter = 0, rootPc = 0, bassLetter = 0, bassPc = 0;
    size_t rootLength = 0, bassLength = 0;
    if (!parseNote(name, 0, rootLetter, rootPc, rootLength) || !parseNote(name, pos, bassLetter, bassPc, bassLength)) {
        respellAt(name, pos);
        return;
    }
    const std::string quality = name.substr(rootLength, pos - 1 - rootLength);
    const int interval = ((bassPc - rootPc) % 12 + 12) % 12;
    static constexpr std::array<int, 12> kSteps{0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6};
    int steps = kSteps[static_cast<size_t>(interval)];
    if (interval == 8 && (quality.find("aug") != std::string::npos || quality.find('+') != std::string::npos ||
                          quality.find("#5") != std::string::npos))
        steps = 4;
    if (interval == 6 && quality.find("#11") != std::string::npos)
        steps = 3;
    const int letter = (rootLetter + steps) % 7;
    int accidental = bassPc - kLetterPc[static_cast<size_t>(letter)];
    if (accidental > 6)
        accidental -= 12;
    if (accidental < -6)
        accidental += 12;
    if (accidental < -1 || accidental > 1) {
        respellAt(name, pos);
        return;
    }
    std::string spelled(1, kLetters[letter]);
    if (accidental != 0)
        spelled += accidental > 0 ? '#' : 'b';
    name.replace(pos, bassLength, spelled);
}

} // namespace

namespace {

/** The CppChordNamer name of these pitch classes (bass first), or empty. */
auto namerName(const std::vector<int> &pcs) -> std::string {
    std::vector<std::string> notes;
    for (const int pc: pcs)
        notes.emplace_back(kSharpNames[static_cast<size_t>(pc)]);
    try {
        const ChordNamer::Chord chord(notes);
        return chord.chordNames.empty() ? std::string() : chord.chordNames.front();
    } catch (const std::exception &) {
        return {};
    }
}

/** A two-pitch-class name from the interval above the bass (sharp spelling). */
auto dyadName(int bassPc, int otherPc) -> std::string {
    const int interval = ((otherPc - bassPc) % 12 + 12) % 12;
    const std::string bass = kSharpNames[static_cast<size_t>(bassPc)];
    const std::string other = kSharpNames[static_cast<size_t>(otherPc)];
    switch (interval) {
        case 1: return other + "maj7/" + bass; // B-C: C over its major seventh
        case 2: return bass + "sus2";
        case 3: return bass + "m";
        case 4: return bass;
        case 5: return other + "/" + bass;    // C-F: F over its fifth
        case 6: return bass + "dim";
        case 7: return bass + "5";
        case 8: return other + "/" + bass;    // C-Ab: Ab over its third
        case 9: return other + "m/" + bass;   // C-A: Am over its third
        case 10: return bass + "7";
        case 11: return bass + "maj7";
        default: return bass;
    }
}

/**
 * Name pitch classes (bass first): the namer; if it finds nothing, the same with one upper
 * note left out (each in turn, recursively, down to two notes); two notes are a dyad.
 */
auto inferName(const std::vector<int> &pcs) -> std::string {
    if (pcs.size() == 1)
        return kSharpNames[static_cast<size_t>(pcs.front())];
    if (pcs.size() == 2)
        return dyadName(pcs[0], pcs[1]);
    if (auto name = namerName(pcs); ! name.empty())
        return name;
    // leave out an upper note, the highest first (inner voices matter more)
    for (size_t drop = pcs.size() - 1; drop >= 1; --drop) {
        std::vector<int> fewer;
        for (size_t k = 0; k < pcs.size(); ++k)
            if (k != drop)
                fewer.push_back(pcs[k]);
        if (fewer.size() >= 3)
            if (auto name = namerName(fewer); ! name.empty())
                return name;
    }
    std::vector<int> fewer(pcs.begin(), pcs.end() - 1);
    return inferName(fewer);
}

} // namespace

auto nameFor(std::vector<int> midiPitches) -> std::string {
    if (midiPitches.size() < 2)
        return {};
    std::ranges::sort(midiPitches);
    std::vector<int> pcs; // distinct pitch classes, lowest (the bass) first
    std::array<bool, 12> seen{};
    for (const int pitch: midiPitches) {
        const int pc = ((pitch % 12) + 12) % 12;
        if (seen[static_cast<size_t>(pc)])
            continue;
        seen[static_cast<size_t>(pc)] = true;
        pcs.push_back(pc);
    }
    std::string name = inferName(pcs);
    if (name.empty())
        return {};
    respellAt(name, 0);
    if (const auto slash = name.find('/'); slash != std::string::npos)
        spellBassAt(name, slash + 1);
    return name;
}

auto rootIsFlat(const std::string &name) -> bool {
    return name.size() >= 2 && name[1] == 'b' && name[0] >= 'A' && name[0] <= 'G';
}

} // namespace ChordNaming
