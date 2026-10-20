#include "VirtualOrch/ChordNaming.h"

#include <algorithm>
#include <array>
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

} // namespace

auto nameFor(std::vector<int> midiPitches) -> std::string {
    std::ranges::sort(midiPitches);
    std::vector<std::string> notes; // distinct pitch classes, lowest (the bass) first
    std::array<bool, 12> seen{};
    for (const int pitch: midiPitches) {
        const int pc = ((pitch % 12) + 12) % 12;
        if (seen[static_cast<size_t>(pc)])
            continue;
        seen[static_cast<size_t>(pc)] = true;
        notes.emplace_back(kSharpNames[static_cast<size_t>(pc)]);
    }
    if (notes.size() < 3)
        return {};

    std::string name;
    try {
        const ChordNamer::Chord chord(notes);
        if (chord.chordNames.empty())
            return {};
        name = chord.chordNames.front();
    } catch (const std::exception &) {
        return {};
    }

    respellAt(name, 0);
    if (const auto slash = name.find('/'); slash != std::string::npos)
        respellAt(name, slash + 1);
    return name;
}

auto rootIsFlat(const std::string &name) -> bool {
    return name.size() >= 2 && name[1] == 'b' && name[0] >= 'A' && name[0] <= 'G';
}

} // namespace ChordNaming
