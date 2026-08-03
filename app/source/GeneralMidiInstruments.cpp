#include "VirtualOrch/GeneralMidiInstruments.h"

#include <array>

#include "VirtualOrchBinaryData.h"

namespace GeneralMidiInstruments {
namespace {

auto loadNames() -> std::array<juce::String, kNumInstruments> {
    std::array<juce::String, kNumInstruments> names{};
    for (int32_t i = 0; i < kNumInstruments; ++i)
        names[static_cast<size_t>(i)] = "Instrument " + juce::String(i);

    const juce::String csv(VirtualOrchBinaryData::general_midi_instruments_csv,
                           static_cast<size_t>(VirtualOrchBinaryData::general_midi_instruments_csvSize));

    for (const auto &line: juce::StringArray::fromLines(csv)) {
        const auto trimmed = line.trim();
        if (trimmed.isEmpty() || trimmed.startsWith("instrument_id"))
            continue;

        const auto comma = trimmed.indexOfChar(',');
        if (comma <= 0)
            continue;

        const auto id = trimmed.substring(0, comma).getIntValue();
        if (id < 0 || id >= kNumInstruments)
            continue;

        names[static_cast<size_t>(id)] = trimmed.substring(comma + 1).trim();
    }

    return names;
}

auto names() -> const std::array<juce::String, kNumInstruments> & {
    static const auto cached = loadNames();
    return cached;
}

} // namespace

auto nameForId(int32_t id) -> juce::String {
    if (id < 0 || id >= kNumInstruments)
        return "Instrument " + juce::String(id);
    return names()[static_cast<size_t>(id)];
}

} // namespace GeneralMidiInstruments
