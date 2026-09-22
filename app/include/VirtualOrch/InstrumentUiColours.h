#pragma once

#include <JuceHeader.h>

#include <cstdint>
#include <optional>

#include "VirtualOrch/LaunchpadLighting.h"

/**
 * UI colours matching Launchpad row families (AppSession pad layout):
 * strings pink, woodwinds green, brass blue, other yellow.
 */
namespace InstrumentUiColours {

[[nodiscard]] inline auto fromPalette(uint8_t palette) -> juce::Colour {
    switch (palette) {
        case LaunchpadLighting::kPink:
            return juce::Colour(0xfff06ab0);
        case LaunchpadLighting::kGreen:
            return juce::Colour(0xff2ecf5a);
        case LaunchpadLighting::kBlue:
            return juce::Colour(0xff3b7cff);
        case LaunchpadLighting::kYellow:
            return juce::Colour(0xfff0c820);
        case LaunchpadLighting::kRed:
            return juce::Colour(0xffe63b3b);
        case LaunchpadLighting::kOff:
        default:
            return juce::Colour(0xff1c1c1c);
    }
}

/** Family row 0..3 matching LaunchpadLighting::colourForRow, if known. */
[[nodiscard]] inline auto familyRowForLocalInstrument(int32_t localId) -> std::optional<int> {
    if (localId >= 0 && localId <= 6)
        return 0; // strings
    if (localId >= 7 && localId <= 10)
        return 1; // woodwinds
    if (localId >= 11 && localId <= 14)
        return 2; // brass
    if (localId >= 15 && localId <= 19)
        return 3; // other (includes piano 15)
    return std::nullopt;
}

[[nodiscard]] inline auto forLocalInstrument(int32_t localId) -> juce::Colour {
    if (const auto row = familyRowForLocalInstrument(localId))
        return fromPalette(LaunchpadLighting::colourForRow(*row));
    return juce::Colour(0xff9aa0a6);
}

} // namespace InstrumentUiColours
