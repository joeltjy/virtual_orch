#pragma once

#include <JuceHeader.h>

#include <cstdint>
#include <optional>

#include "VirtualOrch/LaunchpadLighting.h"

/**
 * On-screen approximations of the Launchpad palette colours, so UI instrument colours
 * match the pads: strings yellow, woodwinds lime, brass orange, other mint.
 */
namespace InstrumentUiColours {

[[nodiscard]] inline auto fromPalette(uint8_t palette) -> juce::Colour {
    switch (palette) {
        case LaunchpadLighting::kWhite:
            return juce::Colour(0xffe8e8e8);
        case LaunchpadLighting::kRed:
            return juce::Colour(0xffe63b3b);
        case LaunchpadLighting::kOrange:
            return juce::Colour(0xffff8c1a);
        case LaunchpadLighting::kYellow:
            return juce::Colour(0xfff0c820);
        case LaunchpadLighting::kLime:
            return juce::Colour(0xff9be22e);
        case LaunchpadLighting::kGreen:
            return juce::Colour(0xff2ecf5a);
        case LaunchpadLighting::kMint:
            return juce::Colour(0xff2ed8a0);
        case LaunchpadLighting::kCyan:
            return juce::Colour(0xff2ee6e6);
        case LaunchpadLighting::kSky:
            return juce::Colour(0xff3fb4ff);
        case LaunchpadLighting::kBlue:
            return juce::Colour(0xff3b7cff);
        case LaunchpadLighting::kPurple:
            return juce::Colour(0xff8a5cff);
        case LaunchpadLighting::kMagenta:
            return juce::Colour(0xffd84fe0);
        case LaunchpadLighting::kPink:
            return juce::Colour(0xffff5c9a);
        case LaunchpadLighting::kOff:
        default:
            return juce::Colour(0xff1c1c1c);
    }
}

[[nodiscard]] inline auto forLocalInstrument(int32_t localId) -> juce::Colour {
    if (const auto family = LaunchpadLighting::familyForLocalInstrument(localId))
        return fromPalette(LaunchpadLighting::colourForFamily(*family));
    return juce::Colour(0xff9aa0a6);
}

} // namespace InstrumentUiColours
