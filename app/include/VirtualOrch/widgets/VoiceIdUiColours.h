#pragma once

#include <JuceHeader.h>

#include <cstdint>

/** Categorical colours for vocsep voiceId (piano roll + Vocsep table column). */
namespace VoiceIdUiColours {

inline constexpr juce::uint32 kPalette[] = {
    0xffe6194b, 0xff3cb44b, 0xffffe119, 0xff4363d8, 0xfff58231, 0xff911eb4, 0xff42d4f4,
    0xfff032e6, 0xffbfef45, 0xfffabed4, 0xff469990, 0xffdcbeff, 0xff9a6324, 0xfffffac8,
    0xff800000, 0xffaaffc3,
};
inline constexpr int kPaletteSize =
    static_cast<int>(sizeof(kPalette) / sizeof(kPalette[0]));

[[nodiscard]] inline auto forVoiceId(int32_t voiceId) -> juce::Colour {
    if (voiceId < 0)
        return juce::Colours::grey;
    return juce::Colour(kPalette[voiceId % kPaletteSize]);
}

} // namespace VoiceIdUiColours
