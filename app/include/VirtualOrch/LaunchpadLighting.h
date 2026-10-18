#pragma once

#include <array>
#include <cstdint>
#include <optional>

/** Programmer-mode static palette colours for Launchpad Mini Mk3 / X. */
namespace LaunchpadLighting {

inline constexpr uint8_t kOff = 0;
inline constexpr uint8_t kWhite = 3;
inline constexpr uint8_t kRed = 5;
inline constexpr uint8_t kOrange = 9;
inline constexpr uint8_t kYellow = 13;
inline constexpr uint8_t kLime = 17;
inline constexpr uint8_t kGreen = 21;
inline constexpr uint8_t kMint = 29;
inline constexpr uint8_t kCyan = 37;
inline constexpr uint8_t kSky = 41;
inline constexpr uint8_t kBlue = 45;
inline constexpr uint8_t kPurple = 49;
inline constexpr uint8_t kMagenta = 53;
inline constexpr uint8_t kPink = 57;

/** Flush side button: always lit. */
inline constexpr uint8_t kFlush = kWhite;

/** Instrument families, taxonomy order: strings 0–6, woodwinds 7–10, brass 11–14, other 15–19. */
enum class Family : uint8_t { Strings, Woodwinds, Brass, Other };

[[nodiscard]] inline auto familyForLocalInstrument(int32_t localId) -> std::optional<Family> {
    if (localId >= 0 && localId <= 6)
        return Family::Strings;
    if (localId >= 7 && localId <= 10)
        return Family::Woodwinds;
    if (localId >= 11 && localId <= 14)
        return Family::Brass;
    if (localId >= 15 && localId <= 19)
        return Family::Other;
    return std::nullopt;
}

/** Strings yellow, woodwinds lime, brass orange, other (timpani, snare, …) mint. */
[[nodiscard]] inline auto colourForFamily(Family family) -> uint8_t {
    switch (family) {
        case Family::Strings:
            return kYellow;
        case Family::Woodwinds:
            return kLime;
        case Family::Brass:
            return kOrange;
        case Family::Other:
        default:
            return kMint;
    }
}

/** Instrument pad: lit in its family colour only when on. */
[[nodiscard]] inline auto colourForInstrumentPad(int32_t localId, bool on) -> uint8_t {
    const auto family = familyForLocalInstrument(localId);
    return on && family.has_value() ? colourForFamily(*family) : kOff;
}

/** TD levels 1..32, low → high: cyan, sky, blue, purple, magenta, pink. */
inline constexpr std::array<uint8_t, 6> kTdGradient{{kCyan, kSky, kBlue, kPurple, kMagenta, kPink}};

/** TD level pad: lit in its gradient colour only when selected. */
[[nodiscard]] inline auto colourForTdLevel(int levelIndex, bool selected) -> uint8_t {
    if (! selected || levelIndex < 0 || levelIndex >= static_cast<int>(kTdGradient.size()))
        return kOff;
    return kTdGradient[static_cast<size_t>(levelIndex)];
}

/** AUTO_TD side button: green when on, red when off. */
[[nodiscard]] inline auto colourForAutoTd(bool autoOn) -> uint8_t {
    return autoOn ? kGreen : kRed;
}

/** Edit/Jam pads: the selected mode is lit. */
[[nodiscard]] inline auto colourForModePad(bool selected) -> uint8_t {
    return selected ? kWhite : kOff;
}

} // namespace LaunchpadLighting
