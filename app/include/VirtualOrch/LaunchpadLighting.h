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

/** Flush side button (lit in Edit mode only). */
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

/** Dark grey: an instrument that is on but frozen (the user's side in Jam / Accompaniment). */
inline constexpr uint8_t kGrey = 1;

/**
 * A dimmer shade of a palette colour. The Mk3 palette groups each hue in fours from 4
 * (4k+1 full, 4k+2 dimmer, 4k+3 dimmest); white 3 dims to grey 2, then 1.
 * `level` 1 = dimmer, 2 = dimmest. Other colours come back unchanged.
 */
[[nodiscard]] inline auto dimmed(uint8_t colour, int level) -> uint8_t {
    level = level < 1 ? 1 : (level > 2 ? 2 : level);
    if (colour == kWhite)
        return static_cast<uint8_t>(kWhite - level);
    if (colour >= 4 && colour < 64 && (colour - 4) % 4 == 1)
        return static_cast<uint8_t>(colour + level);
    return colour;
}

/** How much the model's side is dimmed while the pads control yours (tune on the hardware). */
inline constexpr int kOtherSideDimLevel = 2;

/**
 * Instrument pad in the two-sided layout. `controlled` = the pads control this side in the
 * current mode (bright); otherwise the user's side shows grey and the model's side dim.
 */
[[nodiscard]] inline auto colourForSidedInstrumentPad(int32_t localId, bool on, bool controlled,
                                                      bool userSide) -> uint8_t {
    const auto bright = colourForInstrumentPad(localId, on);
    if (bright == kOff || controlled)
        return bright;
    return userSide ? kGrey : dimmed(bright, kOtherSideDimLevel);
}

/** Reset side button colour (Flush keeps kFlush). */
inline constexpr uint8_t kReset = kRed;

/** Flush / Reset side buttons: lit only in Edit mode, where they work. */
[[nodiscard]] inline auto colourForEditOnlyButton(bool editMode, uint8_t colour) -> uint8_t {
    return editMode ? colour : kOff;
}

/** Mood preset pad: full colour when active, dimmer when active but changed since, dimmest otherwise. */
[[nodiscard]] inline auto colourForPresetPad(uint8_t colour, bool active, bool modified) -> uint8_t {
    if (active && ! modified)
        return colour;
    return dimmed(colour, active ? 1 : 2);
}

/** TD levels 1..64, low → high: cyan, sky, blue, purple, magenta, pink, red. */
inline constexpr std::array<uint8_t, 7> kTdGradient{{kCyan, kSky, kBlue, kPurple, kMagenta, kPink, kRed}};

/** TD level pad: lit in its gradient colour only when selected. */
[[nodiscard]] inline auto colourForTdLevel(int levelIndex, bool selected) -> uint8_t {
    if (! selected || levelIndex < 0 || levelIndex >= static_cast<int>(kTdGradient.size()))
        return kOff;
    return kTdGradient[static_cast<size_t>(levelIndex)];
}

/** Register pads LOW / MEDIUM / HIGH (RegisterRange values 1..3): lit only when selected. */
[[nodiscard]] inline auto colourForRegister(int registerValue, bool selected) -> uint8_t {
    if (! selected)
        return kOff;
    switch (registerValue) {
        case 1: return kBlue;
        case 2: return kMint;
        case 3: return kMagenta;
        default: return kOff;
    }
}

/** AUTO_TD side button: green when on, red when off. */
[[nodiscard]] inline auto colourForAutoTd(bool autoOn) -> uint8_t {
    return autoOn ? kGreen : kRed;
}

/** Polyphony pads: only the selected one is lit, and only in Jam / Accompaniment (where they work). */
[[nodiscard]] inline auto colourForPolyphonyPad(bool active, bool selected) -> uint8_t {
    return active && selected ? kYellow : kOff;
}

/** Edit / Jam / Accompaniment pads: the selected mode is lit. */
[[nodiscard]] inline auto colourForModePad(bool selected) -> uint8_t {
    return selected ? kWhite : kOff;
}

} // namespace LaunchpadLighting
