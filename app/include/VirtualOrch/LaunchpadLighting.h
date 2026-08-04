#pragma once

#include <cstdint>

/** Programmer-mode static palette colours for Launchpad Mini Mk3 / X. */
namespace LaunchpadLighting {

inline constexpr uint8_t kOff = 0;
inline constexpr uint8_t kPink = 53;
inline constexpr uint8_t kGreen = 21;
inline constexpr uint8_t kBlue = 45;
inline constexpr uint8_t kYellow = 13;
inline constexpr uint8_t kRed = 5;

/** Row colour: 0/4 pink, 1/5 green, 2/6 blue, 3/7 yellow. */
[[nodiscard]] inline auto colourForRow(int rowIdx) -> uint8_t {
    switch (rowIdx % 4) {
        case 0:
            return kPink;
        case 1:
            return kGreen;
        case 2:
            return kBlue;
        default:
            return kYellow;
    }
}

/** Lit only when effective (grid × side) is on. */
[[nodiscard]] inline auto colourForGridEffective(int rowIdx, bool effectiveOn) -> uint8_t {
    return effectiveOn ? colourForRow(rowIdx) : kOff;
}

/** Side column: muted → red; unmuted → off. */
[[nodiscard]] inline auto colourForSide(bool unmuted) -> uint8_t {
    return unmuted ? kOff : kRed;
}

} // namespace LaunchpadLighting
