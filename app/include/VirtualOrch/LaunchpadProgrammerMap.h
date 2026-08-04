#pragma once

#include <optional>

/** Novation Launchpad Programmer-mode MIDI map (Mini Mk3 / X-style).
 *
 *  8x8 pads: notes 11-18 … 81-88 (row 0 = top).
 *  Right side column: notes 19,29,...,89 → col = 9.
 *  Top row: CCs 104-111 → scene idx 0-7 (idx = cc - 104).
 */
namespace LaunchpadProgrammerMap {

struct Pad {
    int row = 0; // 0 = top
    int col = 0; // 0 = left; side column uses 9
};

/** Grid pad (col 0-7) or side note (col 9) from Note number. */
[[nodiscard]] auto padFromNote(int noteNumber) -> std::optional<Pad>;

/** Inverse of padFromNote for grid (col 0-7) or side (col 9). */
[[nodiscard]] auto noteFromPad(int row, int col) -> std::optional<int>;

/** Top-row CC → scene idx 0..7, or nullopt. */
[[nodiscard]] auto sceneIndexFromTopCc(int controllerNumber) -> std::optional<int>;

} // namespace LaunchpadProgrammerMap
