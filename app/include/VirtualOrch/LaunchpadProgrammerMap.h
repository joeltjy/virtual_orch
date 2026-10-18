#pragma once

#include <optional>

/** Novation Launchpad Programmer-mode MIDI map (Mini Mk3 / X-style).
 *
 *  8x8 pads: notes 11-18 … 81-88 (row 0 = top).
 *  Right side column: notes 19,29,...,89 (MK2) or CCs 19,29,...,89 (Mini Mk3 / X) → col = 9.
 *  Top row: CCs 104-111 (MK2) or 91-98 (Mini Mk3 / X) → scene idx 0-7.
 */
namespace LaunchpadProgrammerMap {

/** First top-row CC on Mini Mk3 / X (MK2 uses LaunchpadGrid::kTopCcBase = 104). */
inline constexpr int kMiniTopCcBase = 91;

struct Pad {
    int row = 0; // 0 = top
    int col = 0; // 0 = left; side column uses 9
};

/** Grid pad (col 0-7) or side note (col 9) from Note number. */
[[nodiscard]] auto padFromNote(int noteNumber) -> std::optional<Pad>;

/** Inverse of padFromNote for grid (col 0-7) or side (col 9). */
[[nodiscard]] auto noteFromPad(int row, int col) -> std::optional<int>;

/** Top-row CC (104-111 or 91-98) → scene idx 0..7, or nullopt. */
[[nodiscard]] auto sceneIndexFromTopCc(int controllerNumber) -> std::optional<int>;

/** Side-column CC 19,29,...,89 (Mini Mk3 / X Programmer mode) → row (0 = top), or nullopt. */
[[nodiscard]] auto sideRowFromCc(int controllerNumber) -> std::optional<int>;

} // namespace LaunchpadProgrammerMap
