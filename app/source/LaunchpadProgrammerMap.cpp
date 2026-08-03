#include "VirtualOrch/LaunchpadProgrammerMap.h"

#include "VirtualOrch/LaunchpadGrid.h"

namespace LaunchpadProgrammerMap {

namespace {

auto rowFromTensDigit(int tens) -> std::optional<int> {
    // tens 1..8 → hardware bottom..top; invert so row 0 is top.
    if (tens < 1 || tens > 8)
        return std::nullopt;
    return 8 - tens;
}

} // namespace

auto padFromNote(int noteNumber) -> std::optional<Pad> {
    if (noteNumber < 11 || noteNumber > 89)
        return std::nullopt;

    const int tens = noteNumber / 10;
    const int ones = noteNumber % 10;
    const auto row = rowFromTensDigit(tens);
    if (!row.has_value())
        return std::nullopt;

    if (ones >= 1 && ones <= 8)
        return Pad{.row = *row, .col = ones - 1};

    if (ones == 9)
        return Pad{.row = *row, .col = LaunchpadGrid::kSideCol};

    return std::nullopt;
}

auto sceneIndexFromTopCc(int controllerNumber) -> std::optional<int> {
    const int idx = controllerNumber - LaunchpadGrid::kTopCcBase;
    if (idx < 0 || idx >= LaunchpadGrid::kTopCcCount)
        return std::nullopt;
    return idx;
}

} // namespace LaunchpadProgrammerMap
