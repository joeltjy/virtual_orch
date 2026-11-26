#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "VirtualOrch/LaunchpadLighting.h"
#include "VirtualOrch/RegisterMix.h"
#include "VirtualOrch/RegisterRange.h"

/**
 * Mood presets: the Launchpad's bottom row (LaunchpadGrid::kPresetRow), calm → intense
 * left to right, coloured blue → red. A press sets the instruments of the side the pads
 * control in the current mode (yours in Edit, the model's in Jam / Accompaniment), plus
 * temporal density and register.
 *
 * Edit this table to change the presets. Local instrument ids (AppSession kInstrumentPads):
 *   4 tremolo, 5 pizzicato, 6 string ensemble,
 *   7 flute, 8 oboe, 9 clarinet, 10 bassoon,
 *   11 trumpet, 12 horn, 13 trombone, 14 tuba, 16 timpani
 * tdLevelIndex indexes TemporalDensity::kLevels {1, 2, 4, 8, 16, 32, 64}; -1 = AUTO.
 * registerMix: the shares of the reduction's notes below C3 / C3 to C5 / above C5 the RegisterMix controller steers
 * toward (empty = none). Which of the two a preset press sets is an option (RegisterMix::Target::presetsUseMix):
 * off (the default) registerValue as before; on, AUTO (C4 +/- 2.5 octaves) and the mix (a hard LOW / MEDIUM / HIGH
 * range would leave some bands empty). A share of 0 masks that band.
 */
namespace MoodPresets {

struct Preset {
    const char *name;
    uint8_t colour; // LaunchpadLighting palette, full brightness
    std::vector<int32_t> instruments;
    int tdLevelIndex;  // -1 = AUTO
    int32_t registerValue; // RegisterRange::kAuto / kLow / kMedium / kHigh
    std::optional<RegisterMix::Mix> registerMix; // below C3, C3-C5, above C5
};

inline constexpr int kCount = 8;

[[nodiscard]] inline auto all() -> const std::array<Preset, kCount> & {
    using namespace LaunchpadLighting;
    static const std::array<Preset, kCount> presets{{
        //                                                         register mix: below C3, C3-C5, above C5
        {"Solemn", kBlue, {6, 10, 12, 13, 14}, 1, RegisterRange::kLow, RegisterMix::Mix{0.45f, 0.50f, 0.05f}},
        {"Mysterious", kPurple, {4, 9, 10}, 4, RegisterRange::kLow, RegisterMix::Mix{0.35f, 0.45f, 0.20f}},
        {"Tender", kSky, {6, 9, 12}, 2, RegisterRange::kAuto, RegisterMix::Mix{0.10f, 0.70f, 0.20f}},
        {"Playful", kLime, {5, 7, 8, 10}, 4, RegisterRange::kHigh, RegisterMix::Mix{0.05f, 0.45f, 0.50f}},
        {"Passionate", kPink, {6, 8, 9, 12, 13}, 4, RegisterRange::kAuto, RegisterMix::Mix{0.20f, 0.55f, 0.25f}},
        {"Heroic", kOrange, {6, 11, 12, 13, 16}, 4, RegisterRange::kMedium, RegisterMix::Mix{0.25f, 0.55f, 0.20f}},
        {"Stormy", kRed, {4, 11, 13, 14, 16}, 5, RegisterRange::kLow, RegisterMix::Mix{0.45f, 0.35f, 0.20f}},
        {"Grand", kWhite, {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16}, 6, RegisterRange::kAuto,
         RegisterMix::Mix{0.30f, 0.40f, 0.30f}},
    }};
    return presets;
}

[[nodiscard]] inline auto contains(const Preset &preset, int32_t localId) -> bool {
    for (const auto id: preset.instruments)
        if (id == localId)
            return true;
    return false;
}

} // namespace MoodPresets
