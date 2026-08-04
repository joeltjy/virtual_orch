#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <utility>

/** 
Map between local instrument indices and General MIDI program IDs.

Used because we aren't allowing that many instruments.
 */
namespace InstrumentConstants {

inline constexpr int32_t kGmInstrumentCount = 128;

using InstrumentPair = std::pair<int32_t, int32_t>; // {local, gm}

inline constexpr std::array<InstrumentPair, 19> kInstrumentMappings{{
    {0, 40}, // Violin
    {1, 41}, // Viola
    {2, 42}, // Cello
    {3, 43}, // Contrabass
    {4, 44}, // Tremolo Strings
    {5, 45}, // Pizzicato Strings
    {6, 46}, // Harp
    {7, 48}, // String Ensemble 1
    {8, 73}, // Flute
    {9, 71}, // Clarinet
    {10, 70}, // Bassoon
    {11, 56}, // Trumpet
    {12, 57}, // Trombone
    {13, 58}, // Tuba
    {14, 0}, // Piano   
    {15, 47}, // Timpani
    {16, 116}, // Snare Drum
    {17, 25}, // Acoustic guitar
    {18, 52}, // Choir
}}; // TODO: add horn! How did i miss it

[[nodiscard]] auto toGmInstrumentId(int32_t localInstrumentId) -> std::optional<int32_t>;

[[nodiscard]] auto toLocalInstrumentId(int32_t gmInstrumentId) -> std::optional<int32_t>;

[[nodiscard]] auto isValidLocalInstrumentId(int32_t localInstrumentId) -> bool;

} // namespace InstrumentConstants
