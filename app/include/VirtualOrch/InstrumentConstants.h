#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <utility>

#include "VirtualOrch/GeneralMidiPitchRanges.h"

/**
Map between local instrument indices and General MIDI program IDs.

Used because we aren't allowing that many instruments.
 */
namespace InstrumentConstants {

inline constexpr int32_t kGmInstrumentCount = 128;

using InstrumentPair = std::pair<int32_t, int32_t>; // {local, gm}

inline constexpr std::array<InstrumentPair, 20> kInstrumentMappings{{
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
    {12, 60}, // French Horn
    {13, 57}, // Trombone
    {14, 58}, // Tuba
    {15, 0}, // Piano
    {16, 47}, // Timpani
    {17, 116}, // Snare Drum
    {18, 25}, // Acoustic guitar
    {19, 52}, // Choir
}};

/** Sentinel: local instrument is not routed to a MIDI output channel. */
inline constexpr int32_t kNoOutputChannel = -1;

/** Local id for String Ensemble 1 — routed by pitch to a solo string channel. */
inline constexpr int32_t kStringEnsembleLocalId = 7;

/** Solo strings, highest → lowest: violin, viola, cello, contrabass. */
inline constexpr std::array<int32_t, 4> kSoloStringLocalIds{{0, 1, 2, 3}};

/**
 * 1-based MIDI output channel for each local instrument id [0, 19].
 * kNoOutputChannel means do not send note on/off for that instrument
 * (String Ensemble is resolved via pitch instead — see outputChannelForLocalInstrumentId).
 */
inline constexpr std::array<int32_t, 20> kLocalInstrumentOutputChannels{{
    1,  2,  3,  4,  5,  6,  // 0–5
    kNoOutputChannel, kNoOutputChannel, // 6–7 (harp; string ens. → pitch remap)
    7,  8,  9, // 8–10 woodwinds
    10, 11, 12, 13, 14, // 11–15: trumpet, horn, trombone, tuba, piano
    kNoOutputChannel, // 16 (timpani)
    15, 16, // 17–18 snare, guitar
    kNoOutputChannel, // 19 (choir)
}};

[[nodiscard]] inline auto gmIdForLocalInstrumentId(int32_t localInstrumentId) -> std::optional<int32_t> {
    for (const auto &[localId, gmId]: kInstrumentMappings) {
        if (localId == localInstrumentId)
            return gmId;
    }
    return std::nullopt;
}

/**
 * MIDI channel for a local instrument. For String Ensemble 1, picks the highest
 * solo string (violin → viola → cello → bass) whose GM pitch range contains `pitch`.
 */
[[nodiscard]] inline auto outputChannelForLocalInstrumentId(int32_t localInstrumentId,
                                                            int32_t pitch)
    -> std::optional<int32_t> {
    if (localInstrumentId == kStringEnsembleLocalId) {
        for (const auto stringLocalId: kSoloStringLocalIds) {
            const auto gmId = gmIdForLocalInstrumentId(stringLocalId);
            if (! gmId.has_value())
                continue;
            const auto range = GeneralMidiPitchRanges::pitchRangeForId(*gmId);
            if (pitch >= range.low && pitch <= range.high)
                return kLocalInstrumentOutputChannels[static_cast<size_t>(stringLocalId)];
        }
        return std::nullopt;
    }

    if (localInstrumentId < 0
        || localInstrumentId >= static_cast<int32_t>(kLocalInstrumentOutputChannels.size()))
        return std::nullopt;
    const auto channel = kLocalInstrumentOutputChannels[static_cast<size_t>(localInstrumentId)];
    if (channel == kNoOutputChannel)
        return std::nullopt;
    return channel;
}

[[nodiscard]] auto toGmInstrumentId(int32_t localInstrumentId) -> std::optional<int32_t>;

[[nodiscard]] auto toLocalInstrumentId(int32_t gmInstrumentId) -> std::optional<int32_t>;

[[nodiscard]] auto isValidLocalInstrumentId(int32_t localInstrumentId) -> bool;

} // namespace InstrumentConstants
