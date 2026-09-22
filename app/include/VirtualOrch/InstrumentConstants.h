#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "VirtualOrch/GeneralMidiPitchRanges.h"

/**
Map between local instrument indices and General MIDI program IDs.

Used because we aren't allowing that many instruments.
 */
namespace InstrumentConstants {

inline constexpr int32_t kGmInstrumentCount = 128;

using InstrumentPair = std::pair<int32_t, int32_t>; // {local, gm}

/**
 * Local id == taxonomy_index of instrument_taxonomy_20, which the
 * InstrumentCombinations checkpoint is trained on. Families are contiguous:
 * strings 0–6, woodwind 7–10, brass 11–14, other 15–19 — see
 * InstrumentCombinations::allowedTokensForFamily, which relies on that order.
 */
inline constexpr std::array<InstrumentPair, 20> kInstrumentMappings{{
    {0, 40}, // Violin
    {1, 41}, // Viola
    {2, 42}, // Cello
    {3, 43}, // Contrabass
    {4, 44}, // Tremolo Strings
    {5, 45}, // Pizzicato Strings
    {6, 48}, // String Ensemble 1
    {7, 73}, // Flute
    {8, 68}, // Oboe
    {9, 71}, // Clarinet
    {10, 70}, // Bassoon
    {11, 56}, // Trumpet
    {12, 60}, // French Horn
    {13, 57}, // Trombone
    {14, 58}, // Tuba
    {15, 0}, // Piano
    {16, 47}, // Timpani
    {17, 116}, // Taiko / Snare Drum
    {18, 25}, // Acoustic guitar (steel)
    {19, 52}, // Choir
}};

/** Sentinel: local instrument is not routed to a MIDI output channel. */
inline constexpr int32_t kNoOutputChannel = -1;

/** Local id for String Ensemble 1 — routed by pitch to a solo string channel. */
inline constexpr int32_t kStringEnsembleLocalId = 6;

/** Solo strings, highest → lowest: violin, viola, cello, contrabass. */
inline constexpr std::array<int32_t, 4> kSoloStringLocalIds{{0, 1, 2, 3}};

/**
 * 1-based MIDI output channel for each local instrument id [0, 19].
 * kNoOutputChannel means do not send note on/off for that instrument
 * (String Ensemble is resolved via pitch instead — see outputChannelForLocalInstrumentId).
 *
 * MIDI channels 1–16 follow the bus order:
 *   violin, viola, cello + contrabass (shared 3), tremolo, pizz,
 *   flute, oboe, clarinet, bassoon, trumpet, horn, trombone, tuba,
 *   timpani, snare, piano.
 * Extra local ids (string ens, guitar, choir) do not take a channel.
 *
 * Channel 16 is the piano / keyboard monitor bus: MIDI keyboard input-thru
 * (`relayMidi`) is forced here in Jam mode only (never through OT).
 * Orchestrated piano (local 15) also maps here.
 */
inline constexpr std::array<int32_t, 20> kLocalInstrumentOutputChannels{{
    1, 2, 3, 3, 4, 5, // 0–5: violin, viola, cello, contrabass (shared 3), tremolo, pizz
    kNoOutputChannel, // 6: string ens. → pitch remap
    6, 7, 8, 9, // 7–10: flute, oboe, clarinet, bassoon
    10, 11, 12, 13, // 11–14: trumpet, horn, trombone, tuba
    16, // 15: piano
    14, // 16: timpani
    15, kNoOutputChannel, // 17–18: snare; guitar dropped
    kNoOutputChannel, // 19: choir
}};

/** MIDI channel for keyboard input-thru and reduction / piano monitor bus. */
inline constexpr int32_t kReductionMonitorMidiChannel = 16;

/** Piano local id — OT/playback keyboard monitor (MIDI channel 16). */
inline constexpr int32_t kPianoLocalId = 15;

/**
 * Instrument id embedded in MIDI-keyboard tokens for the reduction model.
 * Dense vocab only allows instruments in [0, MaxInstr) (MaxInstr == 5); piano (15)
 * overflows the embedding (Gather OOB). Always 0 here.
 */
inline constexpr int32_t kReductionInputLocalInstrumentId = 0;

/**
 * How the keyboard is labeled/routed on the OT side (piano → ch 16).
 * Applied in InputFilter when fanning to OT midiInputIncoming — not in reduction inputData.
 */
inline constexpr int32_t kKeyboardLocalInstrumentId = kPianoLocalId;

/**
 * Local id on ClearQueue tokens pushed to OutputPlayback (not a sounding note).
 */
inline constexpr int32_t kReductionPlaybackLocalId = kPianoLocalId;

[[nodiscard]] inline auto gmIdForLocalInstrumentId(int32_t localInstrumentId) -> std::optional<int32_t> {
    for (const auto &[localId, gmId]: kInstrumentMappings) {
        if (localId == localInstrumentId)
            return gmId;
    }
    return std::nullopt;
}

[[nodiscard]] inline auto localInstrumentAllowsPitch(int32_t localInstrumentId, int32_t pitch)
    -> bool {
    const auto gmId = gmIdForLocalInstrumentId(localInstrumentId);
    if (! gmId.has_value())
        return false;
    const auto range = GeneralMidiPitchRanges::pitchRangeForId(*gmId);
    return pitch >= range.low && pitch <= range.high;
}

[[nodiscard]] inline auto filterLocalInstrumentsForPitch(const std::vector<int32_t> &instruments,
                                                         int32_t pitch) -> std::vector<int32_t> {
    std::vector<int32_t> filtered;
    filtered.reserve(instruments.size());
    for (const int32_t id: instruments) {
        if (localInstrumentAllowsPitch(id, pitch))
            filtered.push_back(id);
    }
    return filtered;
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
