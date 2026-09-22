#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "VirtualOrch/GeneralMidiPitchRanges.h"
#include "VirtualOrch/InstrumentConstants.h"
#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/orchestration-models/OrchestrationModel.h"

/** Compact IODP combo vocab (instrument × Δ∈{-1,0,+1} bits; not ICE-20). */
namespace IodPretrainedTypes {

inline constexpr int32_t numFamilies = 4;
inline constexpr int32_t deltasPerInstrument = 3;
inline constexpr int32_t comboLen = 1 + numFamilies; // GROUPS + 4 family slots
inline constexpr int32_t noteTokensPerNote = 3;      // onset, duration, pitch
inline constexpr int32_t nNotesMax = 128;
inline constexpr int32_t noteSeqLen = nNotesMax * noteTokensPerNote; // 384

inline constexpr int32_t bos = 0;
inline constexpr int32_t groupsSize = (1 << numFamilies) - 1; // 15
inline constexpr int32_t groupsOffset = bos + 1;              // 1
inline constexpr int32_t groupsHi = groupsOffset + groupsSize; // 16 exclusive end of GROUPS ids

inline constexpr int32_t stringsBits = 9;   // 3 instr × 3 deltas
inline constexpr int32_t woodwindBits = 12; // 4 × 3
inline constexpr int32_t brassBits = 12;
inline constexpr int32_t otherBits = 6; // 2 × 3

inline constexpr int32_t stringsSize = 1 << stringsBits;     // 512
inline constexpr int32_t woodwindSize = 1 << woodwindBits;   // 4096
inline constexpr int32_t brassSize = 1 << brassBits;         // 4096
inline constexpr int32_t otherSize = 1 << otherBits;         // 64

inline constexpr int32_t stringsOffset = groupsOffset + groupsSize; // 16
inline constexpr int32_t woodwindOffset = stringsOffset + stringsSize;
inline constexpr int32_t brassOffset = woodwindOffset + woodwindSize;
inline constexpr int32_t otherOffset = brassOffset + brassSize;
inline constexpr int32_t endNote = otherOffset + otherSize; // 8784
inline constexpr int32_t comboVocabSize = endNote + 1;      // 8785

static_assert(comboVocabSize == 1 + 15 + 512 + 4096 + 4096 + 64 + 1);
static_assert(noteSeqLen == 384);

inline constexpr std::array<int32_t, 3> stringsPrograms{{44, 45, 48}};
inline constexpr std::array<int32_t, 4> woodwindPrograms{{73, 68, 71, 70}};
inline constexpr std::array<int32_t, 4> brassPrograms{{56, 60, 57, 58}};
inline constexpr std::array<int32_t, 2> otherPrograms{{47, 52}};

inline constexpr std::array<int32_t, 7> disabledTax20Locals{{0, 1, 2, 3, 15, 17, 18}};

struct FamilySpec {
    const int32_t *programs = nullptr;
    int32_t numInstruments = 0;
    int32_t tokenOffset = 0;
    int32_t tokenSize = 0;
    int32_t numBits = 0;
};

inline constexpr std::array<FamilySpec, numFamilies> familySpecs{{
    {stringsPrograms.data(), 3, stringsOffset, stringsSize, stringsBits},
    {woodwindPrograms.data(), 4, woodwindOffset, woodwindSize, woodwindBits},
    {brassPrograms.data(), 4, brassOffset, brassSize, brassBits},
    {otherPrograms.data(), 2, otherOffset, otherSize, otherBits},
}};

[[nodiscard]] inline constexpr auto iodBit(int32_t instrInFamily, int32_t delta) -> int32_t {
    return deltasPerInstrument * instrInFamily + (delta + 1);
}

[[nodiscard]] inline auto programDeltaFromBit(const FamilySpec &family, int32_t bit)
    -> std::optional<std::pair<int32_t, int32_t>> {
    if (bit < 0 || bit >= family.numBits)
        return std::nullopt;
    const int32_t instrIdx = bit / deltasPerInstrument;
    const int32_t deltaCode = bit % deltasPerInstrument;
    if (instrIdx < 0 || instrIdx >= family.numInstruments)
        return std::nullopt;
    return std::pair{family.programs[instrIdx], deltaCode - 1};
}

[[nodiscard]] inline auto isDisabledTax20Local(int32_t localId) -> bool {
    for (const int32_t id: disabledTax20Locals) {
        if (id == localId)
            return true;
    }
    return false;
}

[[nodiscard]] inline auto localIdForIodGm(int32_t gmProgram) -> std::optional<int32_t> {
    for (const auto &[localId, gmId]: InstrumentConstants::kInstrumentMappings) {
        if (gmId == gmProgram)
            return localId;
    }
    return std::nullopt;
}

[[nodiscard]] inline auto familyInstrIndexForGm(int32_t gmProgram)
    -> std::optional<std::pair<int32_t, int32_t>> {
    for (int32_t f = 0; f < numFamilies; ++f) {
        const auto &spec = familySpecs[static_cast<size_t>(f)];
        for (int32_t i = 0; i < spec.numInstruments; ++i) {
            if (spec.programs[i] == gmProgram)
                return std::pair{f, i};
        }
    }
    return std::nullopt;
}

[[nodiscard]] inline auto isIodGmProgram(int32_t gmProgram) -> bool {
    return familyInstrIndexForGm(gmProgram).has_value();
}

[[nodiscard]] inline constexpr auto groupsTokenForMask(int32_t familyMask) -> int32_t {
    return familyMask;
}

[[nodiscard]] inline constexpr auto maskFromGroupsToken(int32_t token) -> int32_t {
    return (token >= groupsOffset && token < groupsHi) ? token : 0;
}

[[nodiscard]] inline constexpr auto familyTokenForMask(const FamilySpec &family, int32_t bitMask)
    -> int32_t {
    return family.tokenOffset + bitMask;
}

[[nodiscard]] inline constexpr auto maskFromFamilyToken(const FamilySpec &family, int32_t token)
    -> int32_t {
    const int32_t rel = token - family.tokenOffset;
    if (rel <= 0 || rel >= family.tokenSize)
        return 0;
    return rel;
}

[[nodiscard]] inline auto gmPitchInRange(int32_t gmProgram, int32_t pitch) -> bool {
    if (gmProgram < 0 || gmProgram >= GeneralMidiPitchRanges::kNumInstruments)
        return false;
    if (pitch < 0 || pitch > 127)
        return false;
    const auto range = GeneralMidiPitchRanges::pitchRangeForId(gmProgram);
    return pitch >= range.low && pitch <= range.high;
}

[[nodiscard]] inline auto localIdIsActive(const std::vector<int32_t> &instruments,
                                          int32_t localId) -> bool {
    for (const int32_t id: instruments) {
        if (id == localId)
            return true;
    }
    return false;
}

/** Per-family illegal (instr×Δ) bits + fully-banned family mask (Python _forbid_bits). */
struct ForbidBits {
    std::array<int32_t, numFamilies> familyForbid{};
    int32_t fullyBannedFamilyBits = 0;
};

/** Pad locals → allowed IOD GMs; forbid (program,Δ) outside pitch/GM range. */
[[nodiscard]] inline auto forbidBitsForPitch(const std::vector<int32_t> &instruments,
                                             int32_t pitch) -> ForbidBits {
    ForbidBits out;
    for (int32_t f = 0; f < numFamilies; ++f) {
        const auto &family = familySpecs[static_cast<size_t>(f)];
        int32_t mask = 0;
        int32_t nOk = 0;
        for (int32_t instrIdx = 0; instrIdx < family.numInstruments; ++instrIdx) {
            const int32_t program = family.programs[instrIdx];
            const auto local = localIdForIodGm(program);
            const bool programBanned =
                ! local.has_value() || ! localIdIsActive(instruments, *local);

            bool legalAny = false;
            for (const int32_t delta: {-1, 0, 1}) {
                const int32_t outPitch = pitch + 12 * delta;
                const bool illegal = programBanned || ! gmPitchInRange(program, outPitch);
                if (illegal)
                    mask |= (1 << iodBit(instrIdx, delta));
                else
                    legalAny = true;
            }
            if (legalAny)
                ++nOk;
        }
        out.familyForbid[static_cast<size_t>(f)] = mask;
        if (nOk == 0)
            out.fullyBannedFamilyBits |= (1 << f);
    }
    return out;
}

[[nodiscard]] inline auto isLegalGroupsMask(int32_t groupsMask, int32_t fullyBannedFamilyBits)
    -> bool {
    return groupsMask >= 1 && groupsMask <= groupsSize
           && (groupsMask & fullyBannedFamilyBits) == 0;
}

[[nodiscard]] inline auto isLegalFamilyMask(int32_t bitMask, int32_t familyForbidBits) -> bool {
    return bitMask > 0 && (bitMask & familyForbidBits) == 0;
}

inline constexpr int32_t encoderDurOffset = 10000;
inline constexpr int32_t encoderNoteOffset = 11000;

/** Fixed-size ONNX encoder window (pad to nNotesMax); comboIn starts as BOS. */
struct EncoderWindow {
    std::array<int64_t, noteSeqLen> inputIds{};
    std::array<int64_t, noteSeqLen> attentionMask{};
    std::array<int64_t, nNotesMax> noteValid{};
    std::array<std::array<int64_t, comboLen>, nNotesMax> comboIn{};
    int32_t nNotes = 0;
    std::array<int32_t, nNotesMax> onsetCs{};
    std::array<int32_t, nNotesMax> durationToken{};
    std::array<int32_t, nNotesMax> basePitch{};
    std::array<int32_t, nNotesMax> velocity{};
};

[[nodiscard]] inline auto midiPitchFromNoteToken(int32_t noteToken) -> int32_t {
    const int32_t rel = noteToken - encoderNoteOffset;
    if (rel < 0)
        return 0;
    return rel % 128;
}

/** Pack up to nNotesMax tokens from `start` (onset/dur/pitch; velocity dropped). */
[[nodiscard]] inline auto packEncoderWindow(const std::vector<Token> &tokens, size_t start = 0)
    -> EncoderWindow {
    EncoderWindow w;
    for (auto &row: w.comboIn)
        row.fill(bos);

    if (start >= tokens.size())
        return w;

    const size_t nKeep = std::min(tokens.size() - start, static_cast<size_t>(nNotesMax));
    w.nNotes = static_cast<int32_t>(nKeep);

    for (size_t i = 0; i < nKeep; ++i) {
        const Token &t = tokens[start + i];
        const int32_t pitch = t.getPitch();
        const int32_t durTok =
            t.duration >= encoderDurOffset ? t.duration
                                           : encoderDurOffset + std::max(0, t.duration);
        const int32_t noteTok = encoderNoteOffset + pitch;
        const int32_t vel = t.velocity > 0 ? t.velocity : 100;

        const size_t base = i * static_cast<size_t>(noteTokensPerNote);
        w.inputIds[base] = t.time;
        w.inputIds[base + 1] = durTok;
        w.inputIds[base + 2] = noteTok;
        w.attentionMask[base] = 1;
        w.attentionMask[base + 1] = 1;
        w.attentionMask[base + 2] = 1;
        w.noteValid[i] = 1;

        w.onsetCs[i] = t.time;
        w.durationToken[i] = durTok;
        w.basePitch[i] = pitch;
        w.velocity[i] = vel;
    }
    return w;
}

/** Chunk starts for streams longer than nNotesMax (Python range(0, n, 128)). */
[[nodiscard]] inline auto encoderWindowStarts(size_t numTokens) -> std::vector<size_t> {
    std::vector<size_t> starts;
    for (size_t s = 0; s < numTokens; s += static_cast<size_t>(nNotesMax))
        starts.push_back(s);
    return starts;
}

/** Decode packed combo (GROUPS + active family tokens + END pad); pitch += 12Δ. */
[[nodiscard]] inline auto decodeComboToNotes(int32_t onset,
                                             int32_t durationToken,
                                             int32_t basePitch,
                                             int32_t velocity,
                                             const std::array<int32_t, comboLen> &comboLocal)
    -> std::vector<OrchestrationNote> {
    const int32_t groupsMask = maskFromGroupsToken(comboLocal[0]);
    if (groupsMask == 0)
        return {};

    std::vector<int32_t> expectedFamilies;
    expectedFamilies.reserve(static_cast<size_t>(numFamilies));
    for (int32_t f = 0; f < numFamilies; ++f) {
        if ((groupsMask & (1 << f)) != 0)
            expectedFamilies.push_back(f);
    }

    std::vector<int32_t> familyTokens;
    familyTokens.reserve(expectedFamilies.size());
    for (size_t i = 1; i < comboLocal.size(); ++i) {
        if (comboLocal[i] == endNote)
            break;
        familyTokens.push_back(comboLocal[i]);
    }
    if (familyTokens.size() != expectedFamilies.size())
        return {};

    std::vector<OrchestrationNote> notes;
    for (size_t k = 0; k < expectedFamilies.size(); ++k) {
        const int32_t familyIdx = expectedFamilies[k];
        const auto &family = familySpecs[static_cast<size_t>(familyIdx)];
        const int32_t bitMask = maskFromFamilyToken(family, familyTokens[k]);
        if (bitMask == 0)
            continue;

        for (int32_t bit = 0; bit < family.numBits; ++bit) {
            if ((bitMask & (1 << bit)) == 0)
                continue;
            const auto pd = programDeltaFromBit(family, bit);
            if (! pd.has_value())
                continue;
            const int32_t gm = pd->first;
            const int32_t delta = pd->second;
            const int32_t outPitch = basePitch + 12 * delta;
            if (outPitch < 0 || outPitch > 127)
                continue;
            const auto local = localIdForIodGm(gm);
            if (! local.has_value())
                continue;

            OrchestrationNote assigned;
            assigned.token.time = onset;
            assigned.token.duration = durationToken;
            assigned.token.note =
                static_cast<int32_t>(Vocab::NoteOffset) + outPitch;
            assigned.velocity = velocity;
            assigned.token.velocity = velocity;
            assigned.localInstrumentId = *local;
            notes.push_back(assigned);
        }
    }
    return notes;
}

} // namespace IodPretrainedTypes
