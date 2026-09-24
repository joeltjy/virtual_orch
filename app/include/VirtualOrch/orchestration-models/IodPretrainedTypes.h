#pragma once

#include <algorithm>
#include <array>
#include <cmath>
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

/** Past-N GROUPS unigram bias on the first combo step (−τ log π_y). */
inline constexpr int32_t groupsBiasWindowNotes = 30;
inline constexpr float groupsBiasTau = 1.0f;
/** Additive smoothing: π_y = (count_y + ε) / (N + groupsSize · ε). */
inline constexpr float groupsBiasEpsilon = 1.0f / static_cast<float>(groupsSize);

/** Clear encoder note/combo stream if onset gap exceeds this (pedal-down / silence). */
inline constexpr int32_t historyGapClearCs = 500; // 5 s

/** Encoder stream duration id = encoderDurOffset + duration_cs. */
inline constexpr int32_t encoderDurOffset = 10000;
inline constexpr int32_t encoderNoteOffset = 11000;

/** Same-voice ensemble: past-N raw (A) predictions for add/remove proportions. */
inline constexpr int32_t ensembleHistoryNotes = 10;
inline constexpr float ensembleAddPDefault = 0.67f;
inline constexpr float ensembleRemovePDefault = 0.3f;

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

/**
 * Outer-voice octave roles among temporally concurrent notes:
 * top = nothing pitched higher while this note sounds → may use Δ=+1;
 * bottom = nothing pitched lower while this note sounds → may use Δ=−1.
 * A lone note allows both.
 */
struct OuterVoiceDeltaAllows {
    bool allowMinus1 = true;
    bool allowPlus1 = true;
};

/** Duration in centiseconds from an encoder / vocab duration token (min 1). */
[[nodiscard]] inline auto durationCsFromToken(int32_t durationToken) -> int32_t {
    int32_t cs = durationToken;
    if (durationToken >= encoderDurOffset)
        cs = durationToken - encoderDurOffset;
    else if (durationToken >= static_cast<int32_t>(Vocab::DurOffset))
        cs = durationToken - static_cast<int32_t>(Vocab::DurOffset);
    return std::max(1, cs);
}

[[nodiscard]] inline auto notesOverlapCs(int32_t onsetA,
                                         int32_t durA,
                                         int32_t onsetB,
                                         int32_t durB) -> bool {
    return onsetA < onsetB + durB && onsetB < onsetA + durA;
}

/**
 * @param durationTokens packed encoder/vocab duration ids (same indexing as onset/pitch),
 *        or nullptr to treat every note as duration 1 cs (onset-only concurrency).
 */
[[nodiscard]] inline auto outerVoiceDeltaAllows(const int32_t *onsetCs,
                                                const int32_t *durationTokens,
                                                const int32_t *basePitch,
                                                int32_t nNotes,
                                                int32_t noteIdx) -> OuterVoiceDeltaAllows {
    OuterVoiceDeltaAllows allows;
    if (onsetCs == nullptr || basePitch == nullptr || noteIdx < 0 || noteIdx >= nNotes)
        return allows;

    const int32_t onset = onsetCs[static_cast<size_t>(noteIdx)];
    const int32_t dur =
        durationTokens != nullptr
            ? durationCsFromToken(durationTokens[static_cast<size_t>(noteIdx)])
            : 1;
    const int32_t pitch = basePitch[static_cast<size_t>(noteIdx)];

    for (int32_t i = 0; i < nNotes; ++i) {
        if (i == noteIdx)
            continue;
        const int32_t otherOnset = onsetCs[static_cast<size_t>(i)];
        const int32_t otherDur =
            durationTokens != nullptr
                ? durationCsFromToken(durationTokens[static_cast<size_t>(i)])
                : 1;
        if (! notesOverlapCs(onset, dur, otherOnset, otherDur))
            continue;
        const int32_t otherPitch = basePitch[static_cast<size_t>(i)];
        if (otherPitch > pitch)
            allows.allowPlus1 = false;
        if (otherPitch < pitch)
            allows.allowMinus1 = false;
    }
    return allows;
}

/** Ban Δ=±1 bits when this note is not the bottom / top among concurrent notes. */
inline auto applyOuterVoiceDeltaForbid(ForbidBits &forbid,
                                       bool allowMinus1,
                                       bool allowPlus1) -> void {
    if (allowMinus1 && allowPlus1)
        return;
    for (int32_t f = 0; f < numFamilies; ++f) {
        const auto &family = familySpecs[static_cast<size_t>(f)];
        int32_t &mask = forbid.familyForbid[static_cast<size_t>(f)];
        for (int32_t instrIdx = 0; instrIdx < family.numInstruments; ++instrIdx) {
            if (! allowMinus1)
                mask |= (1 << iodBit(instrIdx, -1));
            if (! allowPlus1)
                mask |= (1 << iodBit(instrIdx, +1));
        }
        // Recompute fully-banned: any remaining legal (instr×Δ) bit?
        bool anyLegal = false;
        for (int32_t bit = 0; bit < family.numBits; ++bit) {
            if ((mask & (1 << bit)) == 0) {
                anyLegal = true;
                break;
            }
        }
        if (! anyLegal)
            forbid.fullyBannedFamilyBits |= (1 << f);
        else
            forbid.fullyBannedFamilyBits &= ~(1 << f);
    }
}

[[nodiscard]] inline auto isLegalGroupsMask(int32_t groupsMask, int32_t fullyBannedFamilyBits)
    -> bool {
    return groupsMask >= 1 && groupsMask <= groupsSize
           && (groupsMask & fullyBannedFamilyBits) == 0;
}

/**
 * Δ occupancy among instruments in a family bit mask.
 * Bit 0/1/2 ↔ Δ −1/0/+1. Same instrument at multiple Δs is fine; only distinct
 * instruments contribute.
 */
[[nodiscard]] inline auto deltasUsedByFamilyMask(const FamilySpec &family, int32_t bitMask)
    -> int32_t {
    int32_t used = 0;
    for (int32_t instrIdx = 0; instrIdx < family.numInstruments; ++instrIdx) {
        for (const int32_t delta: {-1, 0, 1}) {
            if ((bitMask & (1 << iodBit(instrIdx, delta))) != 0)
                used |= (1 << (delta + 1));
        }
    }
    return used;
}

/** True if ≥2 distinct instruments in this family mask share the same Δ. */
[[nodiscard]] inline auto familyMaskHasSharedOctave(const FamilySpec &family, int32_t bitMask)
    -> bool {
    for (const int32_t delta: {-1, 0, 1}) {
        int32_t nInstr = 0;
        for (int32_t instrIdx = 0; instrIdx < family.numInstruments; ++instrIdx) {
            if ((bitMask & (1 << iodBit(instrIdx, delta))) != 0)
                ++nInstr;
        }
        if (nInstr >= 2)
            return true;
    }
    return false;
}

[[nodiscard]] inline auto isLegalFamilyMask(int32_t bitMask,
                                            int32_t familyForbidBits,
                                            const FamilySpec *family = nullptr,
                                            int32_t claimedDeltas = 0) -> bool {
    if (bitMask <= 0 || (bitMask & familyForbidBits) != 0)
        return false;
    if (family == nullptr)
        return true;
    if (familyMaskHasSharedOctave(*family, bitMask))
        return false;
    const int32_t used = deltasUsedByFamilyMask(*family, bitMask);
    return (used & claimedDeltas) == 0;
}

/**
 * −τ log π_y on GROUPS vocab ids [groupsOffset, groupsHi).
 * π_y = (count_y + ε) / (N + groupsSize · ε) over the recent groups masks (1…15).
 * Skips non-finite logits. No-op when recentMasks empty.
 */
inline auto applyGroupsTokenLogitsBias(float *vocabLogits,
                                       size_t vocabSize,
                                       const int32_t *recentMasks,
                                       size_t recentCount,
                                       float tau = groupsBiasTau,
                                       float epsilon = groupsBiasEpsilon) -> void {
    if (vocabLogits == nullptr || recentCount == 0 || vocabSize < static_cast<size_t>(groupsHi))
        return;
    if (! (tau > 0.0f) || ! (epsilon > 0.0f))
        return;

    std::array<int32_t, static_cast<size_t>(groupsSize)> counts{};
    for (size_t i = 0; i < recentCount; ++i) {
        const int32_t mask = recentMasks[i];
        if (mask >= 1 && mask <= groupsSize)
            ++counts[static_cast<size_t>(mask - 1)];
    }

    const float n = static_cast<float>(recentCount);
    const float denom = n + static_cast<float>(groupsSize) * epsilon;
    for (int32_t mask = 1; mask <= groupsSize; ++mask) {
        const int32_t token = groupsTokenForMask(mask);
        float &logit = vocabLogits[static_cast<size_t>(token)];
        if (! std::isfinite(logit))
            continue;
        const float pi =
            (static_cast<float>(counts[static_cast<size_t>(mask - 1)]) + epsilon) / denom;
        logit -= tau * std::log(pi);
    }
}

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

/** Decode packed combo (GROUPS + active family tokens + END pad); pitch += 12Δ.
 * At most one instrument per octave Δ; same instrument at multiple Δs is kept.
 */
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
    int32_t claimedDeltas = 0; // bit0/1/2 ↔ Δ −1/0/+1
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
            const int32_t deltaBit = 1 << (delta + 1);
            // Different instruments may not share an octave; same instr × multi-Δ is ok.
            if ((claimedDeltas & deltaBit) != 0)
                continue;
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
            assigned.localInstrumentId = *local;
            const int32_t scaledVel =
                InstrumentConstants::scaleVelocityForInstrument(*local, velocity);
            assigned.velocity = scaledVel;
            assigned.token.velocity = scaledVel;
            notes.push_back(assigned);
            claimedDeltas |= deltaBit;
        }
    }
    return notes;
}

/** Packed (gmProgram, Δ) key: high 16 = GM, low 8 = delta+1 (1..3). */
[[nodiscard]] inline constexpr auto packProgramDelta(int32_t gmProgram, int32_t delta) -> int32_t {
    return (gmProgram << 8) | (delta + 1);
}

[[nodiscard]] inline constexpr auto unpackProgram(int32_t key) -> int32_t { return key >> 8; }

[[nodiscard]] inline constexpr auto unpackDelta(int32_t key) -> int32_t {
    return (key & 0xff) - 1;
}

/**
 * Keep at most one instrument per Δ across (GM, Δ) pairs.
 * Same GM at multiple Δs is allowed; a later different GM that reuses a Δ is dropped.
 */
[[nodiscard]] inline auto filterPairsUniqueOctave(const std::vector<int32_t> &pairs)
    -> std::vector<int32_t> {
    std::vector<int32_t> out;
    out.reserve(pairs.size());
    int32_t claimedDeltas = 0; // bit0/1/2 ↔ Δ −1/0/+1
    for (const int32_t key: pairs) {
        const int32_t delta = unpackDelta(key);
        if (delta < -1 || delta > 1)
            continue;
        const int32_t bit = 1 << (delta + 1);
        if ((claimedDeltas & bit) != 0)
            continue;
        claimedDeltas |= bit;
        out.push_back(key);
    }
    return out;
}

/** Drop (GM,Δ) pairs whose Δ is disallowed for this note's outer-voice role. */
[[nodiscard]] inline auto filterPairsByDeltaAllows(const std::vector<int32_t> &pairs,
                                                   bool allowMinus1,
                                                   bool allowPlus1) -> std::vector<int32_t> {
    if (allowMinus1 && allowPlus1)
        return pairs;
    std::vector<int32_t> out;
    out.reserve(pairs.size());
    for (const int32_t key: pairs) {
        const int32_t delta = unpackDelta(key);
        if (delta < 0 && ! allowMinus1)
            continue;
        if (delta > 0 && ! allowPlus1)
            continue;
        out.push_back(key);
    }
    return out;
}

/** Expand combo tokens to unique (GM program, Δ) pairs. */
[[nodiscard]] inline auto pairsFromCombo(const std::array<int32_t, comboLen> &comboLocal)
    -> std::vector<int32_t> {
    std::vector<int32_t> pairs;
    const int32_t groupsMask = maskFromGroupsToken(comboLocal[0]);
    if (groupsMask == 0)
        return pairs;

    std::vector<int32_t> expectedFamilies;
    for (int32_t f = 0; f < numFamilies; ++f) {
        if ((groupsMask & (1 << f)) != 0)
            expectedFamilies.push_back(f);
    }

    std::vector<int32_t> familyTokens;
    for (size_t i = 1; i < comboLocal.size(); ++i) {
        if (comboLocal[i] == endNote)
            break;
        familyTokens.push_back(comboLocal[i]);
    }
    if (familyTokens.size() != expectedFamilies.size())
        return pairs;

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
            pairs.push_back(packProgramDelta(pd->first, pd->second));
        }
    }
    return pairs;
}

/** Rebuild GROUPS + family combo from (GM, Δ) pairs; empty → all endNote (no sounding). */
[[nodiscard]] inline auto comboFromPairs(const std::vector<int32_t> &pairs)
    -> std::array<int32_t, comboLen> {
    std::array<int32_t, comboLen> combo{};
    combo.fill(endNote);

    std::array<int32_t, numFamilies> familyBits{};
    for (const int32_t key: pairs) {
        const int32_t gm = unpackProgram(key);
        const int32_t delta = unpackDelta(key);
        if (delta < -1 || delta > 1)
            continue;
        const auto loc = familyInstrIndexForGm(gm);
        if (! loc.has_value())
            continue;
        const int32_t familyIdx = loc->first;
        const int32_t instrIdx = loc->second;
        familyBits[static_cast<size_t>(familyIdx)] |= (1 << iodBit(instrIdx, delta));
    }

    int32_t groupsMask = 0;
    for (int32_t f = 0; f < numFamilies; ++f) {
        if (familyBits[static_cast<size_t>(f)] != 0)
            groupsMask |= (1 << f);
    }
    if (groupsMask == 0)
        return combo;

    combo[0] = groupsTokenForMask(groupsMask);
    int32_t write = 1;
    for (int32_t f = 0; f < numFamilies; ++f) {
        if ((groupsMask & (1 << f)) == 0)
            continue;
        const auto &family = familySpecs[static_cast<size_t>(f)];
        combo[static_cast<size_t>(write++)] =
            familyTokenForMask(family, familyBits[static_cast<size_t>(f)]);
    }
    return combo;
}

} // namespace IodPretrainedTypes
