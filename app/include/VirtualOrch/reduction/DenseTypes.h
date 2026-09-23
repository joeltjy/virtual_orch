#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

/** Dense configuration and vocab.
 */
namespace DenseConfig {
inline constexpr int32_t MaxTimeInSeconds = 100;
inline constexpr int32_t MaxDurationInSeconds = 10;
inline constexpr int32_t TimeResolution = 100;

inline constexpr int32_t MaxPitch = 128;
inline constexpr int32_t MaxInstr = 5;
inline constexpr int32_t MaxNote = MaxPitch * MaxInstr; // 640
inline constexpr int32_t MaxVelocity = 128;
inline constexpr int32_t DefaultVelocity = 100;

inline constexpr int32_t MaxTime = TimeResolution * MaxTimeInSeconds; // 10000
inline constexpr int32_t MaxDur = TimeResolution * MaxDurationInSeconds; // 1000

/**
 * Fixed instrument roles for ReductionTransformerPianoReduction (amt_dense_maestro_v8
 * piano-reduction finetune): the live piano is always ingested as instrument 0
 * (conditioning only, never generated); the model only ever generates instrument 1
 * (the reduction, later consumed by OrchestrationTransformer). Unlike V1/V2, both
 * bands are meaningfully trained, so this is enforced by masking, not left to config.
 */
inline constexpr int32_t PianoReductionInputInstrument = 0;
inline constexpr int32_t PianoReductionOutputInstrument = 1;
} // namespace DenseConfig

namespace DenseVocab {
inline constexpr size_t TimeOffset = 0;
inline constexpr size_t DurOffset = TimeOffset + DenseConfig::MaxTime;       // 10000
inline constexpr size_t NoteOffset = DurOffset + DenseConfig::MaxDur;        // 11000
inline constexpr size_t VelocityOffset = NoteOffset + DenseConfig::MaxNote; // 11640

inline constexpr size_t VocabSize = VelocityOffset + DenseConfig::MaxVelocity; // 11768
} // namespace DenseVocab

/** MAESTRO v8 grids: duration 5 cs steps in [DurationMinCs, 995], velocity ÷8, pitches A0–C8. */
namespace DenseQuantize {
inline constexpr int32_t DurationStepCs = 5;
/** Sampling + snap floor (was 5; raised to curb live short-duration collapse). */
inline constexpr int32_t DurationMinCs = 10;
inline constexpr int32_t DurationMaxCs = 995;
inline constexpr int32_t VelocityStep = 8;
inline constexpr int32_t VelocityMin = 0;
inline constexpr int32_t VelocityMax = 120;
inline constexpr int32_t PianoKeyLow = 21;   // A0
inline constexpr int32_t PianoKeyHigh = 108; // C8

[[nodiscard]] inline auto snapDurationCs(int32_t cs) -> int32_t {
    const int32_t rounded =
        ((cs + DurationStepCs / 2) / DurationStepCs) * DurationStepCs;
    return std::clamp(rounded, DurationMinCs, DurationMaxCs);
}

[[nodiscard]] inline auto snapVelocity(int32_t vel) -> int32_t {
    const int32_t snapped = ((vel + VelocityStep / 2) / VelocityStep) * VelocityStep;
    return std::clamp(snapped, VelocityMin, VelocityMax);
}

[[nodiscard]] inline auto snapPianoPitch(int32_t pitch) -> int32_t {
    return std::clamp(pitch, PianoKeyLow, PianoKeyHigh);
}

[[nodiscard]] inline auto isLegalDurationCs(int32_t cs) -> bool {
    return cs >= DurationMinCs && cs <= DurationMaxCs && (cs % DurationStepCs) == 0;
}

[[nodiscard]] inline auto isLegalVelocity(int32_t vel) -> bool {
    return vel >= VelocityMin && vel <= VelocityMax && (vel % VelocityStep) == 0;
}

[[nodiscard]] inline auto isLegalPianoPitch(int32_t pitch) -> bool {
    return pitch >= PianoKeyLow && pitch <= PianoKeyHigh;
}

[[nodiscard]] inline auto snapDurationToken(int32_t durToken) -> int32_t {
    const int32_t cs = durToken - static_cast<int32_t>(DenseVocab::DurOffset);
    return static_cast<int32_t>(DenseVocab::DurOffset) + snapDurationCs(cs);
}

[[nodiscard]] inline auto snapVelocityToken(int32_t velToken) -> int32_t {
    const int32_t vel = velToken - static_cast<int32_t>(DenseVocab::VelocityOffset);
    return static_cast<int32_t>(DenseVocab::VelocityOffset) + snapVelocity(vel);
}

[[nodiscard]] inline auto snapNoteToken(int32_t noteToken) -> int32_t {
    const auto rel = noteToken - static_cast<int32_t>(DenseVocab::NoteOffset);
    if (rel < 0 || rel >= DenseConfig::MaxNote)
        return noteToken;
    const int32_t instr = rel / DenseConfig::MaxPitch;
    const int32_t pitch = snapPianoPitch(rel % DenseConfig::MaxPitch);
    return static_cast<int32_t>(DenseVocab::NoteOffset) + instr * DenseConfig::MaxPitch + pitch;
}

/**
 * Instrument band (0..MaxInstr-1) encoded in a packed dense note token, or -1 if
 * `noteToken` is not a note token (e.g. ClearQueue/BarSeparator).
 */
[[nodiscard]] inline auto instrumentOfNoteToken(int32_t noteToken) -> int32_t {
    const auto rel = noteToken - static_cast<int32_t>(DenseVocab::NoteOffset);
    if (rel < 0 || rel >= DenseConfig::MaxNote)
        return -1;
    return rel / DenseConfig::MaxPitch;
}

inline auto snapPackedEvents(std::vector<int32_t> &data, size_t eventWidth = 4) -> void {
    if (eventWidth < 4)
        return;
    for (size_t i = 0; i + 3 < data.size(); i += eventWidth) {
        data[i + 1] = snapDurationToken(data[i + 1]);
        data[i + 2] = snapNoteToken(data[i + 2]);
        data[i + 3] = snapVelocityToken(data[i + 3]);
    }
}

inline auto maskIllegalDurationLogits(std::vector<float> &logits) -> void {
    if (logits.size() < DenseVocab::NoteOffset)
        return;
    for (int32_t cs = 0; cs < DenseConfig::MaxDur; ++cs) {
        if (isLegalDurationCs(cs))
            continue;
        logits[DenseVocab::DurOffset + static_cast<size_t>(cs)] =
            -std::numeric_limits<float>::infinity();
    }
}

inline auto maskIllegalPianoNoteLogits(std::vector<float> &logits) -> void {
    if (logits.size() < DenseVocab::VelocityOffset)
        return;
    for (int32_t instr = 0; instr < DenseConfig::MaxInstr; ++instr) {
        const size_t band =
            DenseVocab::NoteOffset
            + static_cast<size_t>(instr) * static_cast<size_t>(DenseConfig::MaxPitch);
        for (int32_t pitch = 0; pitch < DenseConfig::MaxPitch; ++pitch) {
            if (isLegalPianoPitch(pitch))
                continue;
            logits[band + static_cast<size_t>(pitch)] =
                -std::numeric_limits<float>::infinity();
        }
    }
}

inline auto maskIllegalVelocityLogits(std::vector<float> &logits) -> void {
    if (logits.size() < DenseVocab::VocabSize)
        return;
    for (int32_t vel = 0; vel < DenseConfig::MaxVelocity; ++vel) {
        if (isLegalVelocity(vel))
            continue;
        logits[DenseVocab::VelocityOffset + static_cast<size_t>(vel)] =
            -std::numeric_limits<float>::infinity();
    }
}
} // namespace DenseQuantize

/**
 * Nucleus-sampling parameters per event field. Shared defaults for V1/V2;
 * V2 may diverge in sampling path without changing these constants.
 */
namespace DenseSampling {
inline constexpr float OnsetTopP = 0.98f;
inline constexpr float OnsetTemperature = 0.5f;
inline constexpr float DurationTopP = 0.98f;
/** Duration field temperature (V1 and V2). */
inline constexpr float DurationTemperature = 1.0f;
/** Alias kept for V2 call sites; same as DurationTemperature. */
inline constexpr float V2DurationTemperature = DurationTemperature;
inline constexpr float NoteTopP = 0.98f;
inline constexpr float NoteTemperature = 0.5f;
inline constexpr float VelocityTopP = 0.98f;
inline constexpr float VelocityTemperature = 0.5f;

/** Notes of reduction history fed to the dense ONNX (4 tokens each → 160 ids). */
inline constexpr int32_t ContextNotes = 40;

/**
 * True when `durToken` is still the MIDI provisional hold duration (`inputDuration`
 * cs), before note-off resolves it. Snapped to the dense duration grid.
 */
[[nodiscard]] inline auto isProvisionalDurationToken(int32_t durToken, int32_t provisionalCs)
    -> bool {
    const int32_t cs = DenseQuantize::snapDurationCs(
        durToken - static_cast<int32_t>(DenseVocab::DurOffset));
    return cs == DenseQuantize::snapDurationCs(provisionalCs);
}

/**
 * Copy up to `maxNotes` trailing dense events from `inputData`, skipping notes whose
 * duration is still the provisional `inputDuration`. Events stay in onset order.
 */
[[nodiscard]] inline auto copyDenseContextSkippingProvisional(
    const std::vector<int32_t> &inputData,
    int32_t maxNotes,
    int32_t provisionalCs,
    int eventWidth = 4) -> std::vector<int32_t> {
    std::vector<int32_t> out;
    if (eventWidth < 2 || inputData.size() < static_cast<size_t>(eventWidth) || maxNotes <= 0)
        return out;

    std::vector<size_t> keepStarts;
    keepStarts.reserve(static_cast<size_t>(maxNotes));
    for (size_t i = 0; i + static_cast<size_t>(eventWidth) - 1 < inputData.size();
         i += static_cast<size_t>(eventWidth)) {
        if (isProvisionalDurationToken(inputData[i + 1], provisionalCs))
            continue;
        keepStarts.push_back(i);
    }
    if (static_cast<int32_t>(keepStarts.size()) > maxNotes)
        keepStarts.erase(keepStarts.begin(),
                         keepStarts.end() - static_cast<std::ptrdiff_t>(maxNotes));

    out.reserve(keepStarts.size() * static_cast<size_t>(eventWidth));
    for (const size_t start: keepStarts) {
        for (int k = 0; k < eventWidth; ++k)
            out.push_back(inputData[start + static_cast<size_t>(k)]);
    }
    return out;
}

/** Notes of history the duration ceiling percentile is taken over (V1). */
inline constexpr int32_t CeilingRecentNotes = 15;
inline constexpr int32_t CeilingMultiplier = 4;
inline constexpr float CeilingDurationPercentile = 0.75f;
inline constexpr int32_t CeilingMinDurationCs = 40;
} // namespace DenseSampling

/**
 * Ordering used by the two-instrument (piano + reduction) dense model. The offline
 * training data is sorted by (onset, instrument) ascending — same-onset ties break by
 * ascending instrument, so a piano (instrument 0) event always precedes a reduction
 * (instrument 1) event at the same onset (see PIANO_REDUCTION_TRANSFORMER.md).
 * `NoteWindow::sortStridedByOnset` only compares onset and keeps arrival order on
 * ties, which does not match training once a stream mixes two real instruments — this
 * is additive (a new function, not a change to the shared onset-only sort) so it
 * cannot affect V1/V2/AMT, whose events are always instrument 0.
 */
namespace DensePianoReductionOrder {

inline auto sortStridedByOnsetThenInstrument(std::vector<int32_t> &data, size_t eventWidth = 4)
    -> void {
    if (eventWidth == 0 || data.size() < eventWidth * 2 || data.size() % eventWidth != 0)
        return;

    const size_t n = data.size() / eventWidth;
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        const size_t baseA = a * eventWidth;
        const size_t baseB = b * eventWidth;
        if (data[baseA] != data[baseB])
            return data[baseA] < data[baseB];
        return DenseQuantize::instrumentOfNoteToken(data[baseA + 2])
             < DenseQuantize::instrumentOfNoteToken(data[baseB + 2]);
    });

    bool alreadySorted = true;
    for (size_t i = 0; i < n; ++i) {
        if (order[i] != i) {
            alreadySorted = false;
            break;
        }
    }
    if (alreadySorted)
        return;

    std::vector<int32_t> sorted;
    sorted.reserve(data.size());
    for (const size_t idx: order) {
        const auto base = static_cast<std::ptrdiff_t>(idx * eventWidth);
        sorted.insert(sorted.end(),
                      data.begin() + base,
                      data.begin() + base + static_cast<std::ptrdiff_t>(eventWidth));
    }
    data.swap(sorted);
}

} // namespace DensePianoReductionOrder
