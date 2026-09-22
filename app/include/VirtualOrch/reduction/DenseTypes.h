#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
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
} // namespace DenseConfig

namespace DenseVocab {
inline constexpr size_t TimeOffset = 0;
inline constexpr size_t DurOffset = TimeOffset + DenseConfig::MaxTime;       // 10000
inline constexpr size_t NoteOffset = DurOffset + DenseConfig::MaxDur;        // 11000
inline constexpr size_t VelocityOffset = NoteOffset + DenseConfig::MaxNote; // 11640

inline constexpr size_t VocabSize = VelocityOffset + DenseConfig::MaxVelocity; // 11768
} // namespace DenseVocab

/** MAESTRO v8 grids: duration 5 cs, velocity ÷8, pitches A0–C8 (masks in V2). */
namespace DenseQuantize {
inline constexpr int32_t DurationStepCs = 5;
inline constexpr int32_t DurationMinCs = 5;
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
inline constexpr float DurationTemperature = 1.0f; // V1 default
/** V2 default: same temperature on duration as onset/note/velocity. */
inline constexpr float V2DurationTemperature = 0.5f;
inline constexpr float NoteTopP = 0.98f;
inline constexpr float NoteTemperature = 0.5f;
inline constexpr float VelocityTopP = 0.98f;
inline constexpr float VelocityTemperature = 0.5f;

/** Notes of reduction history fed to the dense ONNX (4 tokens each → 160 ids). */
inline constexpr int32_t ContextNotes = 40;

/** Notes of history the duration ceiling percentile is taken over (V1). */
inline constexpr int32_t CeilingRecentNotes = 15;
inline constexpr int32_t CeilingMultiplier = 4;
inline constexpr float CeilingDurationPercentile = 0.75f;
inline constexpr int32_t CeilingMinDurationCs = 40;
} // namespace DenseSampling
