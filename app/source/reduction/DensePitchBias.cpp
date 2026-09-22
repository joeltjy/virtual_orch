#include "VirtualOrch/reduction/DensePitchBias.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace DensePitchBias {
namespace {

constexpr int denseEventWidth = 4;

auto lastOnsetCs(const std::vector<int32_t> &denseInputData) -> int32_t {
    int32_t last = 0;
    for (size_t i = 0; i + 3 < denseInputData.size(); i += denseEventWidth)
        last = denseInputData[i];
    return last;
}

/**
 * amt_causal offline sampling uses the last onset in the growing prefix as "now"
 * for the 5 s π / s_y window and the τ hold/ramp. Live RT also has a wall clock:
 * take max(clock, lastOnset) so generated-ahead notes stay inside the window (like
 * offline) while silence can still age the window once the clock catches up.
 * Pass nowCs < 0 when there is no clock (tests / offline-style).
 */
auto resolveNowCs(const std::vector<int32_t> &denseInputData, int32_t nowCs) -> int32_t {
    const int32_t last = lastOnsetCs(denseInputData);
    if (nowCs < 0)
        return last;
    return std::max(nowCs, last);
}

} // namespace

namespace {

auto collectPitchWindowWithNow(const std::vector<int32_t> &denseInputData, int32_t now)
    -> PitchWindowStats {
    PitchWindowStats stats;
    const int32_t cutoff = now - WindowCs;

    std::array<int, DenseConfig::MaxPitch> pitchCounts{};
    std::array<int, DeltaBins> deltaCounts{};
    std::vector<int32_t> orderedPitches;
    orderedPitches.reserve(denseInputData.size() / denseEventWidth);

    int deltaPairs = 0;
    for (size_t i = 0; i + 3 < denseInputData.size(); i += denseEventWidth) {
        const int32_t onset = denseInputData[i];
        if (onset <= cutoff || onset > now)
            continue;
        const int32_t pitch = pitchFromDenseNoteToken(denseInputData[i + 2]);
        if (pitch < 0 || pitch >= DenseConfig::MaxPitch)
            continue;
        ++pitchCounts[static_cast<size_t>(pitch)];
        if (! orderedPitches.empty()) {
            const int delta = pitch - orderedPitches.back();
            ++deltaCounts[static_cast<size_t>(deltaIndex(delta))];
            ++deltaPairs;
        }
        orderedPitches.push_back(pitch);
    }

    stats.noteCount = static_cast<int>(orderedPitches.size());
    if (! orderedPitches.empty())
        stats.lastPitch = orderedPitches.back();

    std::set<int32_t> unique;
    for (const auto p: orderedPitches)
        unique.insert(p);
    stats.uniquePitches.assign(unique.begin(), unique.end());

    std::set<int32_t> uniqueDeltas;
    for (size_t i = 1; i < orderedPitches.size(); ++i)
        uniqueDeltas.insert(orderedPitches[i] - orderedPitches[i - 1]);
    stats.uniqueDeltas.assign(uniqueDeltas.begin(), uniqueDeltas.end());

    const float n = static_cast<float>(std::max(stats.noteCount, 0));
    const float piDenom = n + static_cast<float>(DenseConfig::MaxPitch) * UnigramEpsilon;
    for (int p = 0; p < DenseConfig::MaxPitch; ++p) {
        stats.pi[static_cast<size_t>(p)] =
            (static_cast<float>(pitchCounts[static_cast<size_t>(p)]) + UnigramEpsilon) / piDenom;
    }

    const float d = static_cast<float>(std::max(deltaPairs, 0));
    const float deltaDenom = d + static_cast<float>(DeltaBins) * DeltaEpsilon;
    for (int i = 0; i < DeltaBins; ++i) {
        stats.sDelta[static_cast<size_t>(i)] =
            (static_cast<float>(deltaCounts[static_cast<size_t>(i)]) + DeltaEpsilon) / deltaDenom;
    }

    return stats;
}

} // namespace

auto collectPitchWindow(const std::vector<int32_t> &denseInputData, int32_t nowCs)
    -> PitchWindowStats {
    return collectPitchWindowWithNow(denseInputData, resolveNowCs(denseInputData, nowCs));
}

auto collectPitchWindowAt(const std::vector<int32_t> &denseInputData, int32_t prefixNowCs)
    -> PitchWindowStats {
    return collectPitchWindowWithNow(denseInputData, prefixNowCs);
}

auto applyNoteLogitsBias(std::vector<float> &logits,
                         const PitchWindowStats &stats,
                         float tau,
                         float tauPrime) -> void {
    if (stats.noteCount <= 0 || stats.lastPitch < 0)
        return;
    if (logits.size() < DenseVocab::VocabSize)
        return;

    for (int instr = 0; instr < DenseConfig::MaxInstr; ++instr) {
        const size_t bandBase =
            DenseVocab::NoteOffset + static_cast<size_t>(instr) * DenseConfig::MaxPitch;
        for (int pitch = 0; pitch < DenseConfig::MaxPitch; ++pitch) {
            float &logit = logits[bandBase + static_cast<size_t>(pitch)];
            if (! std::isfinite(logit))
                continue;

            const float pi = stats.pi[static_cast<size_t>(pitch)];
            const int delta = pitch - stats.lastPitch;
            const float sy = stats.sDelta[static_cast<size_t>(deltaIndex(delta))];
            logit -= tau * std::log(pi);
            logit -= tauPrime * std::log(sy);
        }
    }
}

auto PitchTauScheduler::evaluate(int32_t nowCs,
                                 const std::vector<int32_t> &uniquePitches,
                                 const std::vector<int32_t> &uniqueDeltas) -> Result {
    const auto gainedElement = [](bool hasPrev,
                                  const std::vector<int32_t> &prev,
                                  const std::vector<int32_t> &cur) -> bool {
        if (! hasPrev)
            return true;
        for (const int32_t value: cur) {
            if (! std::binary_search(prev.begin(), prev.end(), value))
                return true;
        }
        return false;
    };

    const auto heldValue = [](int32_t nowCs, int32_t stableSinceCs) -> float {
        const float heldSeconds =
            static_cast<float>(nowCs - stableSinceCs)
            / static_cast<float>(DenseConfig::TimeResolution);
        if (heldSeconds < TauHoldSeconds)
            return TauBase;
        const float extra = heldSeconds - TauHoldSeconds;
        return TauBase * std::pow(2.0f, extra);
    };

    // τ: reset only when S gains a pitch.
    if (gainedElement(hasS, lastS, uniquePitches)) {
        lastS = uniquePitches;
        sStableSinceCs = nowCs;
        hasS = true;
    } else {
        lastS = uniquePitches;
    }

    // τ′: reset only when a new pitch interval enters the window.
    if (gainedElement(hasDeltas, lastDeltas, uniqueDeltas)) {
        lastDeltas = uniqueDeltas;
        deltaStableSinceCs = nowCs;
        hasDeltas = true;
    } else {
        lastDeltas = uniqueDeltas;
    }

    return {heldValue(nowCs, sStableSinceCs), heldValue(nowCs, deltaStableSinceCs)};
}

auto PitchTauScheduler::reset() -> void {
    lastS.clear();
    lastDeltas.clear();
    sStableSinceCs = 0;
    deltaStableSinceCs = 0;
    hasS = false;
    hasDeltas = false;
}

} // namespace DensePitchBias
