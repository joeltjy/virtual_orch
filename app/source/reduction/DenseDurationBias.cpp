#include "VirtualOrch/reduction/DenseDurationBias.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace DenseDurationBias {
namespace {

constexpr int denseEventWidth = 4;

auto lastOnsetCs(const std::vector<int32_t> &denseInputData) -> int32_t {
    int32_t last = 0;
    for (size_t i = 0; i + 3 < denseInputData.size(); i += denseEventWidth)
        last = denseInputData[i];
    return last;
}

auto resolveNowCs(const std::vector<int32_t> &denseInputData, int32_t nowCs) -> int32_t {
    const int32_t last = lastOnsetCs(denseInputData);
    if (nowCs < 0)
        return last;
    return std::max(nowCs, last);
}

auto collectDurationWindowWithNow(const std::vector<int32_t> &denseInputData,
                                  int32_t now,
                                  int32_t skipProvisionalCs,
                                  int32_t onlyInstrument) -> DurationWindowStats {
    DurationWindowStats stats;
    const int32_t cutoff = now - WindowCs;
    const int32_t skipSnapped =
        skipProvisionalCs >= 0 ? DenseQuantize::snapDurationCs(skipProvisionalCs) : -1;

    std::array<int, LegalDurBins> durCounts{};
    std::array<int, DeltaBins> deltaCounts{};
    std::vector<int32_t> orderedDurs;
    orderedDurs.reserve(denseInputData.size() / denseEventWidth);

    int inRangeDeltaPairs = 0;
    for (size_t i = 0; i + 3 < denseInputData.size(); i += denseEventWidth) {
        const int32_t onset = denseInputData[i];
        if (onset <= cutoff || onset > now)
            continue;
        if (onlyInstrument >= 0) {
            const int32_t instr = DenseQuantize::instrumentOfNoteToken(denseInputData[i + 2]);
            if (instr != onlyInstrument)
                continue;
        }
        int32_t durCs = durationCsFromToken(denseInputData[i + 1]);
        durCs = DenseQuantize::snapDurationCs(durCs);
        if (! isLegalDurationCs(durCs))
            continue;
        if (skipSnapped >= 0 && durCs == skipSnapped)
            continue;

        ++durCounts[static_cast<size_t>(legalDurIndex(durCs))];
        if (! orderedDurs.empty()) {
            const int32_t delta = durCs - orderedDurs.back();
            if (isInRangeDeltaCs(delta)) {
                ++deltaCounts[static_cast<size_t>(deltaIndex(delta))];
                ++inRangeDeltaPairs;
            }
        }
        orderedDurs.push_back(durCs);
    }

    stats.noteCount = static_cast<int>(orderedDurs.size());
    if (! orderedDurs.empty())
        stats.lastDurationCs = orderedDurs.back();

    std::set<int32_t> unique;
    for (const auto d: orderedDurs)
        unique.insert(d);
    stats.uniqueDurations.assign(unique.begin(), unique.end());

    std::set<int32_t> uniqueDeltas;
    for (size_t i = 1; i < orderedDurs.size(); ++i) {
        const int32_t delta = orderedDurs[i] - orderedDurs[i - 1];
        if (isInRangeDeltaCs(delta))
            uniqueDeltas.insert(delta);
    }
    stats.uniqueDeltas.assign(uniqueDeltas.begin(), uniqueDeltas.end());

    const float n = static_cast<float>(std::max(stats.noteCount, 0));
    const float piDenom = n + static_cast<float>(LegalDurBins) * UnigramEpsilon;
    for (int i = 0; i < LegalDurBins; ++i) {
        stats.pi[static_cast<size_t>(i)] =
            (static_cast<float>(durCounts[static_cast<size_t>(i)]) + UnigramEpsilon) / piDenom;
    }

    const float d = static_cast<float>(std::max(inRangeDeltaPairs, 0));
    const float deltaDenom = d + static_cast<float>(DeltaBins) * DeltaEpsilon;
    for (int i = 0; i < DeltaBins; ++i) {
        stats.sDelta[static_cast<size_t>(i)] =
            (static_cast<float>(deltaCounts[static_cast<size_t>(i)]) + DeltaEpsilon) / deltaDenom;
    }

    return stats;
}

} // namespace

auto collectDurationWindow(const std::vector<int32_t> &denseInputData,
                           int32_t nowCs,
                           int32_t skipProvisionalCs,
                           int32_t onlyInstrument) -> DurationWindowStats {
    return collectDurationWindowWithNow(denseInputData, resolveNowCs(denseInputData, nowCs),
                                        skipProvisionalCs, onlyInstrument);
}

auto collectDurationWindowAt(const std::vector<int32_t> &denseInputData,
                             int32_t prefixNowCs,
                             int32_t skipProvisionalCs,
                             int32_t onlyInstrument) -> DurationWindowStats {
    return collectDurationWindowWithNow(denseInputData, prefixNowCs, skipProvisionalCs,
                                        onlyInstrument);
}

auto applyDurationLogitsBias(std::vector<float> &logits,
                             const DurationWindowStats &stats,
                             float tau,
                             float tauPrime) -> void {
    if (stats.noteCount <= 0 || stats.lastDurationCs < 0)
        return;
    if (logits.size() < DenseVocab::NoteOffset)
        return;

    for (int32_t cs = DenseQuantize::DurationMinCs; cs <= DenseQuantize::DurationMaxCs;
         cs += DenseQuantize::DurationStepCs) {
        float &logit = logits[DenseVocab::DurOffset + static_cast<size_t>(cs)];
        if (! std::isfinite(logit))
            continue;

        const float pi = stats.pi[static_cast<size_t>(legalDurIndex(cs))];
        logit -= tau * std::log(pi);

        const int32_t delta = cs - stats.lastDurationCs;
        if (! isInRangeDeltaCs(delta))
            continue;
        const float sy = stats.sDelta[static_cast<size_t>(deltaIndex(delta))];
        logit -= tauPrime * std::log(sy);
    }
}

auto DurationTauScheduler::evaluate(int32_t nowCs,
                                    const std::vector<int32_t> &uniqueDurations,
                                    const std::vector<int32_t> &uniqueDeltas,
                                    float tauBase,
                                    float tauPrimeBase,
                                    bool countNote) -> Result {
    // UI timer and the generation thread both call evaluate. Generation often runs
    // ahead of the wall clock; a later UI refresh with a smaller "now" must not
    // rewind the hold/ramp (that pinned τ_d near base, e.g. 0.2→0.3 forever).
    if (hasScheduleNow)
        nowCs = std::max(nowCs, lastScheduleNowCs);
    lastScheduleNowCs = nowCs;
    hasScheduleNow = true;

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

    const auto heldValue = [](int32_t nowCsLocal, int32_t stableSinceCs, int notesSinceReset,
                              float base) -> float {
        const float heldSeconds =
            static_cast<float>(nowCsLocal - stableSinceCs)
            / static_cast<float>(DenseConfig::TimeResolution);
        if (heldSeconds < TauHoldSeconds && notesSinceReset < TauHoldNotes)
            return base;
        return std::min(base * TauRampCapMultiplier,
                        base * std::pow(TauRampPerSecond, static_cast<float>(notesSinceReset) / 10.0f));
    };

    if (uniqueDurations.empty()) {
        lastDurations.clear();
        hasDurations = false;
        durationNotesSinceReset = 0;
    } else if (gainedElement(hasDurations, lastDurations, uniqueDurations)) {
        lastDurations = uniqueDurations;
        durationStableSinceCs = nowCs;
        durationNotesSinceReset = countNote ? 1 : 0;
        hasDurations = true;
    } else {
        lastDurations = uniqueDurations;
        if (countNote)
            ++durationNotesSinceReset;
    }

    if (uniqueDeltas.empty()) {
        lastDeltas.clear();
        hasDeltas = false;
        deltaNotesSinceReset = 0;
    } else if (gainedElement(hasDeltas, lastDeltas, uniqueDeltas)) {
        lastDeltas = uniqueDeltas;
        deltaStableSinceCs = nowCs;
        deltaNotesSinceReset = countNote ? 1 : 0;
        hasDeltas = true;
    } else {
        lastDeltas = uniqueDeltas;
        if (countNote)
            ++deltaNotesSinceReset;
    }

    return {hasDurations ? heldValue(nowCs, durationStableSinceCs, durationNotesSinceReset, tauBase)
                         : tauBase,
            hasDeltas ? heldValue(nowCs, deltaStableSinceCs, deltaNotesSinceReset, tauPrimeBase)
                      : tauPrimeBase};
}

auto DurationTauScheduler::reset() -> void {
    lastDurations.clear();
    lastDeltas.clear();
    durationStableSinceCs = 0;
    deltaStableSinceCs = 0;
    lastScheduleNowCs = 0;
    durationNotesSinceReset = 0;
    deltaNotesSinceReset = 0;
    hasDurations = false;
    hasDeltas = false;
    hasScheduleNow = false;
}

auto collectCoarseDurationWindowAt(const std::vector<int32_t> &denseInputData,
                                   int32_t nowCs,
                                   int32_t onlyInstrument) -> CoarseDurationStats {
    CoarseDurationStats stats;
    std::array<int, CoarseBins> counts{};
    std::array<int, CoarseMoves> moves{};
    std::set<int32_t> bins;
    std::set<int32_t> moveSet;
    int pairs = 0;
    for (size_t i = 0; i + 3 < denseInputData.size(); i += denseEventWidth) {
        const int32_t onset = denseInputData[i];
        if (onset <= nowCs - WindowCs || onset > nowCs)
            continue;
        if (onlyInstrument >= 0 && DenseQuantize::instrumentOfNoteToken(denseInputData[i + 2]) != onlyInstrument)
            continue;
        const int bin = coarseBin(DenseQuantize::snapDurationCs(durationCsFromToken(denseInputData[i + 1])));
        ++counts[static_cast<size_t>(bin)];
        bins.insert(bin);
        if (stats.lastBin >= 0) {
            const int move = bin - stats.lastBin;
            ++moves[static_cast<size_t>(move + CoarseBins - 1)];
            moveSet.insert(move);
            ++pairs;
        }
        stats.lastBin = bin;
        ++stats.noteCount;
    }
    const float n = static_cast<float>(stats.noteCount) + CoarseBins * CoarseEpsilon;
    for (int b = 0; b < CoarseBins; ++b)
        stats.pi[static_cast<size_t>(b)] = (static_cast<float>(counts[static_cast<size_t>(b)]) + CoarseEpsilon) / n;
    const float d = static_cast<float>(pairs) + CoarseMoves * CoarseEpsilon;
    for (int m = 0; m < CoarseMoves; ++m)
        stats.sMove[static_cast<size_t>(m)] = (static_cast<float>(moves[static_cast<size_t>(m)]) + CoarseEpsilon) / d;
    stats.uniqueBins.assign(bins.begin(), bins.end());
    stats.uniqueMoves.assign(moveSet.begin(), moveSet.end());
    return stats;
}

auto applyCoarseDurationLogitsBias(std::vector<float> &logits,
                                   const CoarseDurationStats &stats,
                                   float tau,
                                   float tauPrime) -> void {
    if (stats.noteCount <= 0 || stats.lastBin < 0 || logits.size() < DenseVocab::NoteOffset)
        return;
    for (int32_t cs = DenseQuantize::DurationMinCs; cs <= DenseQuantize::DurationMaxCs;
         cs += DenseQuantize::DurationStepCs) {
        float &logit = logits[DenseVocab::DurOffset + static_cast<size_t>(cs)];
        if (! std::isfinite(logit))
            continue;
        const int bin = coarseBin(cs);
        logit -= tau * std::log(stats.pi[static_cast<size_t>(bin)]);
        logit -= tauPrime * std::log(stats.sMove[static_cast<size_t>(bin - stats.lastBin + CoarseBins - 1)]);
    }
}

} // namespace DenseDurationBias
