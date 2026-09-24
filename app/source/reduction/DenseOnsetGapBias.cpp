#include "VirtualOrch/reduction/DenseOnsetGapBias.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace DenseOnsetGapBias {
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

auto collectOnsetGapWindowWithNow(const std::vector<int32_t> &denseInputData,
                                  int32_t now,
                                  int32_t skipProvisionalCs) -> OnsetGapWindowStats {
    OnsetGapWindowStats stats;
    const int32_t cutoff = now - WindowCs;
    const int32_t skipSnapped =
        skipProvisionalCs >= 0 ? DenseQuantize::snapDurationCs(skipProvisionalCs) : -1;

    std::array<int, GapBins> gapCounts{};
    std::vector<int32_t> orderedOnsets;
    orderedOnsets.reserve(denseInputData.size() / denseEventWidth);

    int inRangeGapPairs = 0;
    for (size_t i = 0; i + 3 < denseInputData.size(); i += denseEventWidth) {
        const int32_t onset = denseInputData[i];
        if (onset <= cutoff || onset > now)
            continue;
        if (skipSnapped >= 0) {
            const int32_t durCs = DenseQuantize::snapDurationCs(
                denseInputData[i + 1] - static_cast<int32_t>(DenseVocab::DurOffset));
            if (durCs == skipSnapped)
                continue;
        }

        if (! orderedOnsets.empty()) {
            const int32_t gap = onset - orderedOnsets.back();
            if (isInRangeGapCs(gap)) {
                ++gapCounts[static_cast<size_t>(gapIndex(gap))];
                ++inRangeGapPairs;
            }
        }
        orderedOnsets.push_back(onset);
    }

    stats.noteCount = static_cast<int>(orderedOnsets.size());
    stats.inRangeGapPairs = inRangeGapPairs;
    if (! orderedOnsets.empty())
        stats.lastOnsetCs = orderedOnsets.back();

    std::set<int32_t> uniqueGaps;
    for (size_t i = 1; i < orderedOnsets.size(); ++i) {
        const int32_t gap = orderedOnsets[i] - orderedOnsets[i - 1];
        if (isInRangeGapCs(gap))
            uniqueGaps.insert(gap);
    }
    stats.uniqueGaps.assign(uniqueGaps.begin(), uniqueGaps.end());

    const float d = static_cast<float>(std::max(inRangeGapPairs, 0));
    const float denom = d + static_cast<float>(GapBins) * GapEpsilon;
    for (int i = 0; i < GapBins; ++i) {
        stats.sGap[static_cast<size_t>(i)] =
            (static_cast<float>(gapCounts[static_cast<size_t>(i)]) + GapEpsilon) / denom;
    }

    return stats;
}

} // namespace

auto collectOnsetGapWindow(const std::vector<int32_t> &denseInputData,
                           int32_t nowCs,
                           int32_t skipProvisionalCs) -> OnsetGapWindowStats {
    return collectOnsetGapWindowWithNow(denseInputData, resolveNowCs(denseInputData, nowCs),
                                        skipProvisionalCs);
}

auto collectOnsetGapWindowAt(const std::vector<int32_t> &denseInputData,
                             int32_t prefixNowCs,
                             int32_t skipProvisionalCs) -> OnsetGapWindowStats {
    return collectOnsetGapWindowWithNow(denseInputData, prefixNowCs, skipProvisionalCs);
}

auto applyOnsetLogitsBias(std::vector<float> &logits,
                          const OnsetGapWindowStats &stats,
                          float tau,
                          int32_t timeOffset) -> void {
    if (stats.noteCount <= 0 || stats.lastOnsetCs < 0)
        return;
    if (logits.size() < DenseVocab::DurOffset)
        return;

    for (int32_t t = 0; t < DenseConfig::MaxTime; ++t) {
        float &logit = logits[DenseVocab::TimeOffset + static_cast<size_t>(t)];
        if (! std::isfinite(logit))
            continue;

        const int32_t absOnset = t + timeOffset;
        const int32_t gap = absOnset - stats.lastOnsetCs;
        if (! isInRangeGapCs(gap))
            continue;
        const float sy = stats.sGap[static_cast<size_t>(gapIndex(gap))];
        logit -= tau * std::log(sy);
    }
}

} // namespace DenseOnsetGapBias
