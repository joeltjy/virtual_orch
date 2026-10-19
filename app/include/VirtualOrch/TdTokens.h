#pragma once

#include <algorithm>
#include <cstdint>

#include "VirtualOrch/TemporalDensity.h"

/**
 * Temporal-density (TD) token ids for the models that read them, and the rules that turn the
 * Launchpad setting (TemporalDensity: AUTO or a level) into one.
 *
 * Boundaries lie every kBoundaryCs on the model's time axis; a TD token is written at each
 * one. Rebasing onsets for a model must therefore subtract a multiple of kBoundaryCs.
 */
namespace TdTokens {

/** Distance between two TD tokens (2 s). */
inline constexpr int32_t kBoundaryCs = 200;

// ---- Dense reduction models (T3/T6/T7, vocab 11775 or 11776) ----

/** Ids 11768..11775 = levels 0,1,2,4,8,16,32,64. Level 64 only exists in vocab 11776. */
inline constexpr int32_t kDenseFirst = 11768;
inline constexpr int32_t kDenseLevelCount = 8;
inline constexpr int32_t kDenseLevels[kDenseLevelCount] = {0, 1, 2, 4, 8, 16, 32, 64};
/** Level written for AUTO until the user has chosen one. */
inline constexpr int32_t kDenseDefaultLevel = 8;

/** Id for a dense level; levels above the vocab's top level clamp to it. -1 if not a level. */
[[nodiscard]] inline auto denseIdForLevel(int32_t level, int32_t vocabSize) -> int32_t {
    int32_t idx = -1;
    for (int32_t i = 0; i < kDenseLevelCount; ++i)
        if (kDenseLevels[i] == level)
            idx = i;
    if (idx < 0)
        return -1;
    const int32_t maxIdx = std::max(0, std::min(kDenseLevelCount, vocabSize - kDenseFirst) - 1);
    return kDenseFirst + std::min(idx, maxIdx);
}

/**
 * The dense vocab has no "unknown" id, so AUTO repeats the last level the user chose
 * (kDenseDefaultLevel before any choice).
 */
class DenseTdResolver {
public:
    [[nodiscard]] auto resolve(int32_t launchpadValue, int32_t vocabSize) -> int32_t {
        if (launchpadValue != TemporalDensity::kAuto && TemporalDensity::isValid(launchpadValue))
            lastLevel = launchpadValue;
        return denseIdForLevel(lastLevel, vocabSize);
    }

private:
    int32_t lastLevel = kDenseDefaultLevel;
};

// ---- IOD orchestration model with TD (T14, vocab 20178) ----

/** Ids 20170..20175 = levels 1,2,4,8,16,32; 20177 = unknown. */
inline constexpr int32_t kIodFirst = 20170;
inline constexpr int32_t kIodLevelCount = 6;
inline constexpr int32_t kIodUnknown = 20177;

/** AUTO → unknown. Level 64 has no id in this vocab and is written as 32. */
[[nodiscard]] inline auto iodIdForLaunchpad(int32_t launchpadValue) -> int32_t {
    if (launchpadValue == TemporalDensity::kAuto || ! TemporalDensity::isValid(launchpadValue))
        return kIodUnknown;
    int32_t level = 1;
    int32_t idx = 0;
    while (level < launchpadValue && idx < kIodLevelCount - 1) {
        level *= 2;
        ++idx;
    }
    return kIodFirst + idx;
}

} // namespace TdTokens
