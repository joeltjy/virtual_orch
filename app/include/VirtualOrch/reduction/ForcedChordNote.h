#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

/**
 * Forced chord notes (Nov 26 2026). A chord is built bottom-up: each stacked note must be above the last, so
 * once the line is high only a few pitches are left under the top of the register, and the mask hands them all
 * the probability even when the model gave them ~1% (sample viewer: clusters at 104-108 under C8). The engines
 * end the chord instead when the pitches still allowed hold less than Settings::minChordNoteShare of the model's
 * own pitch-step distribution (raw logits, temperature 1, over the orchestra and piano note blocks).
 */
namespace ForcedChordNote {

inline constexpr float kDefaultMinShare = 0.05f;

/** The share of `allowed` ([lo, hi) id ranges) in the softmax over `blocks` ([lo, hi) id ranges). */
[[nodiscard]] inline auto share(const std::vector<float> &logits, const std::vector<std::pair<int32_t, int32_t>> &allowed,
                                const std::vector<std::pair<int32_t, int32_t>> &blocks) -> double {
    float peak = -std::numeric_limits<float>::infinity();
    const auto each = [&](const std::vector<std::pair<int32_t, int32_t>> &ranges, auto &&fn) {
        for (const auto &[lo, hi]: ranges)
            for (int32_t id = std::max(lo, 0); id < hi && static_cast<size_t>(id) < logits.size(); ++id)
                if (std::isfinite(logits[static_cast<size_t>(id)]))
                    fn(logits[static_cast<size_t>(id)]);
    };
    each(blocks, [&](float x) { peak = std::max(peak, x); });
    if (! std::isfinite(peak))
        return 1.0;
    double total = 0.0, part = 0.0;
    each(blocks, [&](float x) { total += std::exp(static_cast<double>(x - peak)); });
    each(allowed, [&](float x) { part += std::exp(static_cast<double>(x - peak)); });
    return total > 0.0 ? part / total : 1.0;
}

} // namespace ForcedChordNote
