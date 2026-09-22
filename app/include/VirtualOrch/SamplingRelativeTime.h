#pragma once

#include <cstdint>
#include <limits>
#include <vector>

/**
 * Shared sampling-time helper: find min onset, relativize strided onset slots.
 * Use on a *copy* of the model input sequence before inference.
 * Reduction sampling adds the offset back onto the newly sampled time token;
 * orch models that copy absolute onsets from the original batch do not need add-back.
 */
namespace SamplingRelativeTime {

[[nodiscard]] inline auto minStridedOnset(const std::vector<int32_t> &seq, size_t stride)
    -> int32_t {
    if (stride == 0 || seq.size() < stride)
        return 0;
    int32_t minOnset = std::numeric_limits<int32_t>::max();
    for (size_t i = 0; i + (stride - 1) < seq.size(); i += stride)
        minOnset = std::min(minOnset, seq[i]);
    return minOnset == std::numeric_limits<int32_t>::max() ? 0 : minOnset;
}

inline auto relativizeStridedOnsets(std::vector<int32_t> &seq, size_t stride, int32_t offset)
    -> void {
    if (offset == 0 || stride == 0)
        return;
    for (size_t i = 0; i < seq.size(); i += stride)
        seq[i] -= offset;
}

inline auto restoreStridedOnsets(std::vector<int32_t> &seq, size_t stride, int32_t offset)
    -> void {
    relativizeStridedOnsets(seq, stride, -offset);
}

} // namespace SamplingRelativeTime
