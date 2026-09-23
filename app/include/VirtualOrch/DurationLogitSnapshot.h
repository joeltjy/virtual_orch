#pragma once

#include <array>
#include <cstdint>

#include "VirtualOrch/reduction/DenseTypes.h"

/** Dense RT duration-field logits (post-mask) for the UI graph. */
struct DurationLogitSnapshot {
    static constexpr size_t kNumBins = static_cast<size_t>(DenseConfig::MaxDur); // 1000

    std::array<float, kNumBins> logits{};
    std::array<bool, kNumBins> valid{};
    /** Sampled duration in centiseconds (−1 if unknown). */
    int32_t sampledCs = -1;
    uint64_t sequence = 0;
};
