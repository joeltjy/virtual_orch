#pragma once

#include <array>
#include <cstdint>

#include "VirtualOrch/InstrumentConstants.h"

/** Post-bias singleton family-combo logit per local instrument (UI bar chart). */
struct InstrumentLogitSnapshot {
    static constexpr size_t kNumInstruments = InstrumentConstants::kInstrumentMappings.size();

    std::array<float, kNumInstruments> logits{};
    std::array<bool, kNumInstruments> valid{};
    /** True when proportion bias was applied for this snapshot. */
    bool biasApplied = false;
    uint64_t sequence = 0;
};
