#pragma once

#include <array>
#include <atomic>
#include <cstdint>

/** Live counts of IOD octave Δ ∈ {-1, 0, +1} from decoded orchestration notes. */
struct OctaveDeltaSnapshot {
    /** Index 0 → Δ=−1, 1 → Δ=0, 2 → Δ=+1. */
    std::array<uint64_t, 3> counts{};
    uint64_t total = 0;
    uint64_t sequence = 0;
};

/**
 * Thread-safe accumulator: OT decode thread records; UI timer reads snapshots.
 * Index = delta + 1.
 */
class OctaveDeltaTracker {
public:
    auto reset() -> void {
        for (auto &c: counts)
            c.store(0, std::memory_order_relaxed);
        total.store(0, std::memory_order_relaxed);
        sequence.fetch_add(1, std::memory_order_relaxed);
    }

    auto record(int32_t delta) -> void {
        if (delta < -1 || delta > 1)
            return;
        counts[static_cast<size_t>(delta + 1)].fetch_add(1, std::memory_order_relaxed);
        total.fetch_add(1, std::memory_order_relaxed);
        sequence.fetch_add(1, std::memory_order_relaxed);
    }

    [[nodiscard]] auto snapshot() const -> OctaveDeltaSnapshot {
        OctaveDeltaSnapshot snap;
        for (size_t i = 0; i < snap.counts.size(); ++i)
            snap.counts[i] = counts[i].load(std::memory_order_relaxed);
        snap.total = total.load(std::memory_order_relaxed);
        snap.sequence = sequence.load(std::memory_order_relaxed);
        return snap;
    }

private:
    std::array<std::atomic<uint64_t>, 3> counts{};
    std::atomic<uint64_t> total{0};
    std::atomic<uint64_t> sequence{0};
};
