#pragma once

#include <atomic>

/**
 * Launchpad top button CC107 (scene 3): while on, the user's input never reaches the reduction
 * transformer. InputFilter skips the reduction's input (and conditioning) queue; the notes still
 * go to the orchestration (input instruments) and their note-off updates still flow, so nothing
 * hangs. The reduction keeps generating from what it already has. Starts off.
 */
class ReductionInputGate {
public:
    [[nodiscard]] static auto isBlocked() -> bool { return blocked.load(std::memory_order_relaxed); }
    static auto setBlocked(bool on) -> void { blocked.store(on, std::memory_order_relaxed); }

private:
    static inline std::atomic<bool> blocked{false};
};
