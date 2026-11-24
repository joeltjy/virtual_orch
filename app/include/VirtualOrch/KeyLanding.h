#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>

/**
 * When each live key landed (its note-on reached the app), in steady-clock microseconds, by the
 * key id MidiInputProcess gives it (Token::keyId). The reduction reads it to time a WAIT's fire
 * from the pianist's note to the last orchestra note placed (WAIT Visualizer). Lock-free: the
 * MIDI thread writes, the reduction thread reads; the last kSlots keys are kept.
 */
class KeyLanding {
public:
    static constexpr int32_t kSlots = 1024;

    [[nodiscard]] static auto nowUs() -> int64_t {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    static auto record(int32_t keyId, int64_t us = nowUs()) -> void {
        if (keyId < 0)
            return;
        auto &s = slots()[static_cast<size_t>(keyId % kSlots)];
        s.us.store(us, std::memory_order_relaxed);
        s.key.store(keyId, std::memory_order_release);
    }

    /** When key `keyId` landed; -1 if unknown (no id, or overwritten). */
    [[nodiscard]] static auto get(int32_t keyId) -> int64_t {
        if (keyId < 0)
            return -1;
        const auto &s = slots()[static_cast<size_t>(keyId % kSlots)];
        if (s.key.load(std::memory_order_acquire) != keyId)
            return -1;
        return s.us.load(std::memory_order_relaxed);
    }

private:
    struct Slot {
        std::atomic<int32_t> key{-1};
        std::atomic<int64_t> us{-1};
    };
    static auto slots() -> std::array<Slot, kSlots> & {
        static std::array<Slot, kSlots> s;
        return s;
    }
};
