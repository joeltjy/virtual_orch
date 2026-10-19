#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>

class TemporalDensity {
public:
    static constexpr int32_t kAuto = 0;
    static constexpr std::array<int32_t, 7> kLevels{{1, 2, 4, 8, 16, 32, 64}};

    [[nodiscard]] static auto get() -> int32_t {
        return value.load(std::memory_order_relaxed);
    }

    [[nodiscard]] static auto isAuto() -> bool {
        return get() == kAuto;
    }

    [[nodiscard]] static auto isValid(int32_t candidate) -> bool {
        return candidate == kAuto
               || std::find(kLevels.begin(), kLevels.end(), candidate) != kLevels.end();
    }

private:
    friend class LaunchpadGrid;

    static auto set(int32_t newValue) -> void {
        if (isValid(newValue))
            value.store(newValue, std::memory_order_relaxed);
    }

    static inline std::atomic<int32_t> value{kAuto};
};
