#pragma once

#include <atomic>
#include <cstdint>

/**
 * Global switch for the reduction models' logit biases (the pitch, duration and onset-gap
 * anti-repetition biases): on / off, and when on, either always or only in the parts where the
 * piano has been silent for more than a second. Written by the Logit Adjustments widget, read by
 * every reduction transformer before it draws a note.
 */
class BiasSettings {
public:
    /** "The piano is silent" = no key down and the last key was released this long ago. */
    static constexpr int64_t kSilentAfterMs = 1000;

    [[nodiscard]] static auto enabled() -> bool { return enabledFlag.load(std::memory_order_relaxed); }
    static auto setEnabled(bool on) -> void { enabledFlag.store(on, std::memory_order_relaxed); }

    [[nodiscard]] static auto onlyWhenPianoSilent() -> bool {
        return gateFlag.load(std::memory_order_relaxed);
    }
    static auto setOnlyWhenPianoSilent(bool on) -> void { gateFlag.store(on, std::memory_order_relaxed); }

    /**
     * Whether the biases apply now. pianoSilentForMs: 0 while a key is down, else ms since the last
     * release (a very large value if nothing was ever played).
     */
    [[nodiscard]] static auto allowsNow(int64_t pianoSilentForMs) -> bool {
        if (! enabled())
            return false;
        return ! onlyWhenPianoSilent() || pianoSilentForMs > kSilentAfterMs;
    }

private:
    static inline std::atomic<bool> enabledFlag{true};
    static inline std::atomic<bool> gateFlag{false};
};
