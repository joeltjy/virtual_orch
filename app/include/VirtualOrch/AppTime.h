#pragma once

#include <JuceHeader.h>

#include <atomic>
#include <cstdint>

/**
 * The time the performance pipeline reads (input, reductions, orchestration, playback, Launchpad).
 *
 * Normally the real clocks. The offline simulator (virtual-orch-sim, LogReplay) sets a simulated
 * wall time with setSimulatedMs(): every reader below then returns that instead, so a whole
 * session can be stepped faster or slower than real time and gives the same result every run.
 * The app never sets it.
 *
 *   nowMs()      wall clock in ms           (juce::Time::currentTimeMillis)
 *   counterMs()  ms counter                 (juce::Time::getMillisecondCounter)
 *   hiResMs()    high-resolution ms counter (juce::Time::getMillisecondCounterHiRes)
 *
 * Loop-timing profiles (how long an inference took) keep using the real counters.
 */
namespace AppTime {

inline auto simulated() -> std::atomic<double> & {
    static std::atomic<double> ms{-1.0};
    return ms;
}

/** Use `ms` as the current time everywhere (>= 0, sub-ms allowed), or -1 for the real clocks again. */
inline auto setSimulatedMs(double ms) -> void { simulated().store(ms); }
[[nodiscard]] inline auto isSimulated() -> bool { return simulated().load() >= 0.0; }

[[nodiscard]] inline auto nowMs() -> int64_t {
    const auto s = simulated().load();
    return s >= 0.0 ? static_cast<int64_t>(s) : juce::Time::currentTimeMillis();
}

[[nodiscard]] inline auto counterMs() -> uint32_t {
    const auto s = simulated().load();
    return s >= 0.0 ? static_cast<uint32_t>(s) : juce::Time::getMillisecondCounter();
}

[[nodiscard]] inline auto hiResMs() -> double {
    const auto s = simulated().load();
    return s >= 0.0 ? s : juce::Time::getMillisecondCounterHiRes();
}

} // namespace AppTime
