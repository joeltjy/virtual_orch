#pragma once

#include <JuceHeader.h>

#include "Metrics.h"

class Clock : juce::HighResolutionTimer {
public:
    Clock(Metrics &metrics);

    ~Clock() override;

    auto start() -> void;

    auto startAtTime(uint32_t newTime) -> void;

    /**
     * Leave MTC slaving: cancel the stall watchdog and run freely from the current
     * time, so a clock parked by a missing MTC feed resumes without a restart.
     */
    auto startFreeRunning() -> void;

    auto stop() -> void;

    auto setTime(uint32_t newTime) -> void;

    /**
     * Used to set the MTC Clock and automatically stop the clock
     * if no message is received.
     * @param newTime the time to set the clock to.
     */
    auto setMtcTime(uint32_t newTime) -> void;

    [[nodiscard]] auto getTime() const -> uint32_t;

    void hiResTimerCallback() override;

    auto setMtcOffset(int32_t newMtcOffset) -> void;

    auto isRunning() const -> bool;

private:
    juce::Atomic<bool> isOn = false;

    juce::Atomic<int32_t> offset;
    juce::Atomic<uint32_t> savedLastTime, lastClockTimeOfMtcMessage;

    Metrics &metrics;

    float P_GAIN = .5F;
    uint32_t P_THRESHOLD = 5;

    juce::Atomic<int32_t> mtcOffset;
};
