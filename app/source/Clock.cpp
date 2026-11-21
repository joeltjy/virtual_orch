#include "VirtualOrch/Clock.h"

#include "VirtualOrch/AppTime.h"
#include "VirtualOrch/SessionLog.h"

#include <iostream>

Clock::Clock(Metrics &metrics) : metrics(metrics) {
    const uint32_t clockTime = AppTime::hiResMs();
    offset.set(clockTime);
}

Clock::~Clock() {
    stopTimer();
}


auto Clock::start() -> void {
    isOn.set(true);
}

auto Clock::startAtTime(uint32_t newTime) -> void {
    setTime(newTime);
    isOn.set(true);
}

auto Clock::startFreeRunning() -> void {
    /* Blocks until any in-flight watchdog callback finishes, so it cannot stop us after. */
    stopTimer();
    startAtTime(getTime());
}

auto Clock::stop() -> void {
    savedLastTime.set(getTime());
    isOn.set(false);
}

auto Clock::setTime(const uint32_t newTime) -> void {
    const uint32_t clockTime = AppTime::hiResMs() * 0.1;
    offset.set(clockTime - newTime);
}

auto Clock::setMtcTime(uint32_t newTime) -> void {
    const uint32_t clockTime = AppTime::hiResMs() * 0.1;

    /* P Loop to adjust the offset */
    int32_t updatedOffset = offset.get() * (1 - P_GAIN) + P_GAIN * (clockTime - newTime);
    if (std::abs(static_cast<int32_t>(clockTime - newTime) - offset.get()) > P_THRESHOLD) {
        updatedOffset = (clockTime - newTime);
    }

    /* METRICS LOGGING */
    metrics.Clock_offsetError.push(updatedOffset - offset.get());

    offset.set(updatedOffset);
    lastClockTimeOfMtcMessage.set(clockTime);

    isOn.set(true);

    /* Start the timer to stop the clock if no message is received */
    startTimer(110);
}

void Clock::hiResTimerCallback() {
    const uint32_t clockTime = AppTime::hiResMs() * 0.1;
    if (lastClockTimeOfMtcMessage.get() <= (clockTime - 10)) {
        const bool wasRunning = isOn.get();
        stop();
        if (wasRunning) {
            // Without this the transport just freezes with no hint why (e.g. MTC Clock ticked
            // but no DAW sending MTC on the selected port).
            const auto text = "MTC watchdog: no MTC quarter frame for 100 ms, clock stopped at "
                              + juce::String(getTime())
                              + " cs. Start MTC from the DAW, or untick MTC Clock to run freely.";
            SessionLog::add("CLOCK", text);
            std::cerr << "[clock] " << text << std::endl;
        }
    }
    stopTimer();
}


auto Clock::getTime() const -> uint32_t {
    if (!isOn.get()) {
        return savedLastTime.get();
    }
    const uint32_t clockTime = AppTime::hiResMs() * 0.1;
    return clockTime - offset.get() + mtcOffset.get();
}

auto Clock::setMtcOffset(int32_t newMtcOffset) -> void {
    mtcOffset.set(newMtcOffset);
}

auto Clock::isRunning() const -> bool {
    return isOn.get();
}
