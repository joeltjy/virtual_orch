#include "JordanAI/Clock.h"

Clock::Clock(const ModelConfig &modelConfig, Metrics &metrics) : modelConfig(modelConfig), metrics(metrics) {
    const uint32_t clockTime = juce::Time::getMillisecondCounterHiRes();
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

auto Clock::stop() -> void {
    savedLastTime.set(getTime());
    isOn.set(false);
}

auto Clock::setTime(const uint32_t newTime) -> void {
    const uint32_t clockTime = juce::Time::getMillisecondCounterHiRes() * 0.1;
    offset.set(clockTime - newTime);
}

auto Clock::setMtcTime(uint32_t newTime) -> void {
    const uint32_t clockTime = juce::Time::getMillisecondCounterHiRes() * 0.1;

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
    const uint32_t clockTime = juce::Time::getMillisecondCounterHiRes() * 0.1;
    if (lastClockTimeOfMtcMessage.get() <= (clockTime - 10)) {
        stop();
    }
    stopTimer();
}


auto Clock::getTime() const -> uint32_t {
    if (!isOn.get()) {
        return savedLastTime.get();
    }
    const uint32_t clockTime = juce::Time::getMillisecondCounterHiRes() * 0.1;
    return clockTime - offset.get() + mtcOffset.get();
}

auto Clock::getCurrentBar() const -> uint32_t {
    return getTime() / modelConfig.outputBarLength;
}

auto Clock::getTimeAndBar() const -> std::tuple<uint32_t, uint32_t> {
    const uint32_t time = getTime();
    return {time, time / modelConfig.outputBarLength};
}

auto Clock::setMtcOffset(int32_t newMtcOffset) -> void {
    mtcOffset.set(newMtcOffset);
}

auto Clock::isRunning() const -> bool {
    return isOn.get();
}

