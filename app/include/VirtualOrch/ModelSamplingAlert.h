#pragma once

#include <JuceHeader.h>

#include <cstdint>

/** Thread-safe last model-sampling error for UI (Mode widget). */
class ModelSamplingAlert {
public:
    auto report(const juce::String &source, const juce::String &detail) -> void {
        const juce::ScopedLock lock(mutex);
        message = source + ": " + detail;
        ++sequence;
    }

    [[nodiscard]] auto snapshot() const -> juce::String {
        const juce::ScopedLock lock(mutex);
        return message;
    }

    [[nodiscard]] auto getSequence() const -> uint64_t {
        const juce::ScopedLock lock(mutex);
        return sequence;
    }

    auto clear() -> void {
        const juce::ScopedLock lock(mutex);
        message.clear();
        ++sequence;
    }

private:
    mutable juce::CriticalSection mutex;
    juce::String message;
    uint64_t sequence{0};
};
