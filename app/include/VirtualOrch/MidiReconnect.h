#pragma once

#include <JuceHeader.h>

/**
 * A USB MIDI device that drops off the bus and comes back keeps its identifier (on Linux ALSA,
 * e.g. "20-0"), but the MidiInput the app opened before the drop is dead: no message arrives
 * until it is closed and opened again. The app reopens every input it uses whose identifier was
 * missing at the previous device-list check and is present now.
 */
struct MidiReconnect {
    /** Identifiers in `inUse` that are in `now` but were not in `previous`. */
    [[nodiscard]] static auto reappeared(const juce::StringArray &previous, const juce::StringArray &now,
                                         const juce::StringArray &inUse) -> juce::StringArray {
        juce::StringArray result;
        for (const auto &id: inUse)
            if (id.isNotEmpty() && now.contains(id) && ! previous.contains(id))
                result.addIfNotAlreadyThere(id);
        return result;
    }

    /** Identifiers in `inUse` that were in `previous` and are not in `now`. */
    [[nodiscard]] static auto disappeared(const juce::StringArray &previous, const juce::StringArray &now,
                                          const juce::StringArray &inUse) -> juce::StringArray {
        juce::StringArray result;
        for (const auto &id: inUse)
            if (id.isNotEmpty() && previous.contains(id) && ! now.contains(id))
                result.addIfNotAlreadyThere(id);
        return result;
    }

    [[nodiscard]] static auto identifiers(const juce::Array<juce::MidiDeviceInfo> &devices) -> juce::StringArray {
        juce::StringArray ids;
        for (const auto &d: devices)
            ids.add(d.identifier);
        return ids;
    }
};
