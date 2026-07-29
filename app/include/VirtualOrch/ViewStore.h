#pragma once

#include <JuceHeader.h>

/**
 * Persists workspace layouts as JSON under Documents/virtual-orch/Views/.
 */
class ViewStore {
public:
    ViewStore();

    [[nodiscard]] auto getViewsDir() const -> const juce::File & { return viewsDir; }

    [[nodiscard]] auto listViewNames() const -> juce::StringArray;

    /** Writes view JSON. Returns false on I/O failure. */
    auto saveView(const juce::String &viewName, const juce::var &viewJson) const -> bool;

    /** Parses view JSON. Returns void var if missing or invalid. */
    [[nodiscard]] auto loadView(const juce::String &viewName) const -> juce::var;

    [[nodiscard]] auto viewExists(const juce::String &viewName) const -> bool;

private:
    [[nodiscard]] auto viewFile(const juce::String &viewName) const -> juce::File;

    juce::File viewsDir;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ViewStore)
};
