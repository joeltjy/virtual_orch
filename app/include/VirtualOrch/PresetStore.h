#pragma once

#include <JuceHeader.h>

#include <functional>

#include "VirtualOrch/ui/ModelConfigurationComponent.h"
#include "VirtualOrch/ui/PresetSaveDialog.h"

/**
 * Persists app settings (settings.xml) and model presets (JSON under Presets/).
 */
class PresetStore : public PresetSaver {
public:
    explicit PresetStore(ModelConfig &modelConfig);

    juce::ValueTree settings{"settings"};

    auto loadSettings() -> void;

    auto saveSettings() -> void;

    /** Copy checked-in app/presets/*.json into Documents if missing. */
    auto installBundledPresets() -> void;

    [[nodiscard]] auto getPresetsDir() const -> const juce::File & { return presetsDir; }

    [[nodiscard]] auto listPresetNames() const -> juce::StringArray;

    [[nodiscard]] auto readPreset(const juce::String &presetName) const -> juce::var;

    auto applyPreset(const juce::var &parsedJson) -> void;

    auto savePreset(const juce::String &presetName) -> bool override;

    auto setModelNameProvider(std::function<juce::String()> provider) -> void {
        modelNameProvider = std::move(provider);
    }

    auto setOrchestrationModelNameProvider(std::function<juce::String()> provider) -> void {
        orchestrationModelNameProvider = std::move(provider);
    }

    /** Returns "amt", "v1", or "v2" for the Reduction dropdown. */
    auto setReductionTypeProvider(std::function<juce::String()> provider) -> void {
        reductionTypeProvider = std::move(provider);
    }

    auto setOnPresetSaved(std::function<void(const juce::String &)> callback) -> void {
        onPresetSaved = std::move(callback);
    }

private:
    ModelConfig &modelConfig;
    juce::File appDataDir;
    juce::File presetsDir;
    juce::File settingsFile;

    std::function<juce::String()> modelNameProvider;
    std::function<juce::String()> orchestrationModelNameProvider;
    std::function<juce::String()> reductionTypeProvider;
    std::function<void(const juce::String &)> onPresetSaved;

    [[nodiscard]] auto buildPresetJson(const juce::String &modelName) const -> juce::var;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PresetStore)
};
