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

    /**
     * Install the checked-in app/presets/*.json into Presets/ and keep them current. What was
     * installed is recorded in Presets/.bundled/: an installed preset still equal to its record
     * (not edited here) follows the checked-in one when that changes; an edited one is left. A
     * preset installed before the records were kept and different from the checked-in one is
     * updated, the old file kept as Presets/.bundled/<name>.<stamp>.bak.
     */
    auto installBundledPresets() -> void;
    /** installBundledPresets from `bundledDir` into `presetsDir` (tests use temporary folders). */
    static auto installBundledPresets(const juce::File &bundledDir, const juce::File &presetsDir) -> void;

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

    /** Returns "edit" or "jam". */
    auto setOrchestrationModeProvider(std::function<juce::String()> provider) -> void {
        orchestrationModeProvider = std::move(provider);
    }

    /** Returns "amt", "v1", or "v2" for the Reduction dropdown. */
    auto setReductionTypeProvider(std::function<juce::String()> provider) -> void {
        reductionTypeProvider = std::move(provider);
    }

    /** The "logitAdjustments" object written into saved presets (AppSession::logitAdjustmentsToVar). */
    auto setLogitAdjustmentsProvider(std::function<juce::var()> provider) -> void {
        logitAdjustmentsProvider = std::move(provider);
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
    std::function<juce::String()> orchestrationModeProvider;
    std::function<juce::String()> reductionTypeProvider;
    std::function<juce::var()> logitAdjustmentsProvider;
    std::function<void(const juce::String &)> onPresetSaved;

    [[nodiscard]] auto buildPresetJson(const juce::String &modelName) const -> juce::var;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PresetStore)
};
