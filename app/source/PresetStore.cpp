#include "VirtualOrch/PresetStore.h"

PresetStore::PresetStore(ModelConfig &modelConfig)
    : modelConfig(modelConfig),
      appDataDir(juce::File::getSpecialLocation(juce::File::SpecialLocationType::userDocumentsDirectory)
                     .getChildFile("virtual-orch")),
      presetsDir(appDataDir.getChildFile("Presets")),
      settingsFile(appDataDir.getChildFile("settings.xml")) {
    presetsDir.createDirectory();
}

auto PresetStore::loadSettings() -> void {
    if (!settingsFile.exists()) {
        settingsFile.create();
        settings = juce::ValueTree("settings");
    } else {
        std::unique_ptr<juce::XmlElement> xmlSettings = juce::XmlDocument(settingsFile).getDocumentElement();
        if (xmlSettings != nullptr) {
            settings = juce::ValueTree::fromXml(*xmlSettings);
        } else {
            settings = juce::ValueTree("settings");
        }
    }
}

auto PresetStore::saveSettings() -> void {
    std::unique_ptr<juce::XmlElement> xmlSettings = settings.createXml();
    if (xmlSettings != nullptr) {
        xmlSettings->writeTo(settingsFile);
    }
}

auto PresetStore::listPresetNames() const -> juce::StringArray {
    juce::StringArray names;
    juce::Array<juce::File> presetsFiles;
    presetsDir.findChildFiles(presetsFiles, juce::File::findFiles, false, "*.json");
    for (const auto &file: presetsFiles) {
        names.add(file.getFileNameWithoutExtension());
    }
    return names;
}

auto PresetStore::readPreset(const juce::String &presetName) const -> juce::var {
    const juce::File presetFile(presetsDir.getChildFile(presetName + ".json"));
    if (!presetFile.exists()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "Preset file does not exist.");
        return {};
    }

    const juce::var preset = juce::JSON::parse(presetFile);
    if (preset.isVoid()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "Could not parse preset file.");
        return {};
    }
    return preset;
}

auto PresetStore::applyPreset(const juce::var &parsedJson) -> void {
    modelConfig.loadFromPreset(parsedJson);
}

auto PresetStore::buildPresetJson(const juce::String &modelName) const -> juce::var {
    juce::var presetJson(new juce::DynamicObject());
    presetJson.getDynamicObject()->setProperty("model", modelName);

    switch (modelConfig.inputMode) {
        case InputMode::Direct:
            presetJson.getDynamicObject()->setProperty("inputMode", "direct");
            presetJson.getDynamicObject()->setProperty("directInputWindowLength", modelConfig.directInputWindowLength);
            presetJson.getDynamicObject()->setProperty("directInputSendNoteOffs",
                                                       modelConfig.directInputSendNoteOffs);
            presetJson.getDynamicObject()->setProperty("directInputStartOnInput",
                                                       modelConfig.directInputStartOnInput);
            presetJson.getDynamicObject()->setProperty("directInputStartDelay",
                                                       modelConfig.directInputStartDelay);
            presetJson.getDynamicObject()->setProperty("directInputHoldBass",
                                                       modelConfig.directInputHoldBass);
            presetJson.getDynamicObject()->setProperty("directInputBassLow", modelConfig.directInputBassLow);
            presetJson.getDynamicObject()->setProperty("directInputBassHigh", modelConfig.directInputBassHigh);
            presetJson.getDynamicObject()->setProperty("directInputInitialBass", modelConfig.directInputInitialBass);
            break;
        case InputMode::Buffer:
            presetJson.getDynamicObject()->setProperty("inputMode", "buffer");
            presetJson.getDynamicObject()->setProperty("bufferInputSize", modelConfig.bufferInputSize);
            presetJson.getDynamicObject()->setProperty("bufferInputLow", modelConfig.bufferInputLow);
            presetJson.getDynamicObject()->setProperty("bufferInputHigh", modelConfig.bufferInputHigh);
            presetJson.getDynamicObject()->setProperty("bufferInputBassLow", modelConfig.bufferInputBassLow);
            presetJson.getDynamicObject()->setProperty("bufferInputBassHigh", modelConfig.bufferInputBassHigh);
            break;
    }

    presetJson.getDynamicObject()->setProperty("inputInstrument", modelConfig.inputInstrument);
    presetJson.getDynamicObject()->setProperty("inputLow", modelConfig.inputLow);
    presetJson.getDynamicObject()->setProperty("inputHigh", modelConfig.inputHigh);
    presetJson.getDynamicObject()->setProperty("inputDuration", modelConfig.inputDuration);
    presetJson.getDynamicObject()->setProperty("inputInitialData", modelConfig.inputInitialData);

    presetJson.getDynamicObject()->setProperty("outputMinimumDuration", modelConfig.outputMinimumDuration);
    presetJson.getDynamicObject()->setProperty("outputMaximumDuration", modelConfig.outputMaximumDuration);
    juce::var outputTemperatures;
    outputTemperatures.append(modelConfig.outputTemperatures[0]);
    outputTemperatures.append(modelConfig.outputTemperatures[1]);
    outputTemperatures.append(modelConfig.outputTemperatures[2]);
    presetJson.getDynamicObject()->setProperty("outputTemperatures", outputTemperatures);
    presetJson.getDynamicObject()->setProperty("outputStartTime", modelConfig.outputStartTime);
    presetJson.getDynamicObject()->setProperty("outputForceStartTime", modelConfig.outputForceStartTime);

    juce::var outputInstruments;
    for (const auto &[id, instrument]: modelConfig.outputInstruments) {
        juce::var outputInstrument(new juce::DynamicObject());
        outputInstrument.getDynamicObject()->setProperty("id", id);
        outputInstrument.getDynamicObject()->setProperty("active", instrument.active);
        outputInstrument.getDynamicObject()->setProperty("monophony", instrument.monophony);
        outputInstrument.getDynamicObject()->setProperty("low", instrument.low);
        outputInstrument.getDynamicObject()->setProperty("high", instrument.high);
        outputInstruments.append(outputInstrument);
    }
    presetJson.getDynamicObject()->setProperty("outputInstruments", outputInstruments);

    return presetJson;
}

auto PresetStore::savePreset(const juce::String &presetName) -> bool {
    if (!modelNameProvider) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "Cannot save a preset without a model name provider.");
        return false;
    }

    const juce::String modelName = modelNameProvider();
    if (modelName.isEmpty()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "Cannot save a preset without a model.");
        return false;
    }

    const juce::var presetJson = buildPresetJson(modelName);

    juce::File presetFile = presetsDir.getChildFile(presetName + ".json");
    juce::FileOutputStream output(presetFile);
    if (output.openedOk()) {
        output.setPosition(0);
        output.truncate();
    } else {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "Could not open file for writing.");
        return false;
    }
    juce::JSON::writeToStream(output, presetJson);

    if (onPresetSaved) {
        onPresetSaved(presetName);
    }
    return true;
}
