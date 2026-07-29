#include "VirtualOrch/ui/PresetSaveDialog.h"

PresetSaveDialog::PresetSaveDialog(PresetSaver *parentComponent) : presetSaver(parentComponent) {
    Component::setName("Preset Save Dialog");
    setOpaque(true);
    setSize(400, 150);

    addAndMakeVisible(presetNameLabel);
    presetNameLabel.setText("Preset Name", juce::dontSendNotification);
    presetNameLabel.attachToComponent(&presetName, false);

    addAndMakeVisible(presetName);
    presetName.onReturnKey = [this] { saveButton.triggerClick(); };

    addAndMakeVisible(cancelButton);
    cancelButton.onClick = [this] { userTriedToCloseWindow(); };

    const juce::File presetsDir = juce::File::getSpecialLocation(
                juce::File::SpecialLocationType::userDocumentsDirectory)
            .getChildFile("virtual-orch")
            .getChildFile("Presets");
    juce::Array<juce::File> presetsFiles;
    presetsDir.findChildFiles(presetsFiles, juce::File::findFiles, false, "*.json");

    addAndMakeVisible(saveButton);
    saveButton.onClick = [this, presetsDir] {
        const juce::String presetNameToSave = presetName.getText();
        if (presetNameToSave.isEmpty()) {
            juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                                   "Preset name cannot be empty.");
            return;
        }

        if (presetsDir.getChildFile(presetNameToSave + ".json").existsAsFile()) {
            juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                                   "Preset with this name already exists.");
            return;
        }

        if (presetSaver->savePreset(presetNameToSave)) {
            userTriedToCloseWindow();
        }
    };
}

PresetSaveDialog::~PresetSaveDialog() {
}

void PresetSaveDialog::paint(juce::Graphics &g) {
    g.fillAll(juce::Colours::black);
}

void PresetSaveDialog::resized() {
    auto area = getLocalBounds().reduced(10);

    presetName.setBounds(area.removeFromTop(60).withTrimmedTop(30));

    auto buttonArea = area.removeFromBottom(30);
    cancelButton.setBounds(buttonArea.removeFromRight(100).reduced(2));
    saveButton.setBounds(buttonArea.removeFromRight(100).reduced(2));
}

void PresetSaveDialog::userTriedToCloseWindow() {
    delete this;
}
