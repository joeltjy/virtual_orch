#pragma once

#include <JuceHeader.h>

class PresetSaver {
public:
    virtual bool savePreset(const juce::String &presetFilePath) = 0;
};

class PresetSaveDialog : public juce::Component {
public:
    PresetSaveDialog(PresetSaver *parentComponent);

    ~PresetSaveDialog();

    void paint(juce::Graphics &g) override;

    void resized() override;

    void userTriedToCloseWindow() override;

private:
    PresetSaver *presetSaver;

    juce::TextButton cancelButton{"Cancel"}, saveButton{"Save"};

    juce::Label presetNameLabel;
    juce::TextEditor presetName;
};
