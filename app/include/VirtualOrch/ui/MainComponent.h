#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/ScrollableWindow.h"
#include "VirtualOrch/ui/PresetSaveDialog.h"
#include "VirtualOrch/ui/ModelConfigurationComponent.h"
#include "VirtualOrch/ui/TransportComponent.h"
#include "VirtualOrch/ui/MetricsComponent.h"
#include "VirtualOrch/OSCStatusOutputProcessor.h"
#include "VirtualOrch/OSCController.h"
#include "VirtualOrch/ui/CustomProgressBarLookAndFeel.h"

//==============================================================================
/*
    Settings page UI. Engines live in AppSession (owned by AppRootComponent).
*/
class MainComponent : public ParentComponent {
public:
    //==============================================================================
    explicit MainComponent(AppSession &session);

    ~MainComponent() override;

    //==============================================================================
    void paint(juce::Graphics &) override;

    void resized() override;

    void markPresetAsEdited();

    void changeListenerCallback(ChangeBroadcaster *source) override;

private:
    AppSession &session;

    void loadPresetFromName(const juce::String &presetName);

    void moveToLeftPreset();

    void moveToRightPreset();

    void clearModel();

    void updateModel(bool loadDefaultPreset);

    void start();

    void stop();

    juce::Array<juce::MidiDeviceInfo> midiInputs, hardwareMidiOutputs;

    void updateOutputProcessor();

    /**
     Starts listening to a MIDI input device, enabling it if necessary *
     */
    void setMidiInput(int index, int idx);

    juce::AudioDeviceManager deviceManager;
    juce::ComboBox presetList, mtcClockList, midiInputList, launchpadMidiList, modelList, outputList;
    juce::TextButton leftPresetButton{"<"}, rightPresetButton{">"}, savePresetButton{"SAVE"};
    juce::ToggleButton autoConnect, mtcClock, inputThru;
    juce::Label presetListLabel, configLabel, mtcClockLabel, mtcClockOffsetLabel, controllerLabel,
            midiInputListLabel, launchpadMidiListLabel, modelListLabel,
            inputThruLabel, outputListLabel, oscIpLabel, oscPortLabel, bufferOutputOscIpLabel, bufferOutputOscPortLabel,
            statusOutputOscIpLabel, statusOutputOscPortLabel, generationLabel, visualizationBufferSizeLabel;

    juce::TextEditor controllerOscPort, mtcClockOffset, visualizationBufferSizeEditor;
    juce::TextButton controllerConnectButton{"Connect"};

    juce::Component::SafePointer<PresetSaveDialog> presetSaveDialog;

    juce::TextButton modelConfigurationButton{"Model Configuration"};

    bool enableOscConfig = false;
    juce::TextEditor oscIp, oscPort;
    juce::TextButton connectButton{"Connect"};

    juce::TextEditor bufferOutputOscIp, bufferOutputOscPort;
    juce::TextButton bufferOutputConnectButton{"Connect"};

    juce::TextEditor statusOutputOscIp, statusOutputOscPort;
    juce::TextButton statusOutputConnectButton{"Connect"};

    std::unique_ptr<OSCStatusOutputProcessor> statusOutputProcessor;

    int lastInputIndex = 0;
    bool isAddingFromMidiInput = false;

    std::unique_ptr<OSCController> controller;

    // FOR DEBUG ONLY
    std::unique_ptr<juce::MidiInput> virtualMidiInput;

    double finalTime = 10000; // 100 seconds // TODO Set this from somewhere else.

    int lastMtcClockIndex = 0;

    juce::TextButton startButton{"Start Generation"}, stopButton{"Stop"}, saveLastGenButton{"Save Last Generation"};

    std::unique_ptr<juce::ProgressBar> generationStatusProgressBar;
    CustomProgressBarLookAndFeel progressBarLookAndFeel;

    juce::Label generationStatusProgressBarLabel;

    juce::TextButton openTransport{"Open Transport"},
            openMetrics{"Open Metrics"};

    juce::Component::SafePointer<ScrollableWindow> modelConfigurationWindow;
    juce::Component::SafePointer<TransportComponent> transportWindow;
    juce::Component::SafePointer<MetricsComponent> metricsWindow;

    void updateMtcClock();

    bool keyPressed(const KeyPress &key) override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainComponent)
};
