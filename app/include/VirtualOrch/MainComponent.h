#pragma once

#include <JuceHeader.h>

#include "Clock.h"
#include "ScrollableWindow.h"
#include "PresetSaveDialog.h"
#include "PresetStore.h"
#include "ModelConfigurationComponent.h"
#include "MusicTransformer.h"
#include "MidiInputProcess.h"
#include "OutputProcessor.h"
#include "OutputPlayback.h"
#include "TransportComponent.h"
#include "Metrics.h"
#include "MetricsComponent.h"
#include "OSCBufferOutputProcessor.h"
#include "OSCStatusOutputProcessor.h"
#include "OSCController.h"
#include "CustomProgressBarLookAndFeel.h"

//==============================================================================
/*
    This component lives inside our window, and this is where you should put all
    your controls and content.
*/
class MainComponent : public ParentComponent {
public:
    //==============================================================================
    MainComponent();

    ~MainComponent() override;

    //==============================================================================
    void paint(juce::Graphics &) override;

    void resized() override;

    void markPresetAsEdited();

    void changeListenerCallback(ChangeBroadcaster *source) override;

private:
    ModelConfig modelConfig;

    PresetStore presetStore;

    Metrics metrics;

    Clock clock;

    int32_t visualizationBufferSize = 400;

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
    juce::ComboBox presetList, mtcClockList, midiInputList, midiInput2List, modelList, outputList;
    juce::TextButton leftPresetButton{"<"}, rightPresetButton{">"}, savePresetButton{"SAVE"};
    juce::ToggleButton autoConnect, mtcClock, inputThru;
    juce::Label presetListLabel, configLabel, mtcClockLabel, mtcClockOffsetLabel, controllerLabel,
            midiInputListLabel, midiInput2ListLabel, modelListLabel,
            inputThruLabel, outputListLabel, oscIpLabel, oscPortLabel, bufferOutputOscIpLabel, bufferOutputOscPortLabel,
            statusOutputOscIpLabel, statusOutputOscPortLabel, generationLabel, visualizationBufferSizeLabel,
            inputDataLabel;

    juce::TextEditor inputDataDisplay;

    juce::TextEditor controllerOscPort, mtcClockOffset, visualizationBufferSizeEditor;
    juce::TextButton controllerConnectButton{"Connect"};

    juce::Component::SafePointer<PresetSaveDialog> presetSaveDialog;

    juce::TextButton modelConfigurationButton{"Model Configuration"};

    bool enableOscConfig = false;
    juce::TextEditor oscIp, oscPort;
    juce::TextButton connectButton{"Connect"};

    juce::TextEditor bufferOutputOscIp, bufferOutputOscPort;
    juce::TextButton bufferOutputConnectButton{"Connect"};

    std::unique_ptr<OSCBufferOutputProcessor> bufferOutputProcessor;

    juce::TextEditor statusOutputOscIp, statusOutputOscPort;
    juce::TextButton statusOutputConnectButton{"Connect"};

    std::unique_ptr<OSCStatusOutputProcessor> statusOutputProcessor;

    int lastInputIndex = 0;
    bool isAddingFromMidiInput = false;

    std::unique_ptr<OSCController> controller;
    std::unique_ptr<OutputProcessor> outputProcessor;

    // FOR DEBUG ONLY
    std::unique_ptr<juce::MidiInput> virtualMidiInput;

    double finalTime = 10000; // 100 seconds // TODO Set this from somewhere else.

    juce::String selectedMidiInputIdentifier, selectedMidiInput2Identifier, selectedMtcClockIdentifier;
    bool mtcClockActive = false;
    int lastMtcClockIndex = 0;

    MusicTransformer musicTransformer;
    OutputPlayback outputPlayback;
    MidiInputProcess midiInputProcess;

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

    void updateInputDataDisplay(const std::vector<int32_t> &data);

    bool keyPressed(const KeyPress &key) override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainComponent)
};
