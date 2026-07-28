#pragma once

#include <chrono>

#include <JuceHeader.h>

#include "Clock.h"
#include "ScrollableWindow.h"
#include "PresetSaveDialog.h"
#include "ModelConfigurationComponent.h"
#include "MusicTransformer.h"
#include "MusicModelList.h"
#include "OutputProcessor.h"
#include "TransportComponent.h"
#include "Metrics.h"
#include "MetricsComponent.h"
#include "OSCBufferOutputProcessor.h"
#include "OSCStatusOutputProcessor.h"
#include "OSCController.h"
#include "CustomProgressBarLookAndFeel.h"
#include "OSCOutputProcessor.h"

//==============================================================================
/*
    This component lives inside our window, and this is where you should put all
    your controls and content.
*/
class MainComponent : public ParentComponent,
                      public PresetSaver,
                      juce::MidiInputCallback,
                      juce::Thread,
                      juce::Timer,
                      private ump::EndpointsListener {
public:
    //==============================================================================
    MainComponent();

    ~MainComponent() override;

    //==============================================================================
    void paint(juce::Graphics &) override;

    void resized() override;

    void run() override;

    void timerCallback() override;

    void markPresetAsEdited();

    void changeListenerCallback(ChangeBroadcaster *source) override;

    bool savePreset(const juce::String &presetFilePath) override;

    void endpointsChanged() override;

private:
    juce::ValueTree settings;

    Clock clock;

    void loadSettings();

    void saveSettings();

    void loadPresetFromName(const juce::String &presetName, const juce::File &presetsDir);

    void loadPreset(const juce::var &parsedJson);

    void moveToLeftPreset();

    void moveToRightPreset();

    void clearModel();

    void updateModel(bool loadDefaultPreset);

    void start();

    void stop();

    juce::Array<juce::MidiDeviceInfo> midiInputs, hardwareMidiOutputs;
    juce::StringArray midiInputNames, midiOutputNames;
    std::vector<juce::String> outputIdentifiers;

    void updateOutputProcessor();

    /**
     Starts listening to a MIDI input device, enabling it if necessary *
     */
    void setMidiInput(int index);
    void setExtraMidiInput(int index);

    void triggerDirectInput();

    void sendTokensToMusicTransformer(std::optional<int32_t> atTime);

    void setManualPause(uint32_t atTime, bool pause);

    void handleIncomingMidiMessage(juce::MidiInput *source, const juce::MidiMessage &message) override;

    void handleContinuousControl(int controllerNumber, int controllerValue);

    void handleNoteOn(int midiNoteNumber, float velocity);

    void handleNoteOff(int midiNoteNumber);

    enum TokenNoteType {
        TokenNoteOn,
        TokenNoteOff
    };

    bool handleNote(Token token, TokenNoteType tokenEventType, uint32_t time);

    MusicModelList musicModelList;

    juce::AudioDeviceManager deviceManager;
    juce::ComboBox presetList, mtcClockList, midiInputList, extraMidiInputList, modelList, outputList;
    juce::TextButton leftPresetButton{"<"}, rightPresetButton{">"}, savePresetButton{"SAVE"};
    juce::ToggleButton autoConnect, mtcClock, inputThru;
    juce::Label presetListLabel, configLabel, mtcClockLabel, mtcClockOffsetLabel, controllerLabel,
            midiInputListLabel, extraMidiInputListLabel, modelListLabel, inputThruLabel, outputListLabel, oscIpLabel,
            oscPortLabel, bufferOutputOsc1IpLabel, bufferOutputOsc1PortLabel, bufferOutputOsc2IpLabel,
            bufferOutputOsc2PortLabel, visualizationBufferSizeLabel, velocityLabel, statusOutputOscIpLabel,
            statusOutputOscPortLabel, generationLabel;

    juce::TextEditor controllerOscPort, mtcClockOffset, visualizationBufferSize, velocity;
    juce::TextButton controllerConnectButton{"Connect"};

    const juce::File presetsDir = juce::File::getSpecialLocation(
                juce::File::SpecialLocationType::userDocumentsDirectory)
            .getChildFile("JordanAI")
            .getChildFile("Presets");

    juce::Component::SafePointer<PresetSaveDialog> presetSaveDialog;

    juce::TextButton modelConfigurationButton{"Model Configuration"};

    bool enableOscConfig = true;
    juce::TextEditor oscIp, oscPort;
    juce::TextButton oscConnectButton{"Connect"};

    juce::TextEditor bufferOutputOsc1Ip, bufferOutputOsc1Port, bufferOutputOsc2Ip, bufferOutputOsc2Port;
    juce::TextButton bufferOutputOsc1ConnectButton{"Connect"}, bufferOutputOsc2ConnectButton{"Connect"};

    std::unique_ptr<OSCBufferOutputProcessor> bufferOutputOsc1Processor;
    std::unique_ptr<OSCBufferOutputProcessor> bufferOutputOsc2Processor;

    juce::TextEditor statusOutputOscIp, statusOutputOscPort;
    juce::TextButton statusOutputConnectButton{"Connect"};

    std::unique_ptr<OSCStatusOutputProcessor> statusOutputProcessor;

    int lastInputIndex = 0;
    bool isAddingFromMidiInput = false;

    std::deque<Token> tokensToSend;
    uint32_t lastTokenAddedTime = 0;

    std::optional<int32_t> bassHeld = std::nullopt;

    int32_t stopBar = -1; // -1 means no stop time set
    int32_t panicBar = -1; // -1 means no panic time set

    std::map<int32_t, uint32_t> notesOnToSend;

    std::unique_ptr<OSCController> controller;
    std::unique_ptr<OutputProcessor> outputProcessor;
    std::unique_ptr<OSCOutputProcessor> oscOutputProcessor;

    // FOR DEBUG ONLY
    // std::unique_ptr<juce::MidiInput> virtualMidiInput;

    double finalTime = 10000; // 100 seconds // TODO Set this from somewhere else.
    double progress = 0.0;

    MusicTransformer musicTransformer;

    juce::TextButton startButton{"Start Generation"}, stopButton{"Stop"}, saveLastGenButton{"Save Last Generation"};

    std::unique_ptr<juce::ProgressBar> generationStatusProgressBar;
    CustomProgressBarLookAndFeel progressBarLookAndFeel;

    juce::Label generationStatusProgressBarLabel;

    int32_t currentBar = 0;

    /* Whether the queue is paused */
    bool queuePaused = false;

    /* Used to trigger prompting upon reaching next bar (in Buffer mode) */
    bool sendOnNextBar = false;

    juce::TextButton openTransport{"Open Transport"}, openMetrics{"Open Metrics"};

    /* Modified by [ModelConfigurationComponent], Consumed by [MusicTransformer] */
    ModelConfig modelConfig;

    /* Written by [Clock], Consumed by [MetricsComponent] */
    Metrics metrics;

    /* Visualization buffer size for SAM HACKATHON */
    int32_t visualizationBufferSizeValue = 400;

    /* Velocity for SAM HACKATHON */
    int32_t velocityValue = 50;

    juce::Component::SafePointer<ScrollableWindow> modelConfigurationWindow;
    juce::Component::SafePointer<TransportComponent> transportWindow;
    juce::Component::SafePointer<MetricsComponent> metricsWindow;

    bool mtcClockActive = false;

    int32_t hours = 0, minutes = 0, seconds = 0, frames = 0;

    juce::String selectedMidiInputIdentifier, selectedExtraMidiInputIdentifier, selectedMtcClockIdentifier;
    int lastMtcClockIndex = 0;

    void updateMtcClock();

    bool keyPressed(const KeyPress &key) override;

    // optional atomic chrono
    juce::Atomic<bool> lastMidiPlayedSet = false;
    juce::Atomic<std::chrono::steady_clock::time_point> lastMidiPlayed;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainComponent)
};
