#include "VirtualOrch/MainComponent.h"

#include <chrono>

#include "VirtualOrch/MidiOutputProcessor.h"
#include "VirtualOrch/OSCOutputProcessor.h"
#include "VirtualOrch/OSCController.h"

//==============================================================================
MainComponent::MainComponent() : musicTransformer(modelConfig), clock(modelConfig, metrics),
                                 Thread("Main Thread") {
    setOpaque(true);

    setWantsKeyboardFocus(true);

    loadSettings();

    // Get displays
    // This is used to show the transport window on an external window when available
    auto const &displays = Desktop::getInstance().getDisplays().displays;

    ump::Endpoints::getInstance()->addListener (*this);
    endpointsChanged();

    // FOR DEBUG ONLY
    // virtualMidiInput = juce::MidiInput::createNewDevice("virtual-orch Virtual MIDI Input", this);
    // virtualMidiInput->start();

    /* PRESET LIST */
    addAndMakeVisible(presetListLabel);
    presetListLabel.setText("Preset: ", juce::dontSendNotification);
    presetListLabel.attachToComponent(&presetList, true);

    addAndMakeVisible(presetList);
    presetList.addItem("Empty", 1);
    juce::Array<juce::File> presetsFiles;
    presetsDir.findChildFiles(presetsFiles, juce::File::findFiles, false, "*.json");
    juce::File::NaturalFileComparator sortNatural(false);
    presetsFiles.sort(sortNatural);
    for (const auto &file: presetsFiles) {
        presetList.addItem(file.getFileNameWithoutExtension(), presetsFiles.indexOf(file) + 2);
    }
    presetList.onChange = [this] {
        if (presetList.getSelectedItemIndex() == 0) {
            clearModel();
        } else {
            // Retrieve model
            juce::String presetName = presetList.getItemText(presetList.getSelectedItemIndex());
            loadPresetFromName(presetName, presetsDir);
        }
        savePresetButton.setEnabled(false);
    };
    presetList.setSelectedItemIndex(0);

    /* PRESET BUTTONS */
    addAndMakeVisible(leftPresetButton);
    leftPresetButton.onClick = [this] { moveToLeftPreset(); };

    addAndMakeVisible(rightPresetButton);
    rightPresetButton.onClick = [this] { moveToRightPreset(); };

    addAndMakeVisible(savePresetButton);
    savePresetButton.onClick = [this] {
        if (presetSaveDialog) {
            presetSaveDialog->toFront(true);
        } else {
            presetSaveDialog = new PresetSaveDialog(this);
            presetSaveDialog->addToDesktop(juce::ComponentPeer::windowIsTemporary);
            presetSaveDialog->centreWithSize(400, 150);
            presetSaveDialog->setVisible(true);
            presetSaveDialog->toFront(true);
        }
    };

    /**********/
    /* CONFIG */
    /**********/

    addAndMakeVisible(configLabel);
    configLabel.setText("Configuration", juce::dontSendNotification);
    configLabel.setFont(juce::Font(20.0f));

    /* PRESET CONTROLLER */
    addAndMakeVisible(controllerLabel);
    controllerLabel.setText("Controller: ", juce::dontSendNotification);
    controllerLabel.attachToComponent(&controllerOscPort, true);

    addAndMakeVisible(controllerOscPort);
    juce::String savedControllerOscPort = settings.getOrCreateChildWithName("controller", nullptr).
            getProperty("oscPort", "9001");
    controllerOscPort.setText(savedControllerOscPort);
    controllerOscPort.setInputRestrictions(6, "0123456789");
    controllerOscPort.onTextChange = [this] {
        controllerConnectButton.setEnabled(true);
        DBG("Saving Preset Controller OSC Port: " + controllerOscPort.getText());
        settings.getChildWithName("controller").
                setProperty("oscPort", controllerOscPort.getText(), nullptr);
        saveSettings();
    };

    addAndMakeVisible(controllerConnectButton);
    controllerConnectButton.onClick = [this] {
        /* Set controller */
        controller = std::make_unique<OSCController>(controllerOscPort.getText().getIntValue());
        controller->onPresetChanged = [this](const juce::String &presetName) {
            loadPresetFromName(presetName, presetsDir);
        };
        controller->onStart = [this] {
            start();
        };
        controller->onStop = [this] {
            stop();
        };
        controller->onOpenTransport = [this] {
            openTransport.triggerClick();
        };
        controller->onCueStopTrading = [this] {
            auto currentBar = clock.getCurrentBar();
            stopBar = currentBar + 1;
            DBG("Cue Stop Trading at bar " + std::to_string(stopBar));
        };
        controller->onSetOutputRange = [this](const juce::int32 &id, const juce::int32 &low, const juce::int32 &high) {
            modelConfig.outputInstruments[id].low = low;
            modelConfig.outputInstruments[id].high = high;
        };
        controller->onSetModelConfig = [this](const juce::OSCMessage &message) {
            if (message.size() == 2) {
                if (message[1].getType() == juce::OSCTypes::string) {
                    modelConfig.updateParameter(message[0].getString(), message[1].getString());
                } else if (message[1].getType() == juce::OSCTypes::int32) {
                    modelConfig.updateParameter(message[0].getString(), message[1].getInt32());
                } else {
                    throw std::runtime_error("Invalid OSC type " +
                                             std::to_string(message[1].getType()) +
                                             " for setting model parameter " +
                                             message[0].getString().toStdString());
                }
            } else if (message.size() == 3) {
                if (message[1].getType() != juce::OSCTypes::int32 || message[2].getType() != juce::OSCTypes::int32) {
                    throw std::runtime_error("Invalid OSC types " +
                                             std::to_string(message[1].getType()) + ", " +
                                             std::to_string(message[2].getType()) +
                                             " for setting model parameter " +
                                             message[0].getString().toStdString() +
                                             " with instrument id");
                }
                modelConfig.updateParameter(message[0].getString(), message[1].getInt32(),
                                            message[2].getInt32());
            }
        };
        controller->onSetManualPause = [this](bool value) {
            if (modelConfig.inputMode == InputMode::Direct && modelConfig.directInputManualPause &&
                modelConfig.directInputManualPauseControlType == ControlType::OscController) {
                uint32_t time = clock.getTime();
                setManualPause(time, value);
            }
        };
        controller->onSetVelocity = [this](const juce::int32 &receivedVelocity) {
            if (modelConfig.inputMode == InputMode::Direct) {
                velocity.setText(std::to_string(receivedVelocity), juce::sendNotification);
            }
        };

        controllerConnectButton.setEnabled(false);
    };

    /* MTC CLOCK */
    addAndMakeVisible(mtcClockLabel);
    mtcClockLabel.setText("MTC Clock: ", juce::dontSendNotification);
    mtcClockLabel.attachToComponent(&mtcClock, true);

    addAndMakeVisible(mtcClock);
    mtcClock.onClick = [this] {
        updateMtcClock();
        DBG("Saving MTC Clock Status: " + std::to_string(mtcClock.getToggleState()));
        settings.getChildWithName("mtcClock").setProperty(
            "active", mtcClock.getToggleState(), nullptr);
        saveSettings();
    };
    bool savedMtcClockStatus = settings.getOrCreateChildWithName("mtcClock", nullptr).getProperty("active", false);
    mtcClock.setToggleState(savedMtcClockStatus, juce::sendNotification);

    addAndMakeVisible(mtcClockList);
    mtcClockList.setTextWhenNoChoicesAvailable("No MTC Clocks Enabled");
    mtcClockList.onChange = [this] {
        updateMtcClock();
        DBG("Saving MTC Clock: " + midiInputs[mtcClockList.getSelectedItemIndex()].identifier);
        settings.getChildWithName("mtcClock").setProperty(
            "identifier", midiInputs[mtcClockList.getSelectedItemIndex()].identifier, nullptr);
        saveSettings();
    };

    addAndMakeVisible(mtcClockOffsetLabel);
    mtcClockOffsetLabel.setText("Offset: ", juce::dontSendNotification);
    mtcClockOffsetLabel.attachToComponent(&mtcClockOffset, true);

    addAndMakeVisible(mtcClockOffset);
    juce::String savedMtcClockOffset = settings.getOrCreateChildWithName("mtcClock", nullptr).
            getProperty("offset", "0");
    mtcClockOffset.setInputRestrictions(6, "0123456789");
    mtcClockOffset.onTextChange = [this] {
        updateMtcClock();
        DBG("Saving MTC Clock Offset: " + mtcClockOffset.getText());
        settings.getChildWithName("mtcClock").setProperty(
            "offset", mtcClockOffset.getText(), nullptr);
        saveSettings();
    };
    mtcClockOffset.setText(savedMtcClockOffset);

    /* MIDI INPUT LIST */

    addAndMakeVisible(midiInputListLabel);
    midiInputListLabel.setText("MIDI Input: ", juce::dontSendNotification);
    midiInputListLabel.attachToComponent(&midiInputList, true);

    addAndMakeVisible(midiInputList);
    midiInputList.setTextWhenNoChoicesAvailable("No MIDI Inputs Enabled");

    midiInputList.onChange = [this] {
        if (midiInputList.getSelectedItemIndex() == 0) {
            return;
        }
        setMidiInput(midiInputList.getSelectedItemIndex());
        DBG("Saving input: " + midiInputs[midiInputList.getSelectedItemIndex()].identifier);
        settings.getChildWithName("input").setProperty(
            "identifier", midiInputs[midiInputList.getSelectedItemIndex()].identifier, nullptr);
        saveSettings();
    };

    /* EXTRA MIDI INPUT LIST */

    addAndMakeVisible(extraMidiInputListLabel);
    extraMidiInputListLabel.setText("Extra MIDI Input: ", juce::dontSendNotification);
    extraMidiInputListLabel.attachToComponent(&extraMidiInputList, true);

    addAndMakeVisible(extraMidiInputList);
    extraMidiInputList.setTextWhenNoChoicesAvailable("No MIDI Inputs Enabled");

    extraMidiInputList.onChange = [this] {
        if (extraMidiInputList.getSelectedItemIndex() == 0) {
            return;
        }
        setExtraMidiInput(extraMidiInputList.getSelectedItemIndex());
        DBG("Saving Extra MIDI input: " + midiInputs[extraMidiInputList.getSelectedItemIndex()].identifier);
        settings.getChildWithName("extraMidiInput").setProperty(
            "identifier", midiInputs[extraMidiInputList.getSelectedItemIndex()].identifier, nullptr);
        saveSettings();
    };

    /* MODEL LIST */

    addAndMakeVisible(modelListLabel);
    modelListLabel.setText("Model: ", juce::dontSendNotification);
    modelListLabel.attachToComponent(&modelList, true);

    addAndMakeVisible(modelList);
    modelList.setTextWhenNoChoicesAvailable("No Models Available");

    musicModelList.buildList();
    modelList.addItemList(musicModelList.getListComboBox(), 1);

    modelList.onChange = [this] {
        stop();
        updateModel(true);
    };

    addAndMakeVisible(modelConfigurationButton);
    modelConfigurationButton.onClick = [this] {
        if (modelConfigurationWindow) {
            modelConfigurationWindow->toFront(true);
        } else {
            modelConfigurationWindow = new ScrollableWindow(new ModelConfigurationComponent(modelConfig), this, 700,
                                                            700);
            modelConfigurationWindow->toFront(true);
            modelConfigurationWindow->addChangeListener(this);
        }
    };

    /* MIDI THROUGH */
    addAndMakeVisible(inputThruLabel);
    inputThruLabel.setText("Input Thru:", juce::dontSendNotification);
    inputThruLabel.attachToComponent(&inputThru, true);

    addAndMakeVisible(inputThru);
    bool savedInputThru = settings.getOrCreateChildWithName("input", nullptr).getProperty("thru", false);
    inputThru.setToggleState(savedInputThru, juce::dontSendNotification);
    inputThru.onClick = [this] {
        DBG("Saving input thru: " + std::to_string(inputThru.getToggleState()));
        settings.getChildWithName("input").setProperty("thru", inputThru.getToggleState(), nullptr);
        saveSettings();
    };


    /* OUTPUT LIST */

    addAndMakeVisible(outputListLabel);
    outputListLabel.setText("Output: ", juce::dontSendNotification);
    outputListLabel.attachToComponent(&outputList, true);

    addAndMakeVisible(outputList);

    outputList.onChange = [this] {
        updateOutputProcessor();
        DBG("Saving output: " + outputIdentifiers.at(outputList.getSelectedItemIndex()));
        settings.getChildWithName("output").setProperty(
            "identifier", outputIdentifiers.at(outputList.getSelectedItemIndex()), nullptr);
        saveSettings();
    };

    /* OSC IP */

    addAndMakeVisible(oscIpLabel);
    oscIpLabel.setText("OSC IP: ", juce::dontSendNotification);
    oscIpLabel.attachToComponent(&oscIp, true);

    addAndMakeVisible(oscIp);
    juce::String savedOscIp = settings.getOrCreateChildWithName("output", nullptr).getProperty("oscIp", "127.0.0.1");
    oscIp.setText(savedOscIp);
    oscIp.setInputRestrictions(15, "0123456789.");

    oscIp.onTextChange = [this] {
        oscConnectButton.setEnabled(true);
        DBG("Saving OSC IP: " + oscIp.getText());
        settings.getChildWithName("output").setProperty("oscIp", oscIp.getText(), nullptr);
        saveSettings();
    };

    addAndMakeVisible(oscPortLabel);
    oscPortLabel.setText(": ", juce::dontSendNotification);
    oscPortLabel.attachToComponent(&oscPort, true);

    addAndMakeVisible(oscPort);
    juce::String savedOscPort = settings.getOrCreateChildWithName("output", nullptr).getProperty("oscPort", "9001");
    oscPort.setText(savedOscPort);
    oscPort.setInputRestrictions(6, "0123456789");

    oscPort.onTextChange = [this] {
        oscConnectButton.setEnabled(true);
        DBG("Saving OSC Port: " + oscPort.getText());
        settings.getChildWithName("output").setProperty("oscPort", oscPort.getText(), nullptr);
        saveSettings();
    };

    addAndMakeVisible(oscConnectButton);
    oscConnectButton.onClick = [this] {
        oscOutputProcessor = std::make_unique<OSCOutputProcessor>(oscIp.getText().toStdString(),
                                                                  oscPort.getText().getIntValue());
        oscConnectButton.setEnabled(false);
    };

    /* BUFFER OUTPUT 1 OSC IP */

    addAndMakeVisible(bufferOutputOsc1IpLabel);
    bufferOutputOsc1IpLabel.setText("Buffer Output OSC 1 IP: ", juce::dontSendNotification);
    bufferOutputOsc1IpLabel.attachToComponent(&bufferOutputOsc1Ip, true);

    addAndMakeVisible(bufferOutputOsc1Ip);
    juce::String savedBufferOutputOsc1Ip = settings.getOrCreateChildWithName("bufferOutput1", nullptr)
            .getProperty("oscIp", "127.0.0.1");
    bufferOutputOsc1Ip.setText(savedBufferOutputOsc1Ip);
    bufferOutputOsc1Ip.setInputRestrictions(15, "0123456789.");

    bufferOutputOsc1Ip.onTextChange = [this] {
        bufferOutputOsc1ConnectButton.setEnabled(true);
        DBG("Saving Buffer Output OSC 1 IP: " + bufferOutputOsc1Ip.getText());
        settings.getChildWithName("bufferOutput1").setProperty("oscIp", bufferOutputOsc1Ip.getText(), nullptr);
        saveSettings();
    };

    addAndMakeVisible(bufferOutputOsc1PortLabel);
    bufferOutputOsc1PortLabel.setText(": ", juce::dontSendNotification);
    bufferOutputOsc1PortLabel.attachToComponent(&bufferOutputOsc1Port, true);

    addAndMakeVisible(bufferOutputOsc1Port);
    juce::String savedBufferOutputOsc1Port = settings.getOrCreateChildWithName("bufferOutput1", nullptr).getProperty(
        "oscPort", "9001");
    bufferOutputOsc1Port.setText(savedBufferOutputOsc1Port);
    bufferOutputOsc1Port.setInputRestrictions(6, "0123456789");

    bufferOutputOsc1Port.onTextChange = [this] {
        bufferOutputOsc1ConnectButton.setEnabled(true);
        DBG("Saving Buffer Output OSC 1 Port: " + bufferOutputOsc1Port.getText());
        settings.getChildWithName("bufferOutput1").setProperty("oscPort", bufferOutputOsc1Port.getText(), nullptr);
        saveSettings();
    };

    addAndMakeVisible(bufferOutputOsc1ConnectButton);
    bufferOutputOsc1ConnectButton.onClick = [this] {
        bufferOutputOsc1Processor = std::make_unique<OSCBufferOutputProcessor>(
            bufferOutputOsc1Ip.getText().toStdString(),
            bufferOutputOsc1Port.getText().
            getIntValue());
        bufferOutputOsc1ConnectButton.setEnabled(false);
    };

    /* BUFFER OUTPUT OSC 2 IP */

    addAndMakeVisible(bufferOutputOsc2IpLabel);
    bufferOutputOsc2IpLabel.setText("Buffer Output OSC 2 IP: ", juce::dontSendNotification);
    bufferOutputOsc2IpLabel.attachToComponent(&bufferOutputOsc2Ip, true);

    addAndMakeVisible(bufferOutputOsc2Ip);
    juce::String savedBufferOutputOsc2Ip = settings.getOrCreateChildWithName("bufferOutput2", nullptr)
            .getProperty("oscIp", "127.0.0.1");
    bufferOutputOsc2Ip.setText(savedBufferOutputOsc2Ip);
    bufferOutputOsc2Ip.setInputRestrictions(15, "0123456789.");

    bufferOutputOsc2Ip.onTextChange = [this] {
        bufferOutputOsc2ConnectButton.setEnabled(true);
        DBG("Saving Buffer Output OSC 2 IP: " + bufferOutputOsc2Ip.getText());
        settings.getChildWithName("bufferOutput2").setProperty("oscIp", bufferOutputOsc2Ip.getText(), nullptr);
        saveSettings();
    };

    addAndMakeVisible(bufferOutputOsc2PortLabel);
    bufferOutputOsc2PortLabel.setText(": ", juce::dontSendNotification);
    bufferOutputOsc2PortLabel.attachToComponent(&bufferOutputOsc2Port, true);

    addAndMakeVisible(bufferOutputOsc2Port);
    juce::String savedBufferOutputOsc2Port = settings.getOrCreateChildWithName("bufferOutput2", nullptr).getProperty(
        "oscPort", "9001");
    bufferOutputOsc2Port.setText(savedBufferOutputOsc2Port);
    bufferOutputOsc2Port.setInputRestrictions(6, "0123456789");

    bufferOutputOsc2Port.onTextChange = [this] {
        bufferOutputOsc2ConnectButton.setEnabled(true);
        DBG("Saving Buffer Output OSC 2 Port: " + bufferOutputOsc2Port.getText());
        settings.getChildWithName("bufferOutput2").setProperty("oscPort", bufferOutputOsc2Port.getText(), nullptr);
        saveSettings();
    };

    addAndMakeVisible(bufferOutputOsc2ConnectButton);
    bufferOutputOsc2ConnectButton.onClick = [this] {
        bufferOutputOsc2Processor = std::make_unique<OSCBufferOutputProcessor>(
            bufferOutputOsc2Ip.getText().toStdString(),
            bufferOutputOsc2Port.getText().
            getIntValue());
        bufferOutputOsc2ConnectButton.setEnabled(false);
    };

    /* VISUALIZATION BUFFER SIZE */

    addAndMakeVisible(visualizationBufferSizeLabel);
    visualizationBufferSizeLabel.setText("Visualization Buffer Size:", juce::dontSendNotification);
    visualizationBufferSizeLabel.attachToComponent(&visualizationBufferSize, true);

    addAndMakeVisible(visualizationBufferSize);
    juce::String savedVisualizationBufferSize = settings.getOrCreateChildWithName("bufferOutput", nullptr)
            .getProperty("visualizationBufferSize", "400");
    visualizationBufferSize.setText(savedVisualizationBufferSize);
    visualizationBufferSize.setInputRestrictions(6, "0123456789");

    visualizationBufferSize.onTextChange = [this] {
        DBG("Saving Visualization Buffer Size: " + visualizationBufferSize.getText());
        settings.getChildWithName("bufferOutput").setProperty("visualizationBufferSize",
                                                              visualizationBufferSize.getText(), nullptr);
        visualizationBufferSizeValue = visualizationBufferSize.getText().getIntValue();
        saveSettings();
    };

    /* VELOCITY */

    addAndMakeVisible(velocityLabel);
    velocityLabel.setText("Velocity:", juce::dontSendNotification);
    velocityLabel.attachToComponent(&velocity, true);

    addAndMakeVisible(velocity);
    juce::String savedVelocity = settings.getOrCreateChildWithName("output", nullptr)
            .getProperty("velocity", "50");
    velocity.setText(savedVelocity);
    velocity.setInputRestrictions(3, "0123456789");

    velocity.onTextChange = [this] {
        DBG("Saving Velocity: " + velocity.getText());
        settings.getChildWithName("output").setProperty("velocity", velocity.getText(), nullptr);
        velocityValue = juce::jlimit(1, 127, velocity.getText().getIntValue());
        saveSettings();
    };

    /* STATUS OUTPUT OSC IP */

    addAndMakeVisible(statusOutputOscIpLabel);
    statusOutputOscIpLabel.setText("Status Output OSC IP: ", juce::dontSendNotification);
    statusOutputOscIpLabel.attachToComponent(&statusOutputOscIp, true);

    addAndMakeVisible(statusOutputOscIp);
    juce::String savedStatusOutputOscIp = settings.getOrCreateChildWithName("statusOutput", nullptr)
            .getProperty("oscIp", "127.0.0.1");
    statusOutputOscIp.setText(savedStatusOutputOscIp);
    statusOutputOscIp.setInputRestrictions(15, "0123456789.");

    statusOutputOscIp.onTextChange = [this] {
        statusOutputConnectButton.setEnabled(true);
        DBG("Saving Status Output OSC IP: " + statusOutputOscIp.getText());
        settings.getChildWithName("statusOutput").setProperty("oscIp", statusOutputOscIp.getText(), nullptr);
        saveSettings();
    };

    addAndMakeVisible(statusOutputOscPortLabel);
    statusOutputOscPortLabel.setText(": ", juce::dontSendNotification);
    statusOutputOscPortLabel.attachToComponent(&statusOutputOscPort, true);

    addAndMakeVisible(statusOutputOscPort);
    juce::String savedStatusOutputOscPort = settings.getOrCreateChildWithName("statusOutput", nullptr).getProperty(
        "oscPort", "9001");
    statusOutputOscPort.setText(savedStatusOutputOscPort);
    statusOutputOscPort.setInputRestrictions(6, "0123456789");

    statusOutputOscPort.onTextChange = [this] {
        statusOutputConnectButton.setEnabled(true);
        DBG("Saving Status Output OSC Port: " + statusOutputOscPort.getText());
        settings.getChildWithName("statusOutput").setProperty("oscPort", statusOutputOscPort.getText(), nullptr);
        saveSettings();
    };

    addAndMakeVisible(statusOutputConnectButton);
    statusOutputConnectButton.onClick = [this] {
        statusOutputProcessor = std::make_unique<OSCStatusOutputProcessor>(statusOutputOscIp.getText().toStdString(),
                                                                           statusOutputOscPort.getText().getIntValue());
        statusOutputConnectButton.setEnabled(false);
    };

    /**************/
    /* GENERATION */
    /**************/

    addAndMakeVisible(generationLabel);
    generationLabel.setText("Generation", juce::dontSendNotification);
    generationLabel.setFont(juce::Font(20.0f));

    /* GENERATION STATUS PROGRESS BAR */
    generationStatusProgressBar = std::make_unique<juce::ProgressBar>(progress);
    generationStatusProgressBar->setLookAndFeel(&progressBarLookAndFeel);
    addAndMakeVisible(generationStatusProgressBar.get());
    addAndMakeVisible(generationStatusProgressBarLabel);
    generationStatusProgressBarLabel.setText("Generation Status:", juce::dontSendNotification);
    generationStatusProgressBarLabel.attachToComponent(generationStatusProgressBar.get(), false);

    /* START BUTTON */
    addAndMakeVisible(startButton);
    startButton.onClick = [this] {
        start();
    };

    /* STOP BUTTON */
    addAndMakeVisible(stopButton);
    stopButton.onClick = [this] {
        stop();
    };

    /* SAVE LAST GEN BUTTON */
    addAndMakeVisible(saveLastGenButton);
    saveLastGenButton.onClick = [this] {
        juce::File lastGenFile = juce::File::getSpecialLocation(
                    juce::File::SpecialLocationType::userDocumentsDirectory)
                .getChildFile("VirtualOrch")
                .getChildFile("last_gen.txt");

        // write musicTransformer's inputData into file
        auto outputStream = lastGenFile.createOutputStream();
        if (outputStream->openedOk()) {
            outputStream->setPosition(0);
            outputStream->truncate();
            for (const auto &token: musicTransformer.getInputData()) {
                outputStream->writeText(std::to_string(token) + " ", false, false, "\n");
            }
        }
    };

    /* OPEN TRANSPORT BUTTON */
    addAndMakeVisible(openTransport);
    openTransport.onClick = [this, displays] {
        if (transportWindow) {
            transportWindow->toFront(true);
        } else {
            transportWindow = new TransportComponent(displays.size() >= 2, clock, musicTransformer, modelConfig);
            transportWindow->addToDesktop(juce::ComponentPeer::windowHasCloseButton |
                                          juce::ComponentPeer::windowHasTitleBar);
            if (displays.size() >= 2) {
                transportWindow->setSize(displays[1].userArea.getWidth(), displays[1].userArea.getHeight());
                transportWindow->setTopLeftPosition(
                    displays[1].userArea.getPosition() + transportWindow->getPosition());
            } else {
                transportWindow->setSize(300, 150);
                transportWindow->setTopRightPosition(
                    juce::Desktop::getInstance().getDisplays().getPrimaryDisplay()->userArea.getWidth() / 2 + 150, 0);
            }
            transportWindow->setVisible(true);
            transportWindow->toFront(true);
        }
    };

    /* OPEN METRICS BUTTON */
    addAndMakeVisible(openMetrics);
    openMetrics.onClick = [this] {
        if (metricsWindow) {
            metricsWindow->toFront(true);
        } else {
            metricsWindow = new MetricsComponent(metrics);
            musicTransformer.metricsWindow = metricsWindow;
            metricsWindow->addToDesktop(juce::ComponentPeer::windowHasCloseButton |
                                        juce::ComponentPeer::windowHasTitleBar);
            metricsWindow->setSize(1000, 1000);
            metricsWindow->setTopRightPosition(
                juce::Desktop::getInstance().getDisplays().getPrimaryDisplay()->userArea.getWidth() / 2 + 150, 0);
            metricsWindow->setVisible(true);
            metricsWindow->toFront(true);
        }
    };

    /* AUTO CONNECT BUTTON */
    addAndMakeVisible(autoConnect);
    autoConnect.setButtonText("Auto Connect");
    bool savedAutoConnect = settings.getOrCreateChildWithName("general", nullptr).getProperty("autoConnect", false);
    autoConnect.setToggleState(savedAutoConnect, juce::dontSendNotification);
    autoConnect.onClick = [this] {
        DBG("Saving Auto Connect: " + std::to_string(autoConnect.getToggleState()));
        settings.getChildWithName("general").setProperty("autoConnect", autoConnect.getToggleState(), nullptr);
        saveSettings();
    };
    if (autoConnect.getToggleState()) {
        oscConnectButton.triggerClick();
        controllerConnectButton.triggerClick();
        bufferOutputOsc1ConnectButton.triggerClick();
        bufferOutputOsc2ConnectButton.triggerClick();
        statusOutputConnectButton.triggerClick();
    }

    setSize(700, 800);
}

MainComponent::~MainComponent() {
    saveSettings();
    deviceManager.removeMidiInputDeviceCallback(
        juce::MidiInput::getAvailableDevices()[midiInputList.getSelectedItemIndex()].identifier, this);
    deviceManager.removeMidiInputDeviceCallback(
        juce::MidiInput::getAvailableDevices()[extraMidiInputList.getSelectedItemIndex()].identifier, this);
    musicTransformer.stopThread(-1);
    stopThread(-1);
    generationStatusProgressBar->setLookAndFeel(nullptr);
    delete presetSaveDialog;
    delete modelConfigurationWindow;
    delete transportWindow;
    delete metricsWindow;
}

void MainComponent::endpointsChanged()
{
    midiInputs = juce::MidiInput::getAvailableDevices();
    midiInputNames.clear();
    for (auto input: midiInputs) {
        midiInputNames.add(input.name);
    }

    bool mtcFirstUpdateOrNoDevices = (mtcClockList.getNumItems() == 0);
    juce::NotificationType notifMtc = juce::sendNotificationAsync;
    if (!mtcFirstUpdateOrNoDevices) {
        notifMtc = juce::dontSendNotification;
        mtcClockList.clear(notifMtc);
    }
    mtcClockList.addItemList(midiInputNames, 1);

    // TODO (Perry): Possible race if settings are not saved then won't notify changes properly
    juce::String savedMtcClock = settings.getOrCreateChildWithName("mtcClock", nullptr).getProperty("identifier", "");
    if (savedMtcClock.isNotEmpty()) {
        DBG("Saved MTC Clock: " + savedMtcClock);
        // Find the index of the saved input
        int index = -1;
        for (int i = 0; i < midiInputs.size(); i++) {
            if (midiInputs[i].identifier == savedMtcClock) {
                index = i;
                break;
            }
        }
        // If the saved input is found, set it
        if (index != -1) {
            DBG("Set to saved MTC Clock.");
            mtcClockList.setSelectedItemIndex(index, notifMtc);
        } else {
            DBG("Saved MTC Clock not found, setting to first input.");
            mtcClockList.setSelectedItemIndex(0);
        }
    } else {
        DBG("No saved MTC Clock, setting to first input.");
        mtcClockList.setSelectedItemIndex(0);
    }

    bool inputsFirstUpdateOrNoDevices = (midiInputList.getNumItems() == 0);
    juce::NotificationType notifInputs = juce::sendNotificationAsync;
    if (!inputsFirstUpdateOrNoDevices) {
        notifInputs = juce::dontSendNotification;
        midiInputList.clear(notifInputs);
        extraMidiInputList.clear(notifInputs);
    }
    midiInputList.addItemList(midiInputNames, 1);
    extraMidiInputList.addItemList(midiInputNames, 1);

    // TODO (Perry): Possible race if settings are not saved then won't notify changes properly
    juce::String savedInput = settings.getOrCreateChildWithName("input", nullptr).getProperty("identifier", "");
    if (savedInput.isNotEmpty()) {
        DBG("Saved Input: " + savedInput);
        // Find the index of the saved input
        int index = -1;
        for (int i = 0; i < midiInputs.size(); i++) {
            if (midiInputs[i].identifier == savedInput) {
                index = i;
                break;
            }
        }
        // If the saved input is found, set it
        if (index != -1) {
            DBG("Set to saved input.");
            midiInputList.setSelectedItemIndex(index, notifInputs);
        } else {
            DBG("Saved input not found, setting to first input.");
            midiInputList.setSelectedItemIndex(0);
        }
    } else {
        DBG("No saved input, setting to first input.");
        midiInputList.setSelectedItemIndex(0);
    }

    juce::String savedExtraMidiInput = settings.getOrCreateChildWithName("extraMidiInput", nullptr).getProperty("identifier", "");
    if (savedExtraMidiInput.isNotEmpty()) {
        DBG("Saved Extra MIDI Input: " + savedInput);
        // Find the index of the saved input
        int index = -1;
        for (int i = 0; i < midiInputs.size(); i++) {
            if (midiInputs[i].identifier == savedInput) {
                index = i;
                break;
            }
        }
        // If the saved input is found, set it
        if (index != -1) {
            DBG("Set to saved extra MIDI input.");
            extraMidiInputList.setSelectedItemIndex(index, notifInputs);
        } else {
            DBG("Saved extra MIDI input not found, setting to first input.");
            extraMidiInputList.setSelectedItemIndex(0);
        }
    } else {
        DBG("No saved extra MIDI input, setting to first input.");
        extraMidiInputList.setSelectedItemIndex(0);
    }

    hardwareMidiOutputs = juce::MidiOutput::getAvailableDevices();
    midiOutputNames.clear();
    outputIdentifiers.clear();
    outputIdentifiers.push_back("OSC");
    outputIdentifiers.push_back("Virtual MIDI");
    for (const auto &output: hardwareMidiOutputs) {
        midiOutputNames.add(output.name);
        outputIdentifiers.push_back(output.identifier);
    }

    bool outputsFirstUpdateOrNoDevices = (outputList.getNumItems() == 0);
    juce::NotificationType notifOutputs = juce::sendNotificationAsync;
    if (!outputsFirstUpdateOrNoDevices) {
        notifOutputs = juce::dontSendNotification;
        outputList.clear(notifOutputs);
    }
    outputList.addItem("OSC", 1);
    outputList.addItem("Virtual MIDI", 2);
    outputList.addItemList(midiOutputNames, 3);

    // TODO (Perry): Possible race if settings are not saved then won't notify changes properly
    juce::String savedOutput = settings.getOrCreateChildWithName("output", nullptr).getProperty("identifier", "");
    if (savedOutput.isNotEmpty()) {
        DBG("Saved Output: " + savedOutput);
        // Find the index of the saved input
        int index = -1;
        for (int i = 0; i < outputList.getNumItems(); i++) {
            if (outputIdentifiers.at(i) == savedOutput) {
                index = i;
                break;
            }
        }
        // If the saved input is found, set it
        if (index != -1) {
            DBG("Set to saved output.");
            outputList.setSelectedItemIndex(index, notifOutputs);
        } else {
            DBG("Saved output not found, setting to first output.");
            outputList.setSelectedItemIndex(0);
        }
    } else {
        DBG("No saved output, setting to first output.");
        outputList.setSelectedItemIndex(0);
    }
}

void MainComponent::loadSettings() {
    juce::File settingsFile = juce::File(
        juce::File::getSpecialLocation(juce::File::SpecialLocationType::userDocumentsDirectory)
        .getChildFile("VirtualOrch")
        .getChildFile("settings.xml"));

    // If there is no settings file, create an empty XML file
    if (!settingsFile.exists()) {
        settingsFile.create();
        settings = juce::ValueTree("settings");
    } else {
        std::unique_ptr<juce::XmlElement> xmlSettings = juce::XmlDocument(settingsFile).getDocumentElement();
        settings = juce::ValueTree::fromXml(*xmlSettings);
    }
}

void MainComponent::saveSettings() {
    juce::File settingsFile = juce::File(
        juce::File::getSpecialLocation(juce::File::SpecialLocationType::userDocumentsDirectory)
        .getChildFile("VirtualOrch")
        .getChildFile("settings.xml"));

    std::unique_ptr<juce::XmlElement> xmlSettings = settings.createXml();
    xmlSettings->writeToFile(settingsFile, "");
}

/**
 * Loads both the model and the preset from a given name (if it exists)
 * @param presetName The name of the preset to load
 */
void MainComponent::loadPresetFromName(const juce::String &presetName, const juce::File &presetsDir) {
    const juce::File presetFile(presetsDir.getChildFile(presetName + ".json"));
    if (!presetFile.exists()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "Preset file does not exist.");
        return;
    }

    int index = -1;
    for (int i = 0; i < presetList.getNumItems(); i++) {
        if (presetList.getItemText(i) == presetName) {
            index = i;
            break;
        }
    }
    if (index == -1) {
        DBG("WARNING: Preset " + presetName + " not found in the preset list.");
    } else {
        presetList.setSelectedItemIndex(index, juce::dontSendNotification);
    }

    const juce::var preset = juce::JSON::parse(presetFile);
    if (preset.isVoid()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "Could not parse preset file.");
        return;
    }

    const juce::String modelName = preset.getProperty("model", "");
    // If there is no model
    if (modelName.isEmpty()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "Preset is invalid (model not given).");
        presetList.setSelectedItemIndex(0, juce::dontSendNotification);
        return;
    }

    const juce::String acceleratorName = preset.getProperty("accelerator", "");
    // If there is no accelerator
    if (acceleratorName.isEmpty()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "Preset is invalid (accelerator not given).");
        presetList.setSelectedItemIndex(0, juce::dontSendNotification);
        return;
    }

    MusicModel modelToLoad;
    try {
        modelToLoad = musicModelList.getMusicModelByNameAndAccelerator(modelName, acceleratorName);
    } catch (const std::runtime_error &e) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "Preset is invalid: " + juce::String(e.what()));
        presetList.setSelectedItemIndex(0, juce::dontSendNotification);
        return;
    }

    // Update model only if it is different
    if (modelList.getSelectedId() == 0 || modelToLoad != musicTransformer.getCurrentMusicModel()) {
        // STOP IF CHANGING MODEL
        stop();

        modelList.setSelectedItemIndex(musicModelList.getIndexByMusicModel(modelToLoad), juce::dontSendNotification);
        updateModel(false);
    }
    loadPreset(preset);
}


/**
 * Sets up modelConfig according to the preset values of `parsedJson`.
 * @param parsedJson The JSON object to load the preset from
 */
void MainComponent::loadPreset(const juce::var &parsedJson) {
    modelConfig.loadFromPreset(parsedJson);

    if (modelConfig.inputMode == InputMode::Trading && clock.isRunning()) {
        // Set trading offset to next bar
        DBG("Setting trading offset to next bar: " + std::to_string(clock.getCurrentBar() + 1));
        musicTransformer.tradingOffset.set(currentBar + 1);
    } else {
        // Otherwise reset the trading offset
        musicTransformer.tradingOffset.set(0);
    }

    resized();
}

bool MainComponent::savePreset(const juce::String &presetName) {
    juce::var presetJson(new juce::DynamicObject());
    if (modelList.getSelectedId() == 0) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "Cannot save a preset without a model.");
        return false;
    }

    MusicModel modelSelected = musicModelList.getMusicModelByIndex(modelList.getSelectedItemIndex());
    presetJson.getDynamicObject()->setProperty("model", modelSelected.name);
    juce::String acceleratorName = MLFramework::ACCELERATOR_NAMES.at(modelSelected.accelerator);
    presetJson.getDynamicObject()->setProperty("accelerator", acceleratorName);

    switch (modelConfig.inputMode) {
        case Direct:
            presetJson.getDynamicObject()->setProperty("inputMode", "direct");
            presetJson.getDynamicObject()->setProperty("directInputWindowLength", modelConfig.directInputWindowLength);
            presetJson.getDynamicObject()->setProperty("directInputManualTrigger",
                                                       modelConfig.directInputManualTrigger);
            if (modelConfig.directInputManualTriggerControlType == ControlType::Note) {
                presetJson.getDynamicObject()->setProperty("directInputManualTriggerControlType", "note");
            } else if (modelConfig.directInputManualTriggerControlType == ControlType::CC) {
                presetJson.getDynamicObject()->setProperty("directInputManualTriggerControlType", "CC");
            } else if (modelConfig.directInputManualTriggerControlType == ControlType::OscController) {
                presetJson.getDynamicObject()->setProperty("directInputManualTriggerControlType", "oscController");
            }
            presetJson.getDynamicObject()->setProperty("directInputManualTriggerControlId",
                                                       modelConfig.directInputManualTriggerControlId);
            presetJson.getDynamicObject()->setProperty("directInputSendNoteOffs",
                                                       modelConfig.directInputSendNoteOffs);
            presetJson.getDynamicObject()->setProperty("directInputSnapOnBar",
                                                       modelConfig.directInputSnapOnBar);
            presetJson.getDynamicObject()->setProperty("directInputStartOnInput",
                                                       modelConfig.directInputStartOnInput);
            presetJson.getDynamicObject()->setProperty("directInputStartDelay",
                                                       modelConfig.directInputStartDelay);
            presetJson.getDynamicObject()->setProperty("directInputUnpauseOnInput",
                                                       modelConfig.directInputUnpauseOnInput);
            presetJson.getDynamicObject()->setProperty("directInputUnpauseDelay",
                                                       modelConfig.directInputUnpauseDelay);
            presetJson.getDynamicObject()->setProperty("directInputManualPause",
                                                       modelConfig.directInputManualPause);
            if (modelConfig.directInputManualPauseControlType == ControlType::Note) {
                presetJson.getDynamicObject()->setProperty("directInputManualPauseControlType", "note");
            } else if (modelConfig.directInputManualPauseControlType == ControlType::CC) {
                presetJson.getDynamicObject()->setProperty("directInputManualPauseControlType", "CC");
            } else if (modelConfig.directInputManualPauseControlType == ControlType::OscController) {
                presetJson.getDynamicObject()->setProperty("directInputManualPauseControlType", "oscController");
            }
            if (modelConfig.directInputManualPauseControlTriggerType == ControlTriggerType::Toggle) {
                presetJson.getDynamicObject()->setProperty("directInputManualPauseControlTriggerType", "toggle");
            } else if (modelConfig.directInputManualPauseControlTriggerType == ControlTriggerType::Momentary) {
                presetJson.getDynamicObject()->setProperty("directInputManualPauseControlTriggerType", "momentary");
            }
            presetJson.getDynamicObject()->setProperty("directInputManualPauseControlId",
                                                       modelConfig.directInputManualPauseControlId);
            presetJson.getDynamicObject()->setProperty("directInputManualPauseClearsFutureNotes",
                                                       modelConfig.directInputManualPauseClearsFutureNotes);
            presetJson.getDynamicObject()->setProperty("directInputHoldBass",
                                                       modelConfig.directInputHoldBass);
            presetJson.getDynamicObject()->setProperty("directInputBassLow", modelConfig.directInputBassLow);
            presetJson.getDynamicObject()->setProperty("directInputBassHigh", modelConfig.directInputBassHigh);
            presetJson.getDynamicObject()->setProperty("directInputInitialBass", modelConfig.directInputInitialBass);
            break;
        case Buffer:
            presetJson.getDynamicObject()->setProperty("inputMode", "buffer");
            presetJson.getDynamicObject()->setProperty("bufferInputPromptOnNextBar",
                                                       modelConfig.bufferInputPromptOnNextBar);
            presetJson.getDynamicObject()->setProperty("bufferInputForceOnNextBar",
                                                       modelConfig.bufferInputForceOnNextBar);
            presetJson.getDynamicObject()->setProperty("bufferInputSize", modelConfig.bufferInputSize);
            presetJson.getDynamicObject()->setProperty("bufferInputLow", modelConfig.bufferInputLow);
            presetJson.getDynamicObject()->setProperty("bufferInputHigh", modelConfig.bufferInputHigh);
            presetJson.getDynamicObject()->setProperty("bufferInputBassLow", modelConfig.bufferInputBassLow);
            presetJson.getDynamicObject()->setProperty("bufferInputBassHigh", modelConfig.bufferInputBassHigh);
            break;
        case Trading:
            presetJson.getDynamicObject()->setProperty("inputMode", "trading");
            presetJson.getDynamicObject()->
                    setProperty("tradingInputNumberOfBars", modelConfig.tradingInputNumberOfBars);
            presetJson.getDynamicObject()->
                    setProperty("tradingInputModelGoesFirst", modelConfig.tradingInputModelGoesFirst);
            presetJson.getDynamicObject()->
                    setProperty("tradingInputInitialOffset", modelConfig.tradingInputInitialOffset);
            break;
        case Playback:
            presetJson.getDynamicObject()->setProperty("inputMode", "playback");
            presetJson.getDynamicObject()->setProperty("playbackInputStartTime", modelConfig.playbackInputStartTime);
            presetJson.getDynamicObject()->setProperty("playbackInputEndTime", modelConfig.playbackInputEndTime);
            presetJson.getDynamicObject()->setProperty("playbackInputControlSequence",
                                                       modelConfig.playbackInputControlSequence);
            break;
    }

    presetJson.getDynamicObject()->setProperty("inputInstrument", modelConfig.inputInstrument);
    presetJson.getDynamicObject()->setProperty("inputLow", modelConfig.inputLow);
    presetJson.getDynamicObject()->setProperty("inputHigh", modelConfig.inputHigh);
    presetJson.getDynamicObject()->setProperty("inputDuration", modelConfig.inputDuration);
    presetJson.getDynamicObject()->setProperty("inputClearsPast", modelConfig.inputClearsPast);
    presetJson.getDynamicObject()->setProperty("inputClicks", modelConfig.inputClicks);
    presetJson.getDynamicObject()->setProperty("inputInitialData", modelConfig.inputInitialData);
    presetJson.getDynamicObject()->setProperty("inputInitialDataRefreshRate", modelConfig.inputInitialDataRefreshRate);

    presetJson.getDynamicObject()->setProperty("outputMinimumDuration", modelConfig.outputMinimumDuration);
    presetJson.getDynamicObject()->setProperty("outputMaximumDuration", modelConfig.outputMaximumDuration);
    presetJson.getDynamicObject()->setProperty("outputMinimumTimeInterval", modelConfig.outputMinimumTimeInterval);
    presetJson.getDynamicObject()->setProperty("outputAllowChords", modelConfig.outputAllowChords);
    presetJson.getDynamicObject()->setProperty("outputMaximumChordSize", modelConfig.outputMaximumChordSize);
    presetJson.getDynamicObject()->setProperty("outputSendBarSeparators", modelConfig.outputSendBarSeparators);
    presetJson.getDynamicObject()->setProperty("outputBarLength", modelConfig.outputBarLength);
    juce::var outputTemperatures;
    outputTemperatures.append(modelConfig.outputTemperatures[0]);
    outputTemperatures.append(modelConfig.outputTemperatures[1]);
    outputTemperatures.append(modelConfig.outputTemperatures[2]);
    presetJson.getDynamicObject()->setProperty("outputTemperatures", outputTemperatures);
    presetJson.getDynamicObject()->setProperty("outputStartTime", modelConfig.outputStartTime);
    presetJson.getDynamicObject()->setProperty("outputForceStartTime", modelConfig.outputForceStartTime);
    presetJson.getDynamicObject()->
            setProperty("outputPauseAfterTimeInterval", modelConfig.outputPauseAfterTimeInterval);
    presetJson.getDynamicObject()->setProperty("outputPauseAfterTimeIntervalValue",
                                               modelConfig.outputPauseAfterTimeIntervalValue);
    presetJson.getDynamicObject()->setProperty("outputPauseAfterDuration", modelConfig.outputPauseAfterDuration);
    presetJson.getDynamicObject()->setProperty("outputPauseAfterDurationValue",
                                               modelConfig.outputPauseAfterDurationValue);
    presetJson.getDynamicObject()->setProperty("outputPauseAfterTradingSequence", modelConfig.outputPauseAfterTradingSequence);

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

    presetsDir.createDirectory();
    juce::File presetFile = presetsDir.getChildFile(presetName + ".json");
    juce::FileOutputStream output(presetFile);
    if (output.openedOk()) {
        output.setPosition(0);
        output.truncate();
    } else {
        DBG("Failed to write preset file: " + presetFile.getFullPathName());
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "Could not open file for writing.");
        return false;
    }
    juce::JSON::writeToStream(output, presetJson);

    presetList.addItem(presetName, presetList.getNumItems() + 1);
    presetList.setSelectedItemIndex(presetList.getNumItems(), juce::dontSendNotification);
    return true;
}


void MainComponent::moveToLeftPreset() {
    int index = presetList.getSelectedItemIndex();
    if (index == 0) {
        index = presetList.getNumItems() - 1;
    } else {
        index--;
    }
    presetList.setSelectedItemIndex(index, juce::sendNotification);
}

void MainComponent::moveToRightPreset() {
    int index = presetList.getSelectedItemIndex();
    if (index == presetList.getNumItems() - 1) {
        index = 0;
    } else {
        index++;
    }
    presetList.setSelectedItemIndex(index, juce::sendNotification);
}


void MainComponent::clearModel() {
    modelList.setSelectedId(0, juce::dontSendNotification);
    stop();
    // We resize to remove the `modelConfigurationButton`
    modelConfigurationButton.setEnabled(false);
    resized();
}

void MainComponent::updateModel(const bool loadDefaultPresetIfModelChanged) {
    // Send modelLoading status to statusOutputProcessor
    if (statusOutputProcessor != nullptr) {
        statusOutputProcessor->modelLoading();
    }

    // Get Model Config File
    MusicModel modelSelected = musicModelList.getMusicModelByIndex(modelList.getSelectedItemIndex());
    const auto modelName = modelSelected.name;
    juce::File jsonFile(modelSelected.jsonPath);
    juce::var parsedJson = juce::JSON::parse(jsonFile.loadFileAsString());

    // MODEL CONFIG: Set Model Size
    ModelType modelType = ModelType::Small;
    juce::String modelSize = parsedJson.getProperty("size", "").toString();
    if (modelSize == "small") {
        modelType = ModelType::Small;
    } else if (modelSize == "medium") {
        modelType = ModelType::Medium;
    } else if (modelSize == "") {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "Model Config is invalid (size not given).");
        // TODO: Doesn't this just set to the first item?
        modelList.setSelectedId(0, juce::dontSendNotification);
        return;
    } else {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "Model Config is invalid (size '" + modelSize +
                                               "' is not valid).");
        // TODO: Doesn't this just set to the first item?
        modelList.setSelectedId(0, juce::dontSendNotification);
        return;
    }

    // MODEL CONFIG: Set Output Instruments
    modelConfig.outputInstruments.clear();
    auto *jsonOutputInstruments = parsedJson.getProperty("outputInstruments", var()).getArray();
    if (jsonOutputInstruments && !jsonOutputInstruments->isEmpty()) {
        for (int32_t instrumentId: *jsonOutputInstruments) {
            // Add default instrument config
            OutputInstrumentConfig outputInstrumentConfig{false, false, 36, 120};
            modelConfig.outputInstruments.emplace(instrumentId, outputInstrumentConfig);
        }
    } else {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "Model Config is invalid (no output instrument).");
        // TODO: Doesn't this just set to the first item?
        modelList.setSelectedId(0, juce::dontSendNotification);
        return;
    }

    // Load preset
    if (loadDefaultPresetIfModelChanged && modelName != musicTransformer.getCurrentMusicModel().name) {
        DBG("Loading default preset");
        loadPreset(parsedJson.getProperty("defaultPreset", var()));
        // We are changing the model, so mark preset as modified
        markPresetAsEdited();
    }

    musicTransformer.init(modelSelected, modelType);

    // Send modelLoaded status to statusOutputProcessor
    if (statusOutputProcessor != nullptr) {
        statusOutputProcessor->modelLoaded();
    }

    modelConfigurationButton.setEnabled(true);

    // Update the OuputProcessor with the new instruments
    updateOutputProcessor();
}

void MainComponent::start() {
    if (modelConfig.inputMode == InputMode::Direct && modelConfig.directInputStartOnInput) {
        musicTransformer.directInputBlock = true;
    } else {
        musicTransformer.directInputBlock = false;
    }
    musicTransformer.startThread();
    startThread();
    if (!mtcClockActive) {
        clock.startAtTime(0);
    } else {
        clock.setMtcTime(0);
    }
}

void MainComponent::stop() {
    musicTransformer.signalThreadShouldExit();
    signalThreadShouldExit();
    clock.stop();
    progress = 0.0;
    if (outputProcessor != nullptr) {
        outputProcessor->clear();
    }
    if (oscOutputProcessor != nullptr) {
        oscOutputProcessor->clear();
    }
}

void MainComponent::markPresetAsEdited() {
    if (presetList.getText().getLastCharacter() != '*') {
        presetList.setText(presetList.getText() + "*", juce::dontSendNotification);
    }
    savePresetButton.setEnabled(true);
}


//==============================================================================
void MainComponent::paint(juce::Graphics &g) {
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

void MainComponent::resized() {
    /* Reset visibiliy of conditional input fields */
    oscIp.setVisible(false);
    oscPort.setVisible(false);
    oscConnectButton.setVisible(false);

    auto area = getLocalBounds();

    auto presetArea = area.removeFromTop(60).removeFromRight(getWidth() - 50).reduced(8);
    presetList.setBounds(presetArea.removeFromLeft(getWidth() - 230).reduced(5));
    leftPresetButton.setBounds(presetArea.removeFromLeft(40).reduced(5));
    rightPresetButton.setBounds(presetArea.removeFromLeft(40).reduced(5));
    savePresetButton.setBounds(presetArea.reduced(5));

    configLabel.setBounds(area.removeFromTop(60).withTrimmedTop(20).reduced(8));

    auto controllerArea = area.removeFromTop(36).removeFromRight(getWidth() - 150).reduced(8);
    controllerOscPort.setBounds(controllerArea.removeFromLeft(50));
    controllerConnectButton.setBounds(controllerArea.removeFromLeft(120).withTrimmedLeft(20));
    autoConnect.setBounds(controllerArea.removeFromRight(100));

    auto mtcClockArea = area.removeFromTop(36).removeFromRight(getWidth() - 150).reduced(8);
    mtcClock.setBounds(mtcClockArea.removeFromLeft(getWidth() - 600));
    mtcClockList.setBounds(mtcClockArea.removeFromLeft(380).withTrimmedRight(80));
    mtcClockOffset.setBounds(mtcClockArea);

    auto midiInputArea = area.removeFromTop(36).removeFromRight(getWidth() - 150).reduced(8);
    midiInputList.setBounds(midiInputArea.removeFromLeft(getWidth() - 300));
    inputThru.setBounds(midiInputArea.withTrimmedLeft(80));

    auto extraMidiInputArea = area.removeFromTop(36).removeFromRight(getWidth() - 150).reduced(8);
    extraMidiInputList.setBounds(extraMidiInputArea.removeFromLeft(getWidth() - 300));

    auto modelConfigArea = area.removeFromTop(36).removeFromRight(getWidth() - 150).reduced(8);
    modelList.setBounds(modelConfigArea.removeFromLeft(getWidth() - 300).withTrimmedRight(20));
    modelConfigurationButton.setBounds(modelConfigArea);

    outputList.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 150).reduced(8));

    // if (enableOscConfig) {
    oscIp.setVisible(true);
    oscPort.setVisible(true);
    oscConnectButton.setVisible(true);
    auto oscConfigArea = area.removeFromTop(36).removeFromRight(getWidth() - 150).reduced(8);
    oscIp.setBounds(oscConfigArea.removeFromLeft(getWidth() - 350).withTrimmedRight(20));
    oscPort.setBounds(oscConfigArea.removeFromLeft(60).withTrimmedRight(10));
    oscConnectButton.setBounds(oscConfigArea);
    // }

    auto bufferOutputOsc1ConfigArea = area.removeFromTop(36).removeFromRight(getWidth() - 150).reduced(8);
    bufferOutputOsc1Ip.setBounds(bufferOutputOsc1ConfigArea.removeFromLeft(getWidth() - 350).withTrimmedRight(20));
    bufferOutputOsc1Port.setBounds(bufferOutputOsc1ConfigArea.removeFromLeft(60).withTrimmedRight(10));
    bufferOutputOsc1ConnectButton.setBounds(bufferOutputOsc1ConfigArea);

    auto bufferOutputOsc2ConfigArea = area.removeFromTop(36).removeFromRight(getWidth() - 150).reduced(8);
    bufferOutputOsc2Ip.setBounds(bufferOutputOsc2ConfigArea.removeFromLeft(getWidth() - 350).withTrimmedRight(20));
    bufferOutputOsc2Port.setBounds(bufferOutputOsc2ConfigArea.removeFromLeft(60).withTrimmedRight(10));
    bufferOutputOsc2ConnectButton.setBounds(bufferOutputOsc2ConfigArea);

    auto outputValuesArea = area.removeFromTop(36).removeFromRight(getWidth() - 150).reduced(8);
    visualizationBufferSize.setBounds(outputValuesArea.removeFromLeft(250).withTrimmedRight(200));
    velocity.setBounds(outputValuesArea.removeFromLeft(60).withTrimmedRight(10));

    auto statusOutputOscConfigArea = area.removeFromTop(36).removeFromRight(getWidth() - 150).reduced(8);
    statusOutputOscIp.setBounds(statusOutputOscConfigArea.removeFromLeft(getWidth() - 350).withTrimmedRight(20));
    statusOutputOscPort.setBounds(statusOutputOscConfigArea.removeFromLeft(60).withTrimmedRight(10));
    statusOutputConnectButton.setBounds(statusOutputOscConfigArea);

    generationLabel.setBounds(area.removeFromTop(60).withTrimmedTop(20).reduced(8));

    generationStatusProgressBar->setBounds(area.removeFromTop(60).reduced(8).withTrimmedTop(20));

    auto transportButtonsArea = area.removeFromTop(80);
    startButton.setBounds(transportButtonsArea.removeFromLeft(getWidth() / 2).reduced(20));
    stopButton.setBounds(transportButtonsArea.removeFromLeft(getWidth() / 2).reduced(20));

    auto extraButtonsArea = area.removeFromTop(80);
    openTransport.setBounds(extraButtonsArea.removeFromLeft(getWidth() / 2).reduced(20));
    openMetrics.setBounds(extraButtonsArea.removeFromLeft(getWidth() / 2).reduced(20));

    saveLastGenButton.setBounds(area.removeFromBottom(40).reduced(8));
}

void MainComponent::updateOutputProcessor() {
    outputProcessor = nullptr;
    switch (outputList.getSelectedItemIndex()) {
        case 0: // OSC
            // oscConnectButton.setEnabled(true);
            // enableOscConfig = true;
            break;
        case 1: // Virtual MIDI
            outputProcessor = std::make_unique<
                MidiOutputProcessor>(MidiOutputType::VIRTUAL, "virtual-orch Virtual MIDI Output",
                                     modelConfig.getOutputInstrumentsIds());

            // enableOscConfig = false;
            break;
        default: // Hardware MIDI
            outputProcessor = std::make_unique<MidiOutputProcessor>(MidiOutputType::HARDWARE,
                                                                    hardwareMidiOutputs[
                                                                        outputList.getSelectedItemIndex() - 2].
                                                                    identifier,
                                                                    modelConfig.getOutputInstrumentsIds());

            // enableOscConfig = false;
            break;
    }
    resized();
};

/**
 * Handles the note (RECEIVED BY THE MUSIC TRANSFORMER) by:
 *  - Starting the clock if not using an MTC Clock and it's the first bar
 *  - Sending the notes to the Music Transformer if using Trading Input and it's time to alternate
 *  - Sending the note to the Output Processor
 *
 * If we need to retriggered the MusicTransformer generation, we return true.
 *
 * @param token the token to handle
 * @param tokenEventType the type of event (note on or note off)
 * @param time the time of the event
 * @return whether the token triggered a regeneration (in which case we need to stop playing notes!)
 */
bool MainComponent::handleNote(Token token, TokenNoteType tokenEventType, uint32_t time) {
    bool generationRetriggered = false;
    if (token.note == Vocab::BarSeparator) {
        // If first bar (and not using an external clock), start the clock!
        if (token.time == 0 && !mtcClockActive) {
            DBG("Starting clock from now");
            clock.setTime(0);
        }

        auto tradingOffset = musicTransformer.tradingOffset.get();
        // If we use Trading Input and we need to alternate, send the tokens to the Music Transformer.
        if (modelConfig.inputMode == InputMode::Trading
            && ((modelConfig.tradingInputModelGoesFirst &&
                 ((currentBar - tradingOffset) % (modelConfig.tradingInputNumberOfBars * 2))
                 == 0)
                || (!modelConfig.tradingInputModelGoesFirst &&
                    ((currentBar - tradingOffset) % (modelConfig.tradingInputNumberOfBars * 2))
                    == modelConfig.tradingInputNumberOfBars))) {
            // Add all remaining notes on to the queue
            for (auto &note: notesOnToSend) {
                Token inputToken = {
                    static_cast<int32_t>(note.second), static_cast<int32_t>(Vocab::DurOffset + (time - note.second)),
                    static_cast<int32_t>(Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument + note.
                                         first)
                };
                tokensToSend.push_back(inputToken);
            }
            notesOnToSend.clear();
            sendTokensToMusicTransformer(std::nullopt);
            generationRetriggered = true;
        } else if (currentBar == tradingOffset && currentBar != 0) {
            // If we are at the start of a new trading pattern, clear the queue
            musicTransformer.inputTokenQueue.push(token);
        }

        // Also send the Bar Separator event through the Output Processor.
        BarSeparatorEvent barSeparatorEvent = {currentBar};
        if (outputProcessor != nullptr) {
            outputProcessor->send(barSeparatorEvent);
        }
        if (oscOutputProcessor != nullptr) {
            oscOutputProcessor->send(barSeparatorEvent);
        }
    } else if (outputProcessor != nullptr || oscOutputProcessor != nullptr) {
        int32_t instrument = (token.note - Vocab::NoteOffset) / Config::MaxPitch;
        int32_t pitch = (token.note - Vocab::NoteOffset) % Config::MaxPitch;
        switch (tokenEventType) {
            case TokenNoteOn: {
                NoteOnEvent noteOnEvent = {instrument, pitch, velocityValue / 127.0f};
                if (outputProcessor != nullptr) {
                    outputProcessor->send(noteOnEvent);
                }
                if (oscOutputProcessor != nullptr) {
                    oscOutputProcessor->send(noteOnEvent);
                }
                break;
            }
            case TokenNoteOff: {
                NoteOffEvent noteOffEvent = {instrument, pitch};
                if (outputProcessor != nullptr) {
                    outputProcessor->send(noteOffEvent);
                }
                if (oscOutputProcessor != nullptr) {
                    oscOutputProcessor->send(noteOffEvent);
                }
                break;
            }
        }
    }
    return generationRetriggered;
}

void MainComponent::run() {
    std::multiset<Token> nextTokens;

    auto compareDuration = [](Token a, Token b) { return a.time + a.getRealDuration() < b.time + b.getRealDuration(); };
    std::multiset<Token, decltype(compareDuration)> currentlyPlaying;

    bool clearedQueueThisRun = false;

    /* Used to skip the loop if we retriggered generation */
    bool generationRetriggered = false;

    tokensToSend.clear();
    notesOnToSend.clear();

    // Set bass held if needed
    if (modelConfig.inputMode == InputMode::Direct && modelConfig.directInputHoldBass
        && modelConfig.directInputInitialBass != -1) {
        bassHeld = modelConfig.directInputInitialBass;
    } else {
        bassHeld = std::nullopt;
    }

    while (!threadShouldExit()) {
        clearedQueueThisRun = false;
        auto [time, newCurrentBar] = clock.getTimeAndBar();
        currentBar = newCurrentBar;
        progress = (static_cast<double>(musicTransformer.getCurrentTime()) - static_cast<double>(time)) / 1000.0;

        // If we need to prompt now, send the tokens to the Music Transformer.
        if (modelConfig.inputMode == InputMode::Buffer && modelConfig.bufferInputPromptOnNextBar && sendOnNextBar
            && time % modelConfig.outputBarLength == 0) {
            sendTokensToMusicTransformer(time);
            sendOnNextBar = false;
        }

        // Collect all notes
        Token token = {-1, -1, -1};
        bool firstNote = true;
        while (musicTransformer.outputTokenQueue.pull(token)) {
            if (firstNote) {
                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                bool lastMidiPlayedSetValue = lastMidiPlayedSet.get();
                if (lastMidiPlayedSetValue) {
                    std::chrono::steady_clock::time_point beginning = lastMidiPlayed.get();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - beginning).count();
                    if (metricsWindow) {
                        metricsWindow->responsivenessFifo.push(duration);
                        lastMidiPlayedSet.set(false);
                    }
                }
            }
            // If it's a signal to clear the queue, clear the queue after the corresponding time
            if (token.note == Vocab::ClearQueue) {
                clearedQueueThisRun = true;
                DBG("Clearing queue!");
                while (!nextTokens.empty() && nextTokens.begin()->time >= token.time) {
                    nextTokens.erase(nextTokens.begin());
                }
                continue;
            }

            // If it's a signal to release notes, release the notes after the corresponding time
            if (token.note == Vocab::ReleaseNotes) {
                DBG("Releasing notes!");
                for (auto it = currentlyPlaying.begin(); it != currentlyPlaying.end();) {
                    if (it->time <= token.time) {
                        handleNote(*it, TokenNoteType::TokenNoteOff, token.time);
                        it = currentlyPlaying.erase(it);
                    } else {
                        break;
                    }
                }
                continue;
            }

            // If it's a signal to pause the queue, pause the queue after the corresponding time
            if (token.note == Vocab::PauseQueue) {
                DBG("Pausing queue!");
                queuePaused = true;
            }
            // If it's a signal to unpause the queue, unpause the queue after the corresponding time
            if (token.note == Vocab::UnpauseQueue) {
                DBG("Unpausing queue!");
                queuePaused = false;
            }

            // If the queue is paused, ignore all notes
            if (queuePaused) {
                continue;
            }

            // If it's a rest, continue
            if (token.note == Vocab::Rest) {
                continue;
            }

            // Else insert it to nextTokens
            // DBG("Inserting to nextTokens at " + std::to_string(time) + ": " + token.toUnderstandableString());
            nextTokens.insert(token);
        }

        // Stop the music transformer if it is time
        if (modelConfig.inputMode == InputMode::Trading && currentBar == stopBar) {
            musicTransformer.signalThreadShouldExit();
            panicBar = currentBar + modelConfig.tradingInputNumberOfBars;
            DBG("Stopping, panic bar: " + std::to_string(panicBar));
            stopBar = -1;
        }

        // Send panic if it is time
        if (modelConfig.inputMode == InputMode::Trading && currentBar == panicBar) {
            DBG("panicking");
            if (statusOutputProcessor != nullptr) {
                statusOutputProcessor->panic();
            }
            stop();
            panicBar = -1;
        }

        // Play notes if it is time
        for (auto it = nextTokens.begin(); it != nextTokens.end();) {
            if (it->time <= time) {
                DBG("Playing note at " + std::to_string(time) + " : " + it->toUnderstandableString());
                // Handle note on
                generationRetriggered = handleNote(*it, TokenNoteType::TokenNoteOn, time);

                // If not bar separator, add it to notes currently playing
                if (it->note != Vocab::BarSeparator) {
                    currentlyPlaying.insert(*it);
                }

                it = nextTokens.erase(it);

                if (generationRetriggered) {
                    break;
                }
            } else {
                break;
            }
        }

        // If we retriggered generation, skip the rest of the loop
        if (generationRetriggered) {
            continue;
        }

        //Send the next `visualizationBufferSizeValue` milliseconds of tokens to the buffer output processor
        if (bufferOutputOsc1Processor != nullptr || bufferOutputOsc2Processor != nullptr) {
            if (bufferOutputOsc1Processor != nullptr) {
                bufferOutputOsc1Processor->setTime(time);
            }
            if (bufferOutputOsc2Processor != nullptr) {
                bufferOutputOsc2Processor->setTime(time);
            }
            if (clearedQueueThisRun) {
                Token clearQueue{static_cast<int32_t>(time), Vocab::DurOffset, Vocab::ClearQueue};
                if (bufferOutputOsc1Processor != nullptr) {
                    bufferOutputOsc1Processor->addToBuffer(clearQueue);
                }
                if (bufferOutputOsc2Processor != nullptr) {
                    bufferOutputOsc2Processor->addToBuffer(clearQueue);
                }
            }
            for (auto it = nextTokens.begin(); it != nextTokens.end(); it++) {
                if (it->time <= time + visualizationBufferSizeValue) {
                    if (bufferOutputOsc1Processor != nullptr) {
                        bufferOutputOsc1Processor->addToBuffer(*it);
                    }
                    if (bufferOutputOsc2Processor != nullptr) {
                        bufferOutputOsc2Processor->addToBuffer(*it);
                    }
                } else {
                    break;
                }
            }
            if (bufferOutputOsc1Processor != nullptr) {
                bufferOutputOsc1Processor->sendBuffer(time);
            }
            if (bufferOutputOsc2Processor != nullptr) {
                bufferOutputOsc2Processor->sendBuffer(time);
            }
        }

        // Stop notes that have finished playing
        for (auto it = currentlyPlaying.begin(); it != currentlyPlaying.end();) {
            if ((it->time + it->getRealDuration()) <= time) {
                handleNote(*it, TokenNoteType::TokenNoteOff, time);
                it = currentlyPlaying.erase(it);
            } else {
                break;
            }
        }
    }
}

void MainComponent::sendTokensToMusicTransformer(const std::optional<int32_t> atTime) {
    const uint32_t time = clock.getTime();
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    lastMidiPlayed.set(now);
    lastMidiPlayedSet.set(true);
    for (auto &token: tokensToSend) {
        // In Trading mode, do not add notes that are before the last window
        if (modelConfig.inputMode == InputMode::Trading && token.time < time - modelConfig.outputBarLength * modelConfig
            .
            tradingInputNumberOfBars) {
            continue;
        }
        if (atTime.has_value()) {
            token.time = atTime.value();
        }
        musicTransformer.inputTokenQueue.push(token);
    }
    // If we use trading and pause after trading sequence, we need to send something even if there were no tokens this
    // sequence, in order to unpause the MusicTransformer
    if (modelConfig.inputMode == InputMode::Trading && modelConfig.outputPauseAfterTradingSequence && tokensToSend.empty()) {
        Token restToken = {atTime.has_value() ? atTime.value() : static_cast<int32_t>(time), Vocab::DurOffset,
                              Vocab::Rest};
        musicTransformer.inputTokenQueue.push(restToken);
    }
    tokensToSend.clear();
}

void MainComponent::triggerDirectInput() {
    if (modelConfig.directInputHoldBass && bassHeld.has_value()) {
        DBG("Holding bass " + std::to_string(bassHeld.value()));
        Token bassToken = {
            static_cast<int32_t>(lastTokenAddedTime),
            static_cast<int32_t>(Vocab::DurOffset + modelConfig.inputDuration),
            bassHeld.value()
        };
        tokensToSend.push_back(bassToken);
    }
    // Add all notes on to the queue
    for (auto &note: notesOnToSend) {
        Token inputToken = {
            static_cast<int32_t>(lastTokenAddedTime),
            static_cast<int32_t>(Vocab::DurOffset + modelConfig.inputDuration),
            static_cast<int32_t>(Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument + note.first)
        };
        tokensToSend.push_back(inputToken);
    }

    // If start on input, remove the block
    if (modelConfig.directInputStartOnInput && musicTransformer.directInputBlock.get()) {
        sendTokensToMusicTransformer(lastTokenAddedTime + modelConfig.directInputStartDelay);
        musicTransformer.directInputBlock = false;
    } else if (modelConfig.directInputUnpauseOnInput && musicTransformer.paused.get()) {
        DBG("REMOVING PAUSE AT " + std::to_string(lastTokenAddedTime + modelConfig.directInputUnpauseDelay));
        sendTokensToMusicTransformer(lastTokenAddedTime + modelConfig.directInputUnpauseDelay);
        // no need to remove pause because sending to the Music Transformer already does it
    } else if (modelConfig.directInputSnapOnBar) {
        auto closestBar = floor((lastTokenAddedTime + (modelConfig.outputBarLength / 2))
                                / modelConfig.outputBarLength);
        sendTokensToMusicTransformer(static_cast<int32_t>(closestBar * modelConfig.outputBarLength));
    } else {
        sendTokensToMusicTransformer(std::nullopt);
    }
}

/**
 * Timer is used to collect input tokens in window and send them after a given window time passed.
 */
void MainComponent::timerCallback() {
    if (modelConfig.inputMode != InputMode::Direct) {
        // Currently, the timer is only used in direct input mode. If this is hit, there might be more code to change.
        throw std::runtime_error("Timer should not be running in non-direct input mode.");
    }

    uint32_t time = clock.getTime();
    // If we have notes on to send (or notes off if they are enabled) and it is time (> windowLength), then send!
    if ((!notesOnToSend.empty() || (modelConfig.directInputSendNoteOffs && !tokensToSend.empty()))
        && time - lastTokenAddedTime > modelConfig.directInputWindowLength) {
        triggerDirectInput();
        stopTimer();
    }
}

void MainComponent::setManualPause(uint32_t atTime, bool pause) {
    // Toggle pause
    if (!pause && musicTransformer.paused.get()) {
        // We add a token to unpause the queue
        Token unpauseQueueToken = {
            static_cast<int32_t>(atTime), static_cast<int32_t>(Vocab::DurOffset),
            Vocab::UnpauseQueue
        };
        musicTransformer.outputTokenQueue.push(unpauseQueueToken);

        // We add a rest to force generation at this time.
        Token newToken = {
            static_cast<int32_t>(atTime), static_cast<int32_t>(Vocab::DurOffset + modelConfig.inputDuration),
            Vocab::Rest
        };
        tokensToSend.push_back(newToken);
        sendTokensToMusicTransformer(std::nullopt);
        // no need to remove pause because sending to the Music Transformer already does it
    } else if (pause && !musicTransformer.paused.get()) {
        musicTransformer.paused.set(true);
        if (modelConfig.directInputManualPauseClearsFutureNotes) {
            Token clearQueueToken = {
                static_cast<int32_t>(atTime), static_cast<int32_t>(Vocab::DurOffset),
                Vocab::ClearQueue
            };
            Token pauseQueueToken = {
                static_cast<int32_t>(atTime), static_cast<int32_t>(Vocab::DurOffset),
                Vocab::PauseQueue
            };
            Token releaseNotesToken = {
                static_cast<int32_t>(atTime), static_cast<int32_t>(Vocab::DurOffset),
                Vocab::ReleaseNotes
            };
            musicTransformer.outputTokenQueue.push(clearQueueToken);
            musicTransformer.outputTokenQueue.push(pauseQueueToken);
            musicTransformer.outputTokenQueue.push(releaseNotesToken);
        }
    }
}

void MainComponent::setMidiInput(int index) {
    auto list = juce::MidiInput::getAvailableDevices();

    // deviceManager.removeMidiInputDeviceCallback(list[lastInputIndex].identifier, this);

    auto newInput = list[index];

    if (!deviceManager.isMidiInputDeviceEnabled(newInput.identifier))
        deviceManager.setMidiInputDeviceEnabled(newInput.identifier, true);

    deviceManager.removeMidiInputDeviceCallback(newInput.identifier, this); // Remove old callback if it exists to avoid duplicates
    deviceManager.addMidiInputDeviceCallback(newInput.identifier, this);
    midiInputList.setSelectedId(index + 1, juce::dontSendNotification);

    // lastInputIndex = index;
    selectedMidiInputIdentifier = newInput.identifier;
}

void MainComponent::setExtraMidiInput(int index) {
    auto list = juce::MidiInput::getAvailableDevices();

    // deviceManager.removeMidiInputDeviceCallback(list[lastInputIndex].identifier, this);

    auto newInput = list[index];

    if (!deviceManager.isMidiInputDeviceEnabled(newInput.identifier))
        deviceManager.setMidiInputDeviceEnabled(newInput.identifier, true);

    deviceManager.removeMidiInputDeviceCallback(newInput.identifier, this); // Remove old callback if it exists to avoid duplicates
    deviceManager.addMidiInputDeviceCallback(newInput.identifier, this);
    extraMidiInputList.setSelectedId(index + 1, juce::dontSendNotification);

    // lastInputIndex = index;
    selectedExtraMidiInputIdentifier = newInput.identifier;
}

void MainComponent::handleIncomingMidiMessage(juce::MidiInput *source, const juce::MidiMessage &message) {
    // MIDI Thru
    if (inputThru.getToggleState() && (source->getIdentifier() == selectedMidiInputIdentifier || source->getIdentifier() == selectedExtraMidiInputIdentifier)
        && (outputProcessor != nullptr || oscOutputProcessor != nullptr)) {
        if (outputProcessor != nullptr) {
            outputProcessor->relayMidi(message);
        }
        // if (oscOutputProcessor != nullptr) {
        //     oscOutputProcessor->relayMidi(message);
        // }
    }
    // If we receive a message from an identifier that's not recognized, ignore it and remove the callback
    if (source->getIdentifier() != selectedMidiInputIdentifier && source->getIdentifier() != selectedExtraMidiInputIdentifier
        && source->getIdentifier() != selectedMtcClockIdentifier) {
        deviceManager.removeMidiInputDeviceCallback(source->getIdentifier(), this);
        DBG("Removing callback for " + source->getName() + " with identifier " + source->getIdentifier() +
            " because it is not selected.");
        return;
    }

    // Only if received from MTC Clock input
    if (mtcClockActive && source->getIdentifier() == selectedMtcClockIdentifier && message.isQuarterFrame()) {
        const int value = message.getQuarterFrameValue();
        switch (message.getQuarterFrameSequenceNumber()) {
            case 0: frames = (frames & 0xf0) | value;
                break;
            case 1: frames = (frames & 0x0f) | (value << 4);
                break;
            case 2: seconds = (seconds & 0xf0) | value;
                break;
            case 3: seconds = (seconds & 0x0f) | (value << 4);
                break;
            case 4: minutes = (minutes & 0xf0) | value;
                break;
            case 5: minutes = (minutes & 0x0f) | (value << 4);
                break;
            case 6: hours = (hours & 0xf0) | value;
                break;
            case 7: hours = (hours & 0x0f) | ((value << 4) & 0x10);
                const uint32_t time = (seconds + (frames / 30.0)) * 100 + minutes * 6000; //+ hours * 360000;
                clock.setMtcTime(time);
                break;
        }
    } else if (message.isController()) {
        handleContinuousControl(message.getControllerNumber(), message.getControllerValue());
    } else if (message.isNoteOn()) {
        handleNoteOn(message.getNoteNumber(), message.getVelocity());
    } else if (message.isNoteOff()) {
        handleNoteOff(message.getNoteNumber());
    }
}

void MainComponent::handleContinuousControl(int controllerNumber, int controllerValue) {
    // Stop if the thread didn't start
    if (!isThreadRunning()) {
        return;
    }

    // If it is the Direct Input Trigger Continuous Control Value, trigger direct input
    if (modelConfig.inputMode == InputMode::Direct && modelConfig.directInputManualTrigger
        && modelConfig.directInputManualTriggerControlType == ControlType::CC
        && controllerNumber == modelConfig.directInputManualTriggerControlId
        && controllerValue == 127) {
        triggerDirectInput();
        return;
    }

    uint32_t time = clock.getTime();

    if (modelConfig.inputMode == InputMode::Direct) {
        if (modelConfig.directInputManualPause
            && modelConfig.directInputManualPauseControlType == ControlType::CC
            && controllerNumber == modelConfig.directInputManualPauseControlId
            && (controllerValue == 127
                || (controllerValue == 0 && modelConfig.directInputManualPauseControlTriggerType ==
                    ControlTriggerType::Momentary))
        ) {
            // For toggle, just toggle the state when we receive 127
            // For momentary, pause when we receive 127, unpause when we receive 0
            bool pause = modelConfig.directInputManualPauseControlTriggerType == ControlTriggerType::Toggle
                             ? !musicTransformer.paused.get()
                             : controllerValue == 127;
            setManualPause(time, pause);
        }
    }
}

void MainComponent::handleNoteOn(int midiNoteNumber, float velocity) {
    // Stop if the thread didn't start
    if (!isThreadRunning()) {
        return;
    }

    // If it is the Direct Input Trigger Note, trigger direct input
    if (modelConfig.inputMode == InputMode::Direct && modelConfig.directInputManualTrigger
        && modelConfig.directInputManualTriggerControlType == ControlType::Note
        && midiNoteNumber == modelConfig.directInputManualTriggerControlId) {
        triggerDirectInput();
        return;
    }

    // Stop if not in the input range and note is not used to control manual pause
    if ((midiNoteNumber < modelConfig.inputLow || midiNoteNumber > modelConfig.inputHigh)
        && (!modelConfig.directInputManualPause || midiNoteNumber != modelConfig.directInputManualPauseControlId)) {
        return;
    }

    uint32_t time = clock.getTime();
    Token inputToken = {
        static_cast<int32_t>(time), static_cast<int32_t>(Vocab::DurOffset + modelConfig.inputDuration),
        static_cast<int32_t>(Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument + midiNoteNumber)
    };

    if (modelConfig.inputMode == InputMode::Direct) {
        if (modelConfig.directInputManualPause
            && modelConfig.directInputManualPauseControlType == ControlType::Note
            && midiNoteNumber == modelConfig.directInputManualPauseControlId) {
            // For toggle, just toggle the state when we receive note on
            // For momentary, pause when we receive note on, unpause when we receive note off
            bool pause = modelConfig.directInputManualPauseControlTriggerType == ControlTriggerType::Toggle
                             ? !musicTransformer.paused.get()
                             : true;
            setManualPause(time, pause);
        } else {
            // If in the bass range and we are holding the bass, keep the note saved
            if (modelConfig.directInputHoldBass &&
                (inputToken.note >= Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument +
                 modelConfig.directInputBassLow
                 && inputToken.note < Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument +
                 modelConfig.directInputBassHigh)) {
                bassHeld = inputToken.note;
                DBG("Setting bass to " + std::to_string(bassHeld.value()));
            } else {
                notesOnToSend[midiNoteNumber] = time;
            }
            // Start the timer callback for direct input (only if not manual trigger or
            // manual pause)
            lastTokenAddedTime = time;
            if (!isTimerRunning() && !modelConfig.directInputManualTrigger && !modelConfig.directInputManualPause) {
                startTimer(10);
            }
        }
    } else if (modelConfig.inputMode == InputMode::Buffer) {
        if (inputToken.note >= Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument +
            modelConfig.bufferInputBassLow
            && inputToken.note < Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument +
            modelConfig.bufferInputBassHigh) {
            // If the note is in the bass zone, send tokens
            // We put the bass first to set time accordingly
            tokensToSend.push_front(inputToken);
            if (modelConfig.inputMode == InputMode::Buffer && modelConfig.bufferInputPromptOnNextBar) {
                sendOnNextBar = true;
            } else {
                sendTokensToMusicTransformer(std::nullopt);
            }
        } else if (inputToken.note >= Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument +
                   modelConfig.bufferInputLow
                   && inputToken.note < Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument +
                   modelConfig.bufferInputHigh) {
            // If the note is in the buffer zone, add it to the queue
            tokensToSend.push_back(inputToken);
        }

        // If queue is bigger than buffer size, remove an item
        if (tokensToSend.size() > modelConfig.bufferInputSize) {
            tokensToSend.pop_front();
        }
    } else if (modelConfig.inputMode == InputMode::Trading) {
        // If trading, add the token to a dictionary
        notesOnToSend[midiNoteNumber] = time;
    }
}

void MainComponent::handleNoteOff(int midiNoteNumber) {
    // Stop if the thread didn't start
    if (!isThreadRunning()) {
        return;
    }

    // Stop if not in the input range and note is not used to control manual pause
    if ((midiNoteNumber < modelConfig.inputLow || midiNoteNumber > modelConfig.inputHigh)
        && (!modelConfig.directInputManualPause || midiNoteNumber != modelConfig.directInputManualPauseControlId)) {
        return;
    }

    uint32_t time = clock.getTime();

    if (modelConfig.inputMode == InputMode::Direct) {
        if (modelConfig.directInputManualPause
            && modelConfig.directInputManualPauseControlType == ControlType::Note
            && midiNoteNumber == modelConfig.directInputManualPauseControlId
            && modelConfig.directInputManualPauseControlTriggerType == ControlTriggerType::Momentary) {
            // For momentary, unpause when we receive note off
            setManualPause(time, false);
        } else if (notesOnToSend.contains(midiNoteNumber)) {
            // If we are in Direct mode and the note is in the dictionary, add a token to the queue if sending Note Offs
            // Otherwise, just erase it
            if (modelConfig.directInputSendNoteOffs) {
                const double onsetTime = notesOnToSend[midiNoteNumber];
                Token inputToken = {
                    static_cast<int32_t>(onsetTime), static_cast<int32_t>(Vocab::DurOffset + (time - onsetTime)),
                    static_cast<int32_t>(Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument +
                                         midiNoteNumber)
                };
                tokensToSend.push_back(inputToken);
            }
            notesOnToSend.erase(midiNoteNumber);
        }
    } else if (modelConfig.inputMode == InputMode::Trading && notesOnToSend.contains(midiNoteNumber)) {
        // If we are using the Trading input mode and the note is in the dictionary, add a token to the queue
        const double onsetTime = notesOnToSend[midiNoteNumber];
        Token inputToken = {
            static_cast<int32_t>(onsetTime), static_cast<int32_t>(Vocab::DurOffset + (time - onsetTime)),
            static_cast<int32_t>(Vocab::NoteOffset + Config::MaxPitch * modelConfig.inputInstrument +
                                 midiNoteNumber)
        };
        tokensToSend.push_back(inputToken);
        notesOnToSend.erase(midiNoteNumber);
    }
}

void MainComponent::updateMtcClock() {
    mtcClockActive = mtcClock.getToggleState();

    auto list = juce::MidiInput::getAvailableDevices();
    auto index = mtcClockList.getSelectedItemIndex();

    deviceManager.removeMidiInputDeviceCallback(list[lastMtcClockIndex].identifier, this);

    lastMtcClockIndex = index;

    if (mtcClockActive) {
        auto newMtcClock = list[index];

        if (!deviceManager.isMidiInputDeviceEnabled(newMtcClock.identifier))
            deviceManager.setMidiInputDeviceEnabled(newMtcClock.identifier, true);

        deviceManager.addMidiInputDeviceCallback(newMtcClock.identifier, this);

        selectedMtcClockIdentifier = newMtcClock.identifier;
        clock.setMtcOffset(mtcClockOffset.getText().getIntValue());
    }
}

void MainComponent::changeListenerCallback(ChangeBroadcaster *source) {
    resized();
    markPresetAsEdited();
}

bool MainComponent::keyPressed(const KeyPress &key) {
    if (key.getKeyCode() == juce::KeyPress::leftKey) {
        moveToLeftPreset();
    } else if (key.getKeyCode() == juce::KeyPress::rightKey) {
        moveToRightPreset();
    }
}


