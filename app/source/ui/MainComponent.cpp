#include "VirtualOrch/ui/MainComponent.h"

#include <iostream>

#include "VirtualOrch/MidiOutputProcessor.h"
#include "VirtualOrch/OSCOutputProcessor.h"
#include "VirtualOrch/OSCController.h"

//==============================================================================
MainComponent::MainComponent(AppSession &sessionIn) : session(sessionIn) {
    setOpaque(true);

    setWantsKeyboardFocus(true);

    session.presetStore.setModelNameProvider([this] {
        if (modelList.getSelectedId() == 0) {
            return juce::String();
        }
        return modelList.getItemText(modelList.getSelectedItemIndex());
    });
    session.presetStore.setOnPresetSaved([this](const juce::String &presetName) {
        presetList.addItem(presetName, presetList.getNumItems() + 1);
        presetList.setSelectedId(presetList.getNumItems(), juce::dontSendNotification);
        savePresetButton.setEnabled(false);
    });
    session.presetStore.loadSettings();

    // Get displays (used to place Transport on an external display when available)
    auto const &displays = Desktop::getInstance().getDisplays().displays;

    midiInputs = juce::MidiInput::getAvailableDevices();
    hardwareMidiOutputs = juce::MidiOutput::getAvailableDevices();

    // FOR DEBUG ONLY
    virtualMidiInput = juce::MidiInput::createNewDevice("virtual-orch Virtual MIDI Input", &session.midiInputProcess);
    virtualMidiInput->start();

    /* PRESET LIST */
    addAndMakeVisible(presetListLabel);
    presetListLabel.setText("Preset: ", juce::dontSendNotification);
    presetListLabel.attachToComponent(&presetList, true);

    addAndMakeVisible(presetList);
    presetList.addItem("Empty", 1);
    {
        const auto presetNames = session.presetStore.listPresetNames();
        for (int i = 0; i < presetNames.size(); ++i) {
            presetList.addItem(presetNames[i], i + 2);
        }
    }
    presetList.onChange = [this] {
        if (presetList.getSelectedId() == 1) {
            clearModel();
        } else {
            juce::String presetName = presetList.getItemText(presetList.getSelectedItemIndex());
            loadPresetFromName(presetName);
        }
        savePresetButton.setEnabled(false);
    };
    presetList.setSelectedId(1);

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
            presetSaveDialog = new PresetSaveDialog(&session.presetStore);
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
    juce::String savedControllerOscPort = session.presetStore.settings.getOrCreateChildWithName("controller", nullptr).
            getProperty("oscPort", "9001");
    controllerOscPort.setText(savedControllerOscPort);
    controllerOscPort.setInputRestrictions(6, "0123456789");
    controllerOscPort.onTextChange = [this] {
        controllerConnectButton.setEnabled(true);
        DBG("Saving Preset Controller OSC Port: " + controllerOscPort.getText());
        session.presetStore.settings.getChildWithName("controller").
                setProperty("oscPort", controllerOscPort.getText(), nullptr);
    };

    addAndMakeVisible(controllerConnectButton);
    controllerConnectButton.onClick = [this] {
        /* Set controller */
        controller = std::make_unique<OSCController>(controllerOscPort.getText().getIntValue());
        controller->onPresetChanged = [this](const juce::String &presetName) {
            loadPresetFromName(presetName);
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
        controller->onSetOutputRange = [this](const juce::int32 &id, const juce::int32 &low, const juce::int32 &high) {
            session.modelConfig.outputInstruments[id].low = low;
            session.modelConfig.outputInstruments[id].high = high;
        };
        controller->onSetModelConfig = [this](const juce::OSCMessage &message) {
            if (message.size() == 2) {
                if (message[1].getType() == juce::OSCTypes::string) {
                    session.modelConfig.updateParameter(message[0].getString(), message[1].getString());
                } else if (message[1].getType() == juce::OSCTypes::int32) {
                    session.modelConfig.updateParameter(message[0].getString(), message[1].getInt32());
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
                session.modelConfig.updateParameter(message[0].getString(), message[1].getInt32(),
                                            message[2].getInt32());
            }
            session.rebuildInputFilter();
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
        session.presetStore.settings.getChildWithName("mtcClock").setProperty(
            "active", mtcClock.getToggleState(), nullptr);
    };
    bool savedMtcClockStatus = session.presetStore.settings.getOrCreateChildWithName("mtcClock", nullptr).getProperty("active", false);
    mtcClock.setToggleState(savedMtcClockStatus, juce::sendNotification);

    juce::StringArray midiInputNames;

    for (auto input: midiInputs)
        midiInputNames.add(input.name);

    addAndMakeVisible(mtcClockList);
    mtcClockList.setTextWhenNoChoicesAvailable("No MTC Clocks Enabled");
    mtcClockList.addItemList(midiInputNames, 1);
    mtcClockList.onChange = [this] {
        updateMtcClock();
        DBG("Saving MTC Clock: " + midiInputs[mtcClockList.getSelectedItemIndex()].identifier);
        session.presetStore.settings.getChildWithName("mtcClock").setProperty(
            "identifier", midiInputs[mtcClockList.getSelectedItemIndex()].identifier, nullptr);
    };
    juce::String savedMtcClock = session.presetStore.settings.getOrCreateChildWithName("mtcClock", nullptr).getProperty("identifier", "");
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
            mtcClockList.setSelectedItemIndex(index);
        } else {
            DBG("Saved MTC Clock not found, setting to first input.");
            mtcClockList.setSelectedItemIndex(0);
        }
    } else {
        DBG("No saved MTC Clock, setting to first input.");
        mtcClockList.setSelectedItemIndex(0);
    }

    addAndMakeVisible(mtcClockOffsetLabel);
    mtcClockOffsetLabel.setText("Offset: ", juce::dontSendNotification);
    mtcClockOffsetLabel.attachToComponent(&mtcClockOffset, true);

    addAndMakeVisible(mtcClockOffset);
    juce::String savedMtcClockOffset = session.presetStore.settings.getOrCreateChildWithName("mtcClock", nullptr).
            getProperty("offset", "0");
    mtcClockOffset.setInputRestrictions(6, "0123456789");
    mtcClockOffset.onTextChange = [this] {
        updateMtcClock();
        DBG("Saving MTC Clock Offset: " + mtcClockOffset.getText());
        session.presetStore.settings.getChildWithName("mtcClock").setProperty(
            "offset", mtcClockOffset.getText(), nullptr);
    };
    mtcClockOffset.setText(savedMtcClockOffset);

    /* MIDI INPUT LIST */

    addAndMakeVisible(midiInputListLabel);
    midiInputListLabel.setText("MIDI Input: ", juce::dontSendNotification);
    midiInputListLabel.attachToComponent(&midiInputList, true);

    addAndMakeVisible(midiInputList);
    midiInputList.setTextWhenNoChoicesAvailable("No MIDI Inputs Enabled");

    midiInputList.addItemList(midiInputNames, 1);
    midiInputList.onChange = [this] {
        if (midiInputList.getSelectedItemIndex() == 0) {
            return;
        }
        setMidiInput(midiInputList.getSelectedItemIndex(), 0);
        DBG("Saving input: " + midiInputs[midiInputList.getSelectedItemIndex()].identifier);
        session.presetStore.settings.getChildWithName("input").setProperty(
            "identifier", midiInputs[midiInputList.getSelectedItemIndex()].identifier, nullptr);
    };
    juce::String savedInput = session.presetStore.settings.getOrCreateChildWithName("input", nullptr).getProperty("identifier", "");
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
            midiInputList.setSelectedItemIndex(index);
        } else {
            DBG("Saved input not found, setting to first input.");
            midiInputList.setSelectedItemIndex(0);
        }
    } else {
        DBG("No saved input, setting to first input.");
        midiInputList.setSelectedItemIndex(0);
    }

    /* LAUNCHPAD MIDI LIST */

    addAndMakeVisible(launchpadMidiListLabel);
    launchpadMidiListLabel.setText("Launchpad MIDI: ", juce::dontSendNotification);
    launchpadMidiListLabel.attachToComponent(&launchpadMidiList, true);

    addAndMakeVisible(launchpadMidiList);
    launchpadMidiList.setTextWhenNoChoicesAvailable("No MIDI Inputs Enabled");

    launchpadMidiList.addItemList(midiInputNames, 1);
    launchpadMidiList.onChange = [this] {
        if (launchpadMidiList.getSelectedItemIndex() == 0) {
            return;
        }
        setMidiInput(launchpadMidiList.getSelectedItemIndex(), 1);
        DBG("Saving launchpad input: " + midiInputs[launchpadMidiList.getSelectedItemIndex()].identifier);
        session.presetStore.settings.getOrCreateChildWithName("launchpadInput", nullptr).setProperty(
            "identifier", midiInputs[launchpadMidiList.getSelectedItemIndex()].identifier, nullptr);
    };
    juce::String savedLaunchpadInput =
        session.presetStore.settings.getOrCreateChildWithName("launchpadInput", nullptr).getProperty("identifier", "");
    if (savedLaunchpadInput.isNotEmpty()) {
        DBG("Saved Launchpad Input: " + savedLaunchpadInput);
        int index = -1;
        for (int i = 0; i < midiInputs.size(); i++) {
            if (midiInputs[i].identifier == savedLaunchpadInput) {
                index = i;
                break;
            }
        }
        if (index != -1) {
            DBG("Set to saved launchpad input.");
            launchpadMidiList.setSelectedItemIndex(index);
        } else {
            DBG("Saved launchpad input not found, setting to first input.");
            launchpadMidiList.setSelectedItemIndex(0);
        }
    } else {
        DBG("No saved launchpad input, setting to first input.");
        launchpadMidiList.setSelectedItemIndex(0);
    }

    /* MODEL LIST */

    addAndMakeVisible(modelListLabel);
    modelListLabel.setText("Model: ", juce::dontSendNotification);
    modelListLabel.attachToComponent(&modelList, true);

    addAndMakeVisible(modelList);
    modelList.setTextWhenNoChoicesAvailable("No Models Available");
    juce::File modelsDir = juce::File::getSpecialLocation(juce::File::SpecialLocationType::userDocumentsDirectory)
            .getChildFile("virtual-orch")
            .getChildFile("Models");

    juce::Array<juce::File> modelFiles;
    modelsDir.findChildFiles(modelFiles, juce::File::findFiles, false, "*.onnx");

    for (const auto &file: modelFiles) {
        // Don't add if associated .json doesn't exist
        if (!modelsDir.getChildFile(file.getFileNameWithoutExtension() + ".json").exists()) {
            continue;
        }

        modelList.addItem(file.getFileNameWithoutExtension(), modelFiles.indexOf(file) + 1);
    }

    modelList.onChange = [this] {
        stop();
        updateModel(true);
    };

    addAndMakeVisible(modelConfigurationButton);
    modelConfigurationButton.onClick = [this] {
        if (modelConfigurationWindow) {
            modelConfigurationWindow->toFront(true);
        } else {
            modelConfigurationWindow = new ScrollableWindow(new ModelConfigurationComponent(session.modelConfig), this, 700,
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
    bool savedInputThru = session.presetStore.settings.getOrCreateChildWithName("input", nullptr).getProperty("thru", false);
    inputThru.setToggleState(savedInputThru, juce::dontSendNotification);
    session.midiInputProcess.setInputThru(savedInputThru);
    inputThru.onClick = [this] {
        DBG("Saving input thru: " + std::to_string(inputThru.getToggleState()));
        session.presetStore.settings.getChildWithName("input").setProperty("thru", inputThru.getToggleState(), nullptr);
        session.midiInputProcess.setInputThru(inputThru.getToggleState());
    };


    /* OUTPUT LIST */

    addAndMakeVisible(outputListLabel);
    outputListLabel.setText("Output: ", juce::dontSendNotification);
    outputListLabel.attachToComponent(&outputList, true);

    std::vector<juce::String> outputIdentifiers;

    addAndMakeVisible(outputList);
    outputList.addItem("OSC", 1);
    outputIdentifiers.push_back("OSC");
    outputList.addItem("Virtual MIDI", 2);
    outputIdentifiers.push_back("Virtual MIDI");

    juce::StringArray midiOutputNames;
    for (const auto &output: hardwareMidiOutputs) {
        midiOutputNames.add(output.name);
        outputIdentifiers.push_back(output.identifier);
    }
    outputList.addItemList(midiOutputNames, 3);

    outputList.onChange = [this, outputIdentifiers] {
        updateOutputProcessor();
        DBG("Saving output: " + outputIdentifiers.at(outputList.getSelectedItemIndex()));
        session.presetStore.settings.getChildWithName("output").setProperty(
            "identifier", outputIdentifiers.at(outputList.getSelectedItemIndex()), nullptr);
    };
    juce::String savedOutput = session.presetStore.settings.getOrCreateChildWithName("output", nullptr).getProperty("identifier", "");
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
            outputList.setSelectedItemIndex(index);
        } else {
            DBG("Saved output not found, setting to first output.");
            outputList.setSelectedItemIndex(0);
        }
    } else {
        DBG("No saved output, setting to first output.");
        outputList.setSelectedItemIndex(0);
    }

    /* OSC IP */

    addAndMakeVisible(oscIpLabel);
    oscIpLabel.setText("OSC IP: ", juce::dontSendNotification);
    oscIpLabel.attachToComponent(&oscIp, true);

    addAndMakeVisible(oscIp);
    juce::String savedOscIp = session.presetStore.settings.getOrCreateChildWithName("output", nullptr).getProperty("oscIp", "127.0.0.1");
    oscIp.setText(savedOscIp);
    oscIp.setInputRestrictions(15, "0123456789.");

    oscIp.onTextChange = [this] {
        connectButton.setEnabled(true);
        DBG("Saving OSC IP: " + oscIp.getText());
        session.presetStore.settings.getChildWithName("output").setProperty("oscIp", oscIp.getText(), nullptr);
    };

    addAndMakeVisible(oscPortLabel);
    oscPortLabel.setText(": ", juce::dontSendNotification);
    oscPortLabel.attachToComponent(&oscPort, true);

    addAndMakeVisible(oscPort);
    juce::String savedOscPort = session.presetStore.settings.getOrCreateChildWithName("output", nullptr).getProperty("oscPort", "9001");
    oscPort.setText(savedOscPort);
    oscPort.setInputRestrictions(6, "0123456789");

    oscPort.onTextChange = [this] {
        connectButton.setEnabled(true);
        DBG("Saving OSC Port: " + oscPort.getText());
        session.presetStore.settings.getChildWithName("output").setProperty("oscPort", oscPort.getText(), nullptr);
    };




    addAndMakeVisible(connectButton);
    connectButton.onClick = [this] {
        session.outputProcessor = std::make_unique<OSCOutputProcessor>(oscIp.getText().toStdString(),
                                                               oscPort.getText().getIntValue());
        connectButton.setEnabled(false);
    };

    /* BUFFER OUTPUT OSC IP */

    addAndMakeVisible(bufferOutputOscIpLabel);
    bufferOutputOscIpLabel.setText("Buffer Output OSC IP: ", juce::dontSendNotification);
    bufferOutputOscIpLabel.attachToComponent(&bufferOutputOscIp, true);

    addAndMakeVisible(bufferOutputOscIp);
    juce::String savedBufferOutputOscIp = session.presetStore.settings.getOrCreateChildWithName("bufferOutput", nullptr)
            .getProperty("oscIp", "127.0.0.1");
    bufferOutputOscIp.setText(savedBufferOutputOscIp);
    bufferOutputOscIp.setInputRestrictions(15, "0123456789.");

    bufferOutputOscIp.onTextChange = [this] {
        bufferOutputConnectButton.setEnabled(true);
        DBG("Saving Buffer Output OSC IP: " + bufferOutputOscIp.getText());
        session.presetStore.settings.getChildWithName("bufferOutput").setProperty("oscIp", bufferOutputOscIp.getText(), nullptr);
    };

    addAndMakeVisible(bufferOutputOscPortLabel);
    bufferOutputOscPortLabel.setText(": ", juce::dontSendNotification);
    bufferOutputOscPortLabel.attachToComponent(&bufferOutputOscPort, true);

    addAndMakeVisible(bufferOutputOscPort);
    juce::String savedBufferOutputOscPort = session.presetStore.settings.getOrCreateChildWithName("bufferOutput", nullptr).getProperty(
        "oscPort", "9001");
    bufferOutputOscPort.setText(savedBufferOutputOscPort);
    bufferOutputOscPort.setInputRestrictions(6, "0123456789");

    bufferOutputOscPort.onTextChange = [this] {
        bufferOutputConnectButton.setEnabled(true);
        DBG("Saving Buffer Output OSC Port: " + bufferOutputOscPort.getText());
        session.presetStore.settings.getChildWithName("bufferOutput").setProperty("oscPort", bufferOutputOscPort.getText(), nullptr);
    };

    addAndMakeVisible(bufferOutputConnectButton);
    bufferOutputConnectButton.onClick = [this] {
        session.bufferOutputProcessor = std::make_unique<OSCBufferOutputProcessor>(bufferOutputOscIp.getText().toStdString(),
                                                                           bufferOutputOscPort.getText().getIntValue());
        bufferOutputConnectButton.setEnabled(false);
    };

    /* STATUS OUTPUT OSC IP */

    addAndMakeVisible(statusOutputOscIpLabel);
    statusOutputOscIpLabel.setText("Status Output OSC IP: ", juce::dontSendNotification);
    statusOutputOscIpLabel.attachToComponent(&statusOutputOscIp, true);

    addAndMakeVisible(statusOutputOscIp);
    juce::String savedStatusOutputOscIp = session.presetStore.settings.getOrCreateChildWithName("statusOutput", nullptr)
            .getProperty("oscIp", "127.0.0.1");
    statusOutputOscIp.setText(savedStatusOutputOscIp);
    statusOutputOscIp.setInputRestrictions(15, "0123456789.");

    statusOutputOscIp.onTextChange = [this] {
        statusOutputConnectButton.setEnabled(true);
        DBG("Saving Status Output OSC IP: " + statusOutputOscIp.getText());
        session.presetStore.settings.getChildWithName("statusOutput").setProperty("oscIp", statusOutputOscIp.getText(), nullptr);
    };

    addAndMakeVisible(statusOutputOscPortLabel);
    statusOutputOscPortLabel.setText(": ", juce::dontSendNotification);
    statusOutputOscPortLabel.attachToComponent(&statusOutputOscPort, true);

    addAndMakeVisible(statusOutputOscPort);
    juce::String savedStatusOutputOscPort = session.presetStore.settings.getOrCreateChildWithName("statusOutput", nullptr).getProperty(
        "oscPort", "9001");
    statusOutputOscPort.setText(savedStatusOutputOscPort);
    statusOutputOscPort.setInputRestrictions(6, "0123456789");

    statusOutputOscPort.onTextChange = [this] {
        statusOutputConnectButton.setEnabled(true);
        DBG("Saving Status Output OSC Port: " + statusOutputOscPort.getText());
        session.presetStore.settings.getChildWithName("statusOutput").setProperty("oscPort", statusOutputOscPort.getText(), nullptr);
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
    generationStatusProgressBar = std::make_unique<juce::ProgressBar>(session.outputPlayback.progress);
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
                .getChildFile("virtual-orch")
                .getChildFile("last_gen.txt");

        // write session.musicTransformer's inputData into file
        auto outputStream = lastGenFile.createOutputStream();
        if (outputStream->openedOk()) {
            outputStream->setPosition(0);
            outputStream->truncate();
            for (const auto &token: session.musicTransformer.getInputData()) {
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
            transportWindow = new TransportComponent(displays.size() >= 2, session.clock);
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
            metricsWindow = new MetricsComponent(session.metrics);
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
    bool savedAutoConnect = session.presetStore.settings.getOrCreateChildWithName("general", nullptr).getProperty("autoConnect", false);
    autoConnect.setToggleState(savedAutoConnect, juce::dontSendNotification);
    autoConnect.onClick = [this] {
        DBG("Saving Auto Connect: " + std::to_string(autoConnect.getToggleState()));
        session.presetStore.settings.getChildWithName("general").setProperty("autoConnect", autoConnect.getToggleState(), nullptr);
    };
    if (autoConnect.getToggleState()) {
        connectButton.triggerClick();
        controllerConnectButton.triggerClick();
        bufferOutputConnectButton.triggerClick();
        statusOutputConnectButton.triggerClick();
    }

    // Visualization buffer size for SAM Hackathon
    addAndMakeVisible(visualizationBufferSizeLabel);
    visualizationBufferSizeLabel.setText("Visualization Buffer Size: ", juce::dontSendNotification);
    visualizationBufferSizeLabel.attachToComponent(&visualizationBufferSizeEditor, true);

    addAndMakeVisible(visualizationBufferSizeEditor);
    juce::String savedBufferSize = session.presetStore.settings.getOrCreateChildWithName("output", nullptr)
                                       .getProperty("visualizationBufferSize", "512");
    visualizationBufferSizeEditor.setText(savedBufferSize);
    visualizationBufferSizeEditor.setInputRestrictions(6, "0123456789");

    visualizationBufferSizeEditor.onTextChange = [this] {

        juce::String text = visualizationBufferSizeEditor.getText();
        int newSize = text.getIntValue();  // safely parse to int
        session.visualizationBufferSize = newSize; // <-- keep your variable updated

        DBG("Saving Visualization Buffer Size: " + text);
        session.presetStore.settings.getChildWithName("output")
                .setProperty("visualizationBufferSize", text, nullptr);
    };

    setSize(700, 800);



}

MainComponent::~MainComponent() {
    session.presetStore.saveSettings();
    deviceManager.removeMidiInputDeviceCallback(
        juce::MidiInput::getAvailableDevices()[midiInputList.getSelectedItemIndex()].identifier, &session.midiInputProcess);
    generationStatusProgressBar->setLookAndFeel(nullptr);
    delete presetSaveDialog;
    delete modelConfigurationWindow;
    delete transportWindow;
    delete metricsWindow;
}



/**
 * Loads both the model and the preset from a given name (if it exists)
 * @param presetName The name of the preset to load
 */
void MainComponent::loadPresetFromName(const juce::String &presetName) {
    const juce::var preset = session.presetStore.readPreset(presetName);
    if (preset.isVoid()) {
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

    const juce::String modelName = preset.getProperty("model", "");
    if (modelName.isEmpty()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "Preset is invalid (model not given).");
        presetList.setSelectedId(1, juce::dontSendNotification);
        return;
    }
    if (modelName != modelList.getItemText(modelList.getSelectedItemIndex())) {
        int modelIndex = -1;
        for (int i = 0; i < modelList.getNumItems(); i++) {
            if (modelList.getItemText(i) == modelName) {
                modelIndex = i;
                break;
            }
        }
        if (modelIndex == -1) {
            juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                                   "Model " + modelName + " not found.");
            presetList.setSelectedId(1, juce::dontSendNotification);
            return;
        }
        stop();

        modelList.setSelectedId(modelIndex + 1, juce::dontSendNotification);
        updateModel(false);
    }
    session.presetStore.applyPreset(preset);
    session.rebuildInputFilter();
    resized();
}

void MainComponent::moveToLeftPreset() {
    int index = presetList.getSelectedItemIndex();
    if (index == 0) {
        index = presetList.getNumItems() - 1;
    } else {
        index--;
    }
    presetList.setSelectedId(index + 1, juce::sendNotification);
}

void MainComponent::moveToRightPreset() {
    int index = presetList.getSelectedItemIndex();
    if (index == presetList.getNumItems() - 1) {
        index = 0;
    } else {
        index++;
    }
    presetList.setSelectedId(index + 1, juce::sendNotification);
}


void MainComponent::clearModel() {
    modelList.setSelectedId(0, juce::dontSendNotification);
    stop();
    // We resize to remove the `modelConfigurationButton`
    modelConfigurationButton.setEnabled(false);
    resized();
}

void MainComponent::updateModel(const bool loadDefaultPreset) {
    // Send modelLoading status to statusOutputProcessor
    if (statusOutputProcessor != nullptr) {
        statusOutputProcessor->modelLoading();
    }

    // Get Model Config File
    juce::File modelsDir = juce::File::getSpecialLocation(juce::File::SpecialLocationType::userDocumentsDirectory)
            .getChildFile("virtual-orch")
            .getChildFile("Models");
    const auto modelName = modelList.getItemText(modelList.getSelectedItemIndex());
    juce::File jsonFile(modelsDir.getChildFile(modelName + ".json"));
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
        modelList.setSelectedId(0, juce::dontSendNotification);


        return;
    } else {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "Model Config is invalid (size '" + modelSize +
                                               "' is not valid).");
        modelList.setSelectedId(0, juce::dontSendNotification);
        return;
    }

    // MODEL CONFIG: Set Output Instruments
    session.modelConfig.outputInstruments.clear();
    auto *jsonOutputInstruments = parsedJson.getProperty("outputInstruments", var()).getArray();
    if (jsonOutputInstruments && !jsonOutputInstruments->isEmpty()) {
        for (int32_t instrumentId: *jsonOutputInstruments) {
            // Add default instrument config
            OutputInstrumentConfig outputInstrumentConfig{false, false, 36, 120};
            session.modelConfig.outputInstruments.emplace(instrumentId, outputInstrumentConfig);
        }
    } else {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "Model Config is invalid (no output instrument).");
        modelList.setSelectedId(0, juce::dontSendNotification);
        return;
    }

    // Load preset
    if (loadDefaultPreset) {
        DBG("Loading default preset");
        session.presetStore.applyPreset(parsedJson.getProperty("defaultPreset", var()));
        session.rebuildInputFilter();
        resized();
        // We are changing the model, so mark preset as modified
        markPresetAsEdited();
    }

    const auto modelPath = modelsDir.getChildFile(modelName + ".onnx");
    session.musicTransformer.init(modelPath.getFullPathName().toStdString().c_str(), modelType);

    // Send modelLoaded status to statusOutputProcessor
    if (statusOutputProcessor != nullptr) {
        statusOutputProcessor->modelLoaded();
    }

    modelConfigurationButton.setEnabled(true);

    // Update the OuputProcessor with the new instruments
    updateOutputProcessor();
}

void MainComponent::start() {
    session.startGeneration();
}

void MainComponent::stop() {
    session.stopGeneration();
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
    connectButton.setVisible(false);

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

    auto launchpadMidiArea = area.removeFromTop(36).removeFromRight(getWidth() - 150).reduced(8);
    launchpadMidiList.setBounds(launchpadMidiArea);

    auto modelConfigArea = area.removeFromTop(36).removeFromRight(getWidth() - 150).reduced(8);
    modelList.setBounds(modelConfigArea.removeFromLeft(getWidth() - 300).withTrimmedRight(20));
    modelConfigurationButton.setBounds(modelConfigArea);

    outputList.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 150).reduced(8));

    if (enableOscConfig) {
        oscIp.setVisible(true);
        oscPort.setVisible(true);
        connectButton.setVisible(true);
        auto oscConfigArea = area.removeFromTop(36).removeFromRight(getWidth() - 150).reduced(8);
        oscIp.setBounds(oscConfigArea.removeFromLeft(getWidth() - 350).withTrimmedRight(20));
        oscPort.setBounds(oscConfigArea.removeFromLeft(60).withTrimmedRight(10));
        connectButton.setBounds(oscConfigArea);
    }

    auto bufferOutputOscConfigArea = area.removeFromTop(36).removeFromRight(getWidth() - 150).reduced(8);
    bufferOutputOscIp.setBounds(bufferOutputOscConfigArea.removeFromLeft(getWidth() - 350).withTrimmedRight(20));
    bufferOutputOscPort.setBounds(bufferOutputOscConfigArea.removeFromLeft(60).withTrimmedRight(10));
    bufferOutputConnectButton.setBounds(bufferOutputOscConfigArea);

    auto statusOutputOscConfigArea = area.removeFromTop(36).removeFromRight(getWidth() - 150).reduced(8);
    statusOutputOscIp.setBounds(statusOutputOscConfigArea.removeFromLeft(getWidth() - 350).withTrimmedRight(20));
    statusOutputOscPort.setBounds(statusOutputOscConfigArea.removeFromLeft(60).withTrimmedRight(10));
    statusOutputConnectButton.setBounds(statusOutputOscConfigArea);

    auto visBufArea = area.removeFromTop(48).removeFromRight(getWidth() - 150).reduced(8);
    visualizationBufferSizeLabel.setBounds(visBufArea.removeFromLeft(getWidth() - 350).withTrimmedRight(20));
    visualizationBufferSizeEditor.setBounds(visBufArea.removeFromLeft(60).withTrimmedRight(10));

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
    session.outputProcessor = nullptr;
    switch (outputList.getSelectedItemIndex()) {
        case 0: // OSC
            connectButton.setEnabled(true);
            enableOscConfig = true;
            break;
        case 1: // Virtual MIDI
            session.outputProcessor = std::make_unique<
                MidiOutputProcessor>(MidiOutputType::VIRTUAL, "virtual-orch Virtual MIDI Output",
                                     session.modelConfig.getOutputInstrumentsIds());

            enableOscConfig = false;
            break;
        default: // Hardware MIDI
            session.outputProcessor = std::make_unique<MidiOutputProcessor>(MidiOutputType::HARDWARE,
                                                                    hardwareMidiOutputs[
                                                                        outputList.getSelectedItemIndex() - 2].
                                                                    identifier,
                                                                    session.modelConfig.getOutputInstrumentsIds());

            enableOscConfig = false;
            break;
    }
    resized();
};

void MainComponent::setMidiInput(int index, int idx) {
    auto list = juce::MidiInput::getAvailableDevices();
    if (! juce::isPositiveAndBelow(index, list.size()))
        return;

    auto newInput = list[index];
    auto &selectedForRole = (idx == 0) ? session.selectedMidiInputIdentifier
                                       : session.selectedLaunchpadMidiIdentifier;
    const auto &otherRole = (idx == 0) ? session.selectedLaunchpadMidiIdentifier
                                       : session.selectedMidiInputIdentifier;

    // Drop the previous device for this role unless the other role still needs it.
    if (selectedForRole.isNotEmpty()
        && selectedForRole != newInput.identifier
        && selectedForRole != otherRole) {
        deviceManager.removeMidiInputDeviceCallback(selectedForRole, &session.midiInputProcess);
    }

    if (!deviceManager.isMidiInputDeviceEnabled(newInput.identifier))
        deviceManager.setMidiInputDeviceEnabled(newInput.identifier, true);

    deviceManager.addMidiInputDeviceCallback(newInput.identifier, &session.midiInputProcess);
    selectedForRole = newInput.identifier;

    if (idx == 1) {
        session.midiInputProcess.setLaunchpadMidiOutputForInputDevice(newInput);
    }
}

void MainComponent::updateMtcClock() {
    session.mtcClockActive = mtcClock.getToggleState();

    auto list = juce::MidiInput::getAvailableDevices();
    auto index = mtcClockList.getSelectedItemIndex();

    deviceManager.removeMidiInputDeviceCallback(list[lastMtcClockIndex].identifier, &session.midiInputProcess);

    lastMtcClockIndex = index;

    if (session.mtcClockActive) {
        auto newMtcClock = list[index];

        if (!deviceManager.isMidiInputDeviceEnabled(newMtcClock.identifier))
            deviceManager.setMidiInputDeviceEnabled(newMtcClock.identifier, true);

        deviceManager.addMidiInputDeviceCallback(newMtcClock.identifier, &session.midiInputProcess);

        session.selectedMtcClockIdentifier = newMtcClock.identifier;
        session.clock.setMtcOffset(mtcClockOffset.getText().getIntValue());
    }
}

void MainComponent::changeListenerCallback(ChangeBroadcaster *source) {
    juce::ignoreUnused(source);
    session.rebuildInputFilter();
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

