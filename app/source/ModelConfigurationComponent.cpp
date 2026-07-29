#include "VirtualOrch/ModelConfigurationComponent.h"


ModelConfigurationComponent::ModelConfigurationComponent(ModelConfig &modelConfig) : modelConfig(modelConfig) {
    Component::setName("Model Configuration");
    setOpaque(true);

    /* INPUT */
    addAndMakeVisible(inputTitle);
    inputTitle.setText("Input Configuration", juce::dontSendNotification);
    inputTitle.setFont(juce::Font(20.0f, juce::Font::bold));

    /* INPUT MODE */
    addAndMakeVisible(inputModeLabel);
    inputModeLabel.setText("Input Mode:", juce::dontSendNotification);
    inputModeLabel.attachToComponent(&inputMode, true);

    addAndMakeVisible(inputMode);
    inputMode.addItem("Direct", InputMode::Direct + 1);
    inputMode.addItem("Buffer", InputMode::Buffer + 1);
    inputMode.setSelectedId(modelConfig.inputMode + 1, juce::dontSendNotification);
    inputMode.onChange = [this] {
        sendChangeMessage();

        // Resize to show new parameters
        resized();
    };

    /* INPUT INSTRUMENT */
    addAndMakeVisible(inputInstrumentLabel);
    inputInstrumentLabel.setText("Input Instrument:", juce::dontSendNotification);
    inputInstrumentLabel.attachToComponent(&inputInstrument, true);

    addAndMakeVisible(inputInstrument);
    inputInstrument.setInputRestrictions(10, "0123456789");
    inputInstrument.setText(juce::String(modelConfig.inputInstrument));
    inputInstrument.onTextChange = [this] {
        sendChangeMessage();
    };

    /* INPUT RANGE */
    addAndMakeVisible(inputLowLabel);
    inputLowLabel.setText("Input Range:", juce::dontSendNotification);
    inputLowLabel.attachToComponent(&inputLow, true);

    addAndMakeVisible(inputLow);
    inputLow.setInputRestrictions(10, "0123456789");
    inputLow.setText(juce::String(modelConfig.inputLow));
    inputLow.onTextChange = [this] {
        sendChangeMessage();
    };

    addAndMakeVisible(inputHighLabel);
    inputHighLabel.setText("- ", juce::dontSendNotification);
    inputHighLabel.attachToComponent(&inputHigh, true);

    addAndMakeVisible(inputHigh);
    inputHigh.setInputRestrictions(10, "0123456789");
    inputHigh.setText(juce::String(modelConfig.inputHigh));
    inputHigh.onTextChange = [this] {
        sendChangeMessage();
    };

    /* INPUT DURATION */
    addAndMakeVisible(inputDurationLabel);
    inputDurationLabel.setText("Input Duration:", juce::dontSendNotification);
    inputDurationLabel.attachToComponent(&inputDuration, true);

    addAndMakeVisible(inputDuration);
    inputDuration.setInputRestrictions(10, "0123456789");
    inputDuration.setText(juce::String(modelConfig.inputDuration));
    inputDuration.onTextChange = [this] {
        sendChangeMessage();
    };

    /* INPUT INITIAL DATA */
    addAndMakeVisible(inputInitialDataLabel);
    inputInitialDataLabel.setText("Input Initial Data:", juce::dontSendNotification);
    inputInitialDataLabel.attachToComponent(&inputInitialData, true);

    addAndMakeVisible(inputInitialData);
    inputInitialData.setInputRestrictions(10000, "0123456789 ");
    inputInitialData.setText(modelConfig.inputInitialData);
    inputInitialData.onTextChange = [this] {
        sendChangeMessage();
    };

    /* DIRECT INPUT WINDOW LENGTH */
    addAndMakeVisible(directInputWindowLengthLabel);
    directInputWindowLengthLabel.setText("Window Length (1/100s):", juce::dontSendNotification);
    directInputWindowLengthLabel.attachToComponent(&directInputWindowLength, true);

    addAndMakeVisible(directInputWindowLength);
    directInputWindowLength.setInputRestrictions(10, "0123456789");
    directInputWindowLength.setText(juce::String(modelConfig.directInputWindowLength));
    directInputWindowLength.onTextChange = [this] {
        sendChangeMessage();
    };

    /* DIRECT INPUT SEND NOTE OFFS */
    addAndMakeVisible(directInputSendNoteOffsLabel);
    directInputSendNoteOffsLabel.setText("Send Note Offs:", juce::dontSendNotification);
    directInputSendNoteOffsLabel.attachToComponent(&directInputSendNoteOffs, true);

    addAndMakeVisible(directInputSendNoteOffs);
    directInputSendNoteOffs.setToggleState(modelConfig.directInputSendNoteOffs, juce::dontSendNotification);
    directInputSendNoteOffs.onClick = [this] {
        sendChangeMessage();
    };

    /* DIRECT INPUT START ON INPUT */
    addAndMakeVisible(directInputStartOnInputLabel);
    directInputStartOnInputLabel.setText("Start on Input:", juce::dontSendNotification);
    directInputStartOnInputLabel.attachToComponent(&directInputStartOnInput, true);

    addAndMakeVisible(directInputStartOnInput);
    directInputStartOnInput.setToggleState(modelConfig.directInputStartOnInput, juce::dontSendNotification);
    directInputStartOnInput.onClick = [this] {
        sendChangeMessage();
    };

    /* DIRECT INPUT START DELAY */
    addAndMakeVisible(directInputStartDelayLabel);
    directInputStartDelayLabel.setText("Start Delay:", juce::dontSendNotification);
    directInputStartDelayLabel.attachToComponent(&directInputStartDelay, true);

    addAndMakeVisible(directInputStartDelay);
    directInputStartDelay.setInputRestrictions(10, "0123456789");
    directInputStartDelay.setText(juce::String(modelConfig.directInputStartDelay));
    directInputStartDelay.onTextChange = [this] {
        sendChangeMessage();
    };

    /* DIRECT INPUT HOLD BASS */
    addAndMakeVisible(directInputHoldBassLabel);
    directInputHoldBassLabel.setText("Hold Bass:", juce::dontSendNotification);
    directInputHoldBassLabel.attachToComponent(&directInputHoldBass, true);

    addAndMakeVisible(directInputHoldBass);
    directInputHoldBass.setToggleState(modelConfig.directInputHoldBass, juce::dontSendNotification);
    directInputHoldBass.onClick = [this] {
        sendChangeMessage();
    };

    /* DIRECT INPUT BASS RANGE */
    addAndMakeVisible(directInputBassLowLabel);
    directInputBassLowLabel.setText("Bass Range:", juce::dontSendNotification);
    directInputBassLowLabel.attachToComponent(&directInputBassLow, true);

    addAndMakeVisible(directInputBassLow);
    directInputBassLow.setInputRestrictions(10, "0123456789");
    directInputBassLow.setText(juce::String(modelConfig.directInputBassLow));
    directInputBassLow.onTextChange = [this] {
        sendChangeMessage();
    };

    addAndMakeVisible(directInputBassHighLabel);
    directInputBassHighLabel.setText("- ", juce::dontSendNotification);
    directInputBassHighLabel.attachToComponent(&directInputBassHigh, true);

    addAndMakeVisible(directInputBassHigh);
    directInputBassHigh.setInputRestrictions(10, "0123456789");
    directInputBassHigh.setText(juce::String(modelConfig.directInputBassHigh));
    directInputBassHigh.onTextChange = [this] {
        sendChangeMessage();
    };

    /* DIRECT INPUT INITIAL BASS */
    addAndMakeVisible(directInputInitialBassLabel);
    directInputInitialBassLabel.setText("Initial Bass:", juce::dontSendNotification);
    directInputInitialBassLabel.attachToComponent(&directInputInitialBass, true);

    addAndMakeVisible(directInputInitialBass);
    directInputInitialBass.setInputRestrictions(10, "0123456789");
    directInputInitialBass.setText(juce::String(modelConfig.directInputInitialBass));
    directInputInitialBass.onTextChange = [this] {
        sendChangeMessage();
    };

    /* BUFFER INPUT BUFFER SIZE */
    addAndMakeVisible(bufferInputSizeLabel);
    bufferInputSizeLabel.setText("Buffer Size:", juce::dontSendNotification);
    bufferInputSizeLabel.attachToComponent(&bufferInputSize, true);

    addAndMakeVisible(bufferInputSize);
    bufferInputSize.setInputRestrictions(10, "0123456789");
    bufferInputSize.setText(juce::String(modelConfig.bufferInputSize));
    bufferInputSize.onTextChange = [this] {
        sendChangeMessage();
    };

    /* BUFFER INPUT BUFFER RANGE */
    addAndMakeVisible(bufferInputLowLabel);
    bufferInputLowLabel.setText("Buffer Range:", juce::dontSendNotification);
    bufferInputLowLabel.attachToComponent(&bufferInputLow, true);

    addAndMakeVisible(bufferInputLow);
    bufferInputLow.setInputRestrictions(10, "0123456789");
    bufferInputLow.setText(juce::String(modelConfig.bufferInputLow));
    bufferInputLow.onTextChange = [this] {
        sendChangeMessage();
    };

    addAndMakeVisible(bufferInputHighLabel);
    bufferInputHighLabel.setText("- ", juce::dontSendNotification);
    bufferInputHighLabel.attachToComponent(&bufferInputHigh, true);

    addAndMakeVisible(bufferInputHigh);
    bufferInputHigh.setInputRestrictions(10, "0123456789");
    bufferInputHigh.setText(juce::String(modelConfig.bufferInputHigh));
    bufferInputHigh.onTextChange = [this] {
        sendChangeMessage();
    };

    /* BUFFER INPUT BASS RANGE */
    addAndMakeVisible(bufferInputBassLowLabel);
    bufferInputBassLowLabel.setText("Bass Range:", juce::dontSendNotification);
    bufferInputBassLowLabel.attachToComponent(&bufferInputBassLow, true);

    addAndMakeVisible(bufferInputBassLow);
    bufferInputBassLow.setInputRestrictions(10, "0123456789");
    bufferInputBassLow.setText(juce::String(modelConfig.bufferInputBassLow));
    bufferInputBassLow.onTextChange = [this] {
        sendChangeMessage();
    };

    addAndMakeVisible(bufferInputBassHighLabel);
    bufferInputBassHighLabel.setText("- ", juce::dontSendNotification);
    bufferInputBassHighLabel.attachToComponent(&bufferInputBassHigh, true);

    addAndMakeVisible(bufferInputBassHigh);
    bufferInputBassHigh.setInputRestrictions(10, "0123456789");
    bufferInputBassHigh.setText(juce::String(modelConfig.bufferInputBassHigh));
    bufferInputBassHigh.onTextChange = [this] {
        sendChangeMessage();
    };

    /* OUTPUT */
    addAndMakeVisible(outputTitle);
    outputTitle.setText("Output Configuration", juce::dontSendNotification);
    outputTitle.setFont(juce::Font(20.0f, juce::Font::bold));


    /* OUTPUT MINIMUM DURATION */
    addAndMakeVisible(outputMinimumDurationLabel);
    outputMinimumDurationLabel.setText("Output Minimum Duration:", juce::dontSendNotification);
    outputMinimumDurationLabel.attachToComponent(&outputMinimumDuration, true);

    addAndMakeVisible(outputMinimumDuration);
    outputMinimumDuration.setInputRestrictions(10, "0123456789");
    outputMinimumDuration.setText(juce::String(modelConfig.outputMinimumDuration));
    outputMinimumDuration.onTextChange = [this] {
        sendChangeMessage();
    };

    /* OUTPUT MAXIMUM DURATION */
    addAndMakeVisible(outputMaximumDurationLabel);
    outputMaximumDurationLabel.setText("Output Maximum Duration:", juce::dontSendNotification);
    outputMaximumDurationLabel.attachToComponent(&outputMaximumDuration, true);

    addAndMakeVisible(outputMaximumDuration);
    outputMaximumDuration.setInputRestrictions(10, "0123456789");
    outputMaximumDuration.setText(juce::String(modelConfig.outputMaximumDuration));
    outputMaximumDuration.onTextChange = [this] {
        sendChangeMessage();
    };


    /* OUTPUT TEMPERATURES */
    addAndMakeVisible(outputTemperatureTimeLabel);
    outputTemperatureTimeLabel.setText("Output Temperatures:    Time:", juce::dontSendNotification);
    outputTemperatureTimeLabel.attachToComponent(&outputTemperatureTime, true);
    addAndMakeVisible(outputTemperatureDurationLabel);
    outputTemperatureDurationLabel.setText("Duration:", juce::dontSendNotification);
    outputTemperatureDurationLabel.attachToComponent(&outputTemperatureDuration, true);
    addAndMakeVisible(outputTemperatureNoteLabel);
    outputTemperatureNoteLabel.setText("Note:", juce::dontSendNotification);
    outputTemperatureNoteLabel.attachToComponent(&outputTemperatureNote, true);

    addAndMakeVisible(outputTemperatureTime);
    outputTemperatureTime.setInputRestrictions(10, "0123456789.");
    outputTemperatureTime.setText(juce::String(modelConfig.outputTemperatures[0]));
    outputTemperatureTime.onTextChange = [this] {
        sendChangeMessage();
    };
    addAndMakeVisible(outputTemperatureDuration);
    outputTemperatureDuration.setInputRestrictions(10, "0123456789.");
    outputTemperatureDuration.setText(juce::String(modelConfig.outputTemperatures[1]));
    outputTemperatureDuration.onTextChange = [this] {
        sendChangeMessage();
    };
    addAndMakeVisible(outputTemperatureNote);
    outputTemperatureNote.setInputRestrictions(10, "0123456789.");
    outputTemperatureNote.setText(juce::String(modelConfig.outputTemperatures[2]));
    outputTemperatureNote.onTextChange = [this] {
        sendChangeMessage();
    };

    /* OUTPUT START TIME */
    addAndMakeVisible(outputStartTimeLabel);
    outputStartTimeLabel.setText("Start Time:", juce::dontSendNotification);
    outputStartTimeLabel.attachToComponent(&outputStartTime, true);

    addAndMakeVisible(outputStartTime);
    outputStartTime.setInputRestrictions(15, "0123456789");
    outputStartTime.setText(juce::String(modelConfig.outputStartTime));
    outputStartTime.onTextChange = [this] {
        sendChangeMessage();
    };

    /* OUTPUT FORCE START TIME */
    addAndMakeVisible(outputForceStartTime);
    outputForceStartTime.setButtonText("Force Start Time");
    outputForceStartTime.setToggleState(modelConfig.outputForceStartTime, juce::dontSendNotification);
    outputForceStartTime.onClick = [this] {
        sendChangeMessage();
    };

    /* OUTPUT INSTRUMENTS */
    for (auto [instrumentId, instrument]: modelConfig.outputInstruments) {
        // Create instrument button
        auto *instrumentButton = outputInstruments.add(new juce::ToggleButton());
        instrumentButton->setButtonText(juce::String(instrumentId));

        // If it is in the active instruments, set it to active
        instrumentButton->setToggleState(instrument.active, juce::dontSendNotification);
        instrumentButton->onClick = [this] {
            sendChangeMessage();
        };
        addAndMakeVisible(instrumentButton);

        // Create monophony button
        auto *monophonyButton = outputInstrumentsMonophony.add(new juce::ToggleButton());
        monophonyButton->setButtonText("Monophony");
        monophonyButton->setToggleState(instrument.monophony, juce::dontSendNotification);
        monophonyButton->onClick = [this] {
            sendChangeMessage();
        };
        addAndMakeVisible(monophonyButton);

        // Create range low text editor
        auto *rangeLow = outputInstrumentsRangeLow.add(new juce::TextEditor());
        rangeLow->setInputRestrictions(10, "0123456789");
        rangeLow->setText(juce::String(instrument.low));
        rangeLow->onTextChange = [this] {
            sendChangeMessage();
        };
        addAndMakeVisible(rangeLow);

        // Create range low label
        auto *rangeLowLabel = outputInstrumentsRangeLowLabel.add(new juce::Label());
        rangeLowLabel->setText("Range:", juce::dontSendNotification);
        rangeLowLabel->attachToComponent(outputInstrumentsRangeLow.getLast(), true);
        addAndMakeVisible(rangeLowLabel);

        // Create range high text editor
        auto *rangeHigh = outputInstrumentsRangeHigh.add(new juce::TextEditor());
        rangeHigh->setInputRestrictions(10, "0123456789");
        rangeHigh->setText(juce::String(instrument.high));
        rangeHigh->onTextChange = [this] {
            sendChangeMessage();
        };
        addAndMakeVisible(rangeHigh);

        // Create range high label
        auto *rangeHighLabel = outputInstrumentsRangeHighLabel.add(new juce::Label());
        rangeHighLabel->setText("- ", juce::dontSendNotification);
        rangeHighLabel->attachToComponent(outputInstrumentsRangeHigh.getLast(), true);
        addAndMakeVisible(rangeHighLabel);
    }

    addAndMakeVisible(outputInstrumentsLabel);
    outputInstrumentsLabel.setText("Output Instruments:", juce::dontSendNotification);
    outputInstrumentsLabel.attachToComponent(outputInstruments.getFirst(), true);

    setSize(700, 900);


}

ModelConfigurationComponent::~ModelConfigurationComponent() {
}

void ModelConfigurationComponent::paint(juce::Graphics &g) {
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

void ModelConfigurationComponent::resized() {
    /* Reset visibiliy of conditional input fields */
    directInputWindowLength.setVisible(false);
    directInputSendNoteOffs.setVisible(false);
    directInputStartOnInput.setVisible(false);
    directInputStartDelay.setVisible(false);
    directInputHoldBass.setVisible(false);
    directInputBassLow.setVisible(false);
    directInputBassHigh.setVisible(false);
    directInputInitialBass.setVisible(false);
    bufferInputSize.setVisible(false);
    bufferInputLow.setVisible(false);
    bufferInputHigh.setVisible(false);
    bufferInputBassLow.setVisible(false);
    bufferInputBassHigh.setVisible(false);


    auto area = getLocalBounds().reduced(8);

    inputTitle.setBounds(area.removeFromTop(36).removeFromLeft(300).reduced(8));

    inputMode.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));

    if (inputMode.getSelectedId() == InputMode::Direct + 1) {
        directInputWindowLength.setVisible(true);
        directInputSendNoteOffs.setVisible(true);
        directInputStartOnInput.setVisible(true);
        directInputStartDelay.setVisible(true);
        directInputHoldBass.setVisible(true);
        directInputBassLow.setVisible(true);
        directInputBassHigh.setVisible(true);
        directInputInitialBass.setVisible(true);

        directInputWindowLength.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));
        directInputSendNoteOffs.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));
        auto directInputStartOnInputArea = area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8);
        directInputStartOnInput.setBounds(directInputStartOnInputArea.removeFromLeft(250));
        directInputStartDelay.setBounds(directInputStartOnInputArea);
        auto directInputBassArea = area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8);
        directInputHoldBass.setBounds(directInputBassArea.removeFromLeft(150));
        directInputBassLow.setBounds(directInputBassArea.removeFromLeft(65).withTrimmedRight(25));
        directInputBassHigh.setBounds(directInputBassArea.removeFromLeft(135).withTrimmedRight(90));
        directInputInitialBass.setBounds(directInputBassArea);
    } else if (inputMode.getSelectedId() == InputMode::Buffer + 1) {
        bufferInputSize.setVisible(true);
        bufferInputLow.setVisible(true);
        bufferInputHigh.setVisible(true);
        bufferInputBassLow.setVisible(true);
        bufferInputBassHigh.setVisible(true);

        bufferInputSize.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));

        auto bufferInputRangeArea = area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8);
        bufferInputLow.setBounds(bufferInputRangeArea.removeFromLeft(65).withTrimmedRight(25));
        bufferInputHigh.setBounds(bufferInputRangeArea.removeFromLeft(135).withTrimmedRight(90));
        bufferInputBassLow.setBounds(bufferInputRangeArea.removeFromLeft(65).withTrimmedRight(25));
        bufferInputBassHigh.setBounds(bufferInputRangeArea.removeFromLeft(45).withTrimmedRight(10));
    }

    inputInstrument.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));

    auto inputRangeArea = area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8);
    inputLow.setBounds(inputRangeArea.removeFromLeft(65).withTrimmedRight(25));
    inputHigh.setBounds(inputRangeArea.removeFromLeft(135).withTrimmedRight(90));

    inputDuration.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));
    inputInitialData.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));

    outputTitle.setBounds(area.removeFromTop(36).removeFromLeft(300).reduced(8));

    outputMinimumDuration.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));
    outputMaximumDuration.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));

    auto outputTemperaturesArea = area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8);
    outputTemperatureTime.setBounds(outputTemperaturesArea.removeFromLeft(50));
    outputTemperatureDuration.setBounds(outputTemperaturesArea.removeFromLeft(150).withTrimmedLeft(100));
    outputTemperatureNote.setBounds(outputTemperaturesArea.removeFromLeft(150).withTrimmedLeft(100));

    auto outputStartTimeArea = area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8);
    outputStartTime.setBounds(outputStartTimeArea.removeFromLeft(150));
    outputForceStartTime.setBounds(outputStartTimeArea.withTrimmedLeft(40));

    for (int i = 0; i < outputInstruments.size(); i++) {
        auto instrumentButtonArea = area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8);
        outputInstruments[i]->setBounds(instrumentButtonArea.removeFromLeft(60));
        outputInstrumentsMonophony[i]->setBounds(instrumentButtonArea.removeFromLeft(120));
        outputInstrumentsRangeLow[i]->setBounds(instrumentButtonArea.withTrimmedLeft(50).removeFromLeft(50));
        outputInstrumentsRangeHigh[i]->setBounds(instrumentButtonArea.withTrimmedLeft(125).removeFromLeft(50));
    }
    auto buttonArea = area.removeFromBottom(70).reduced(8);

    cancelButton.setBounds(buttonArea.removeFromRight(100).reduced(8));
    applyButton.setBounds(buttonArea.removeFromRight(100).reduced(8));
    okButton.setBounds(buttonArea.removeFromRight(100).reduced(8));
}

void ModelConfigurationComponent::userTriedToCloseWindow() {
    delete this;
}

bool ModelConfigurationComponent::apply() {
    // Check if at least one instrument is active
    if (std::find_if(outputInstruments.begin(), outputInstruments.end(), [](juce::ToggleButton *button) {
        return button->getToggleState();
    }) == outputInstruments.end()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "At least one instrument must be active.");
        return false;
    }

    // Check that directInputSendNoteOffs and directInputStartOnInput are not both true.
    if (inputMode.getSelectedId() == InputMode::Direct + 1
        && directInputSendNoteOffs.getToggleState() && directInputStartOnInput.getToggleState()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "\"Send Note Offs\" and \"Start on Input\" cannot be both active.");
        return false;
    }

    modelConfig.inputMode = static_cast<InputMode>(inputMode.getSelectedId() - 1);
    modelConfig.inputInstrument = inputInstrument.getText().getIntValue();
    modelConfig.inputLow = inputLow.getText().getIntValue();
    modelConfig.inputHigh = inputHigh.getText().getIntValue();
    modelConfig.inputDuration = inputDuration.getText().getIntValue();
    modelConfig.inputInitialData = inputInitialData.getText();
    modelConfig.directInputWindowLength = directInputWindowLength.getText().getIntValue();
    modelConfig.directInputSendNoteOffs = directInputSendNoteOffs.getToggleState();
    modelConfig.directInputStartOnInput = directInputStartOnInput.getToggleState();
    modelConfig.directInputStartDelay = directInputStartDelay.getText().getIntValue();
    modelConfig.directInputHoldBass = directInputHoldBass.getToggleState();
    modelConfig.directInputBassLow = directInputBassLow.getText().getIntValue();
    modelConfig.directInputBassHigh = directInputBassHigh.getText().getIntValue();
    modelConfig.directInputInitialBass = directInputInitialBass.getText().getIntValue();
    modelConfig.bufferInputSize = bufferInputSize.getText().getIntValue();
    modelConfig.bufferInputLow = bufferInputLow.getText().getIntValue();
    modelConfig.bufferInputHigh = bufferInputHigh.getText().getIntValue();
    modelConfig.bufferInputBassLow = bufferInputBassLow.getText().getIntValue();
    modelConfig.bufferInputBassHigh = bufferInputBassHigh.getText().getIntValue();
    modelConfig.outputMinimumDuration = outputMinimumDuration.getText().getIntValue();
    modelConfig.outputMaximumDuration = outputMaximumDuration.getText().getIntValue();
    modelConfig.outputTemperatures[0] = outputTemperatureTime.getText().getFloatValue();
    modelConfig.outputTemperatures[1] = outputTemperatureDuration.getText().getFloatValue();
    modelConfig.outputTemperatures[2] = outputTemperatureNote.getText().getFloatValue();
    modelConfig.outputStartTime = outputStartTime.getText().getIntValue();
    modelConfig.outputForceStartTime = outputForceStartTime.getToggleState();

    modelConfig.sortedActiveOutputInstruments.clear();

    for (int32_t i = 0; i < outputInstruments.size(); i++) {
        const int32_t instrumentId = outputInstruments[i]->getButtonText().getIntValue();
        modelConfig.outputInstruments[instrumentId].active = outputInstruments[i]->getToggleState();
        modelConfig.outputInstruments[instrumentId].monophony = outputInstrumentsMonophony[i]->getToggleState();
        modelConfig.outputInstruments[instrumentId].low = outputInstrumentsRangeLow[i]->getText().getIntValue();
        modelConfig.outputInstruments[instrumentId].high = outputInstrumentsRangeHigh[i]->getText().getIntValue();

        // If the instrument is active, add it to the set of sorted active instruments
        if (modelConfig.outputInstruments[instrumentId].active) {
            modelConfig.sortedActiveOutputInstruments.emplace(instrumentId);
        }
    }

    return true;
}
