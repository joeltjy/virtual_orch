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
    inputMode.addItem("Trading", InputMode::Trading + 1);
    inputMode.addItem("Playback", InputMode::Playback + 1);
    inputMode.setSelectedId(modelConfig.inputMode + 1, juce::dontSendNotification);
    inputMode.onChange = [this] {
        sendChangeMessage();

        // If Trading, force "Send Bar Separators"
        if (inputMode.getSelectedId() == InputMode::Trading + 1) {
            outputSendBarSeparators.setToggleState(true, juce::dontSendNotification);
            outputSendBarSeparators.setEnabled(false);
        } else {
            outputSendBarSeparators.setEnabled(true);
        }

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

    /* INPUT CLEARS PAST */
    addAndMakeVisible(inputClearsPastLabel);
    inputClearsPastLabel.setText("Input Clears Past:", juce::dontSendNotification);
    inputClearsPastLabel.attachToComponent(&inputClearsPast, true);

    addAndMakeVisible(inputClearsPast);
    inputClearsPast.setToggleState(modelConfig.inputClearsPast, juce::dontSendNotification);
    inputClearsPast.onClick = [this] {
        sendChangeMessage();
    };

    /* INPUT CLICKS */
    addAndMakeVisible(inputClicksLabel);
    inputClicksLabel.setText("Input Clicks:", juce::dontSendNotification);
    inputClicksLabel.attachToComponent(&inputClicks, true);

    addAndMakeVisible(inputClicks);
    inputClicks.setToggleState(modelConfig.inputClicks, juce::dontSendNotification);
    inputClicks.onClick = [this] {
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

    /* INPUT INITIAL DATA REFRESH RATE */
    addAndMakeVisible(inputInitialDataRefreshRateLabel);
    inputInitialDataRefreshRateLabel.setText("Input Initial Data Refresh Rate:", juce::dontSendNotification);
    inputInitialDataRefreshRateLabel.attachToComponent(&inputInitialDataRefreshRate, true);

    addAndMakeVisible(inputInitialDataRefreshRate);
    inputInitialDataRefreshRate.setInputRestrictions(15, "0123456789");
    inputInitialDataRefreshRate.setText(juce::String(modelConfig.inputInitialDataRefreshRate));
    inputInitialDataRefreshRate.onTextChange = [this] {
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

    /* DIRECT INPUT MANUAL TRIGGER */
    addAndMakeVisible(directInputManualTriggerLabel);
    directInputManualTriggerLabel.setText("Manual Trigger:", juce::dontSendNotification);
    directInputManualTriggerLabel.attachToComponent(&directInputManualTrigger, true);

    addAndMakeVisible(directInputManualTrigger);
    directInputManualTrigger.setToggleState(modelConfig.directInputManualTrigger, juce::dontSendNotification);
    directInputManualTrigger.onClick = [this] {
        sendChangeMessage();
    };

    /* DIRECT INPUT MANUAL TRIGGER CONTROL TYPE */
    addAndMakeVisible(directInputManualTriggerControlType);
    directInputManualTriggerControlType.addItem("Note", ControlType::Note + 1);
    directInputManualTriggerControlType.addItem("CC", ControlType::CC + 1);
    directInputManualTriggerControlType.addItem("OSCController", ControlType::OscController + 1);
    directInputManualTriggerControlType.setSelectedId(
        static_cast<int>(modelConfig.directInputManualTriggerControlType) + 1, juce::dontSendNotification);
    directInputManualTriggerControlType.onChange = [this] {
        sendChangeMessage();
        // Resize to show new parameters
        resized();
    };

    /* DIRECT INPUT MANUAL TRIGGER CONTROL ID */
    addAndMakeVisible(directInputManualTriggerControlId);
    directInputManualTriggerControlId.setInputRestrictions(10, "0123456789");
    directInputManualTriggerControlId.setText(juce::String(modelConfig.directInputManualTriggerControlId));
    directInputManualTriggerControlId.onTextChange = [this] {
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

    /* DIRECT INPUT SNAP ON BAR */
    addAndMakeVisible(directInputSnapOnBarLabel);
    directInputSnapOnBarLabel.setText("Snap on Closest Bar:", juce::dontSendNotification);
    directInputSnapOnBarLabel.attachToComponent(&directInputSnapOnBar, true);

    addAndMakeVisible(directInputSnapOnBar);
    directInputSnapOnBar.setToggleState(modelConfig.directInputSnapOnBar, juce::dontSendNotification);
    directInputSnapOnBar.onClick = [this] {
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

    /* DIRECT INPUT UNPAUSE ON INPUT */
    addAndMakeVisible(directInputUnpauseOnInputLabel);
    directInputUnpauseOnInputLabel.setText("Unpause on Input:", juce::dontSendNotification);
    directInputUnpauseOnInputLabel.attachToComponent(&directInputUnpauseOnInput, true);

    addAndMakeVisible(directInputUnpauseOnInput);
    directInputUnpauseOnInput.setToggleState(modelConfig.directInputUnpauseOnInput, juce::dontSendNotification);
    directInputUnpauseOnInput.onClick = [this] {
        sendChangeMessage();
    };

    /* DIRECT INPUT UNPAUSE DELAY */
    addAndMakeVisible(directInputUnpauseDelayLabel);
    directInputUnpauseDelayLabel.setText("Unpause Delay:", juce::dontSendNotification);
    directInputUnpauseDelayLabel.attachToComponent(&directInputUnpauseDelay, true);

    addAndMakeVisible(directInputUnpauseDelay);
    directInputUnpauseDelay.setInputRestrictions(10, "0123456789");
    directInputUnpauseDelay.setText(juce::String(modelConfig.directInputUnpauseDelay));
    directInputUnpauseDelay.onTextChange = [this] {
        sendChangeMessage();
    };

    /* DIRECT INPUT MANUAL PAUSE */
    addAndMakeVisible(directInputManualPauseLabel);
    directInputManualPauseLabel.setText("Manual Pause:", juce::dontSendNotification);
    directInputManualPauseLabel.attachToComponent(&directInputManualPause, true);

    addAndMakeVisible(directInputManualPause);
    directInputManualPause.setToggleState(modelConfig.directInputManualPause, juce::dontSendNotification);
    directInputManualPause.onClick = [this] {
        sendChangeMessage();
    };

    /* DIRECT INPUT MANUAL PAUSE CONTROL TYPE */
    addAndMakeVisible(directInputManualPauseControlType);
    directInputManualPauseControlType.addItem("Note", ControlType::Note + 1);
    directInputManualPauseControlType.addItem("CC", ControlType::CC + 1);
    directInputManualPauseControlType.addItem("OSCController", ControlType::OscController + 1);
    directInputManualPauseControlType.setSelectedId(
        static_cast<int>(modelConfig.directInputManualPauseControlType) + 1, juce::dontSendNotification);
    directInputManualPauseControlType.onChange = [this] {
        sendChangeMessage();
        // Resize to show new parameters
        resized();
    };

    /* DIRECT INPUT MANUAL PAUSE CONTROL ID */
    addAndMakeVisible(directInputManualPauseControlId);
    directInputManualPauseControlId.setInputRestrictions(10, "0123456789");
    directInputManualPauseControlId.setText(juce::String(modelConfig.directInputManualPauseControlId));
    directInputManualPauseControlId.onTextChange = [this] {
        sendChangeMessage();
    };

    /* DIRECT INPUT MANUAL PAUSE CONTROL TRIGGER TYPE */
    addAndMakeVisible(directInputManualPauseControlTriggerType);
    directInputManualPauseControlTriggerType.addItem("Toggle", ControlTriggerType::Toggle + 1);
    directInputManualPauseControlTriggerType.addItem("Momentary", ControlTriggerType::Momentary + 1);
    directInputManualPauseControlTriggerType.setSelectedId(
        static_cast<int>(modelConfig.directInputManualPauseControlTriggerType) + 1, juce::dontSendNotification);
    directInputManualPauseControlTriggerType.onChange = [this] {
        sendChangeMessage();
    };

    /* DIRECT INPUT MANUAL PAUSE CLEARS FUTURE NOTES */
    addAndMakeVisible(directInputManualPauseClearsFutureNotesLabel);
    directInputManualPauseClearsFutureNotesLabel.setText("X Future:", juce::dontSendNotification);
    directInputManualPauseClearsFutureNotesLabel.attachToComponent(&directInputManualPauseClearsFutureNotes, true);

    addAndMakeVisible(directInputManualPauseClearsFutureNotes);
    directInputManualPauseClearsFutureNotes.setToggleState(modelConfig.directInputManualPauseClearsFutureNotes,
                                                           juce::dontSendNotification);
    directInputManualPauseClearsFutureNotes.onClick = [this] {
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

    /* BUFFER INPUT PROMPT ON NEXT BAR */
    addAndMakeVisible(bufferInputPromptOnNextBarLabel);
    bufferInputPromptOnNextBarLabel.setText("Prompt on Next Bar:", juce::dontSendNotification);
    bufferInputPromptOnNextBarLabel.attachToComponent(&bufferInputPromptOnNextBar, true);

    addAndMakeVisible(bufferInputPromptOnNextBar);
    bufferInputPromptOnNextBar.setToggleState(modelConfig.bufferInputPromptOnNextBar, juce::dontSendNotification);
    bufferInputPromptOnNextBar.onClick = [this] {
        sendChangeMessage();
    };

    /* BUFFER INPUT FORCE ON NEXT BAR */
    addAndMakeVisible(bufferInputForceOnNextBarLabel);
    bufferInputForceOnNextBarLabel.setText("Force all instruments on Next Bar:", juce::dontSendNotification);
    bufferInputForceOnNextBarLabel.attachToComponent(&bufferInputForceOnNextBar, true);

    addAndMakeVisible(bufferInputForceOnNextBar);
    bufferInputForceOnNextBar.setToggleState(modelConfig.bufferInputForceOnNextBar, juce::dontSendNotification);
    bufferInputForceOnNextBar.onClick = [this] {
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

    /* TRADING INPUT NUMBER OF BARS */
    addAndMakeVisible(tradingInputNumberOfBarsLabel);
    tradingInputNumberOfBarsLabel.setText("Number of Bars for Trading:", juce::dontSendNotification);
    tradingInputNumberOfBarsLabel.attachToComponent(&tradingInputNumberOfBars, true);

    addAndMakeVisible(tradingInputNumberOfBars);
    tradingInputNumberOfBars.setInputRestrictions(10, "0123456789");
    tradingInputNumberOfBars.setText(juce::String(modelConfig.tradingInputNumberOfBars));
    tradingInputNumberOfBars.onTextChange = [this] {
        sendChangeMessage();
    };

    /* TRADING INPUT MODEL GOES FIRST */
    addAndMakeVisible(tradingInputModelGoesFirst);
    tradingInputModelGoesFirst.setButtonText("Model Goes First");
    tradingInputModelGoesFirst.setToggleState(modelConfig.tradingInputModelGoesFirst, juce::dontSendNotification);
    tradingInputModelGoesFirst.onClick = [this] {
        sendChangeMessage();
    };

    /* TRADING INPUT INITIAL OFFSET */
    addAndMakeVisible(tradingInputInitialOffsetLabel);
    tradingInputInitialOffsetLabel.setText("Initial Offset:", juce::dontSendNotification);
    tradingInputInitialOffsetLabel.attachToComponent(&tradingInputInitialOffset, true);

    addAndMakeVisible(tradingInputInitialOffset);
    tradingInputInitialOffset.setInputRestrictions(15, "0123456789");
    tradingInputInitialOffset.setText(juce::String(modelConfig.tradingInputInitialOffset));
    tradingInputInitialOffset.onTextChange = [this] {
        sendChangeMessage();
    };

    /* PLAYBACK INPUT START TIME */
    addAndMakeVisible(playbackInputStartTimeLabel);
    playbackInputStartTimeLabel.setText("Playback Start Time:", juce::dontSendNotification);
    playbackInputStartTimeLabel.attachToComponent(&playbackInputStartTime, true);

    addAndMakeVisible(playbackInputStartTime);
    playbackInputStartTime.setInputRestrictions(15, "0123456789");
    playbackInputStartTime.setText(juce::String(modelConfig.playbackInputStartTime));
    playbackInputStartTime.onTextChange = [this] {
        sendChangeMessage();
    };

    /* PLAYBACK INPUT END TIME */
    addAndMakeVisible(playbackInputEndTimeLabel);
    playbackInputEndTimeLabel.setText("Playback End Time:", juce::dontSendNotification);
    playbackInputEndTimeLabel.attachToComponent(&playbackInputEndTime, true);

    addAndMakeVisible(playbackInputEndTime);
    playbackInputEndTime.setInputRestrictions(15, "0123456789");
    playbackInputEndTime.setText(juce::String(modelConfig.playbackInputEndTime));
    playbackInputEndTime.onTextChange = [this] {
        sendChangeMessage();
    };

    /* PLAYBACK INPUT CONTROL SEQUENCE */
    addAndMakeVisible(playbackInputControlSequenceLabel);
    playbackInputControlSequenceLabel.setText("Control Sequence:", juce::dontSendNotification);
    playbackInputControlSequenceLabel.attachToComponent(&playbackInputControlSequence, true);

    addAndMakeVisible(playbackInputControlSequence);
    playbackInputControlSequence.setInputRestrictions(10000, "0123456789 ");
    playbackInputControlSequence.setText(modelConfig.playbackInputControlSequence);
    playbackInputControlSequence.onTextChange = [this] {
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

    /* OUTPUT MINIMUM TIME INTERVAL */
    addAndMakeVisible(outputMinimumTimeIntervalLabel);
    outputMinimumTimeIntervalLabel.setText("Output Minimum Time Interval:", juce::dontSendNotification);
    outputMinimumTimeIntervalLabel.attachToComponent(&outputMinimumTimeInterval, true);

    addAndMakeVisible(outputMinimumTimeInterval);
    outputMinimumTimeInterval.setInputRestrictions(10, "0123456789");
    outputMinimumTimeInterval.setText(juce::String(modelConfig.outputMinimumTimeInterval));
    outputMinimumTimeInterval.onTextChange = [this] {
        sendChangeMessage();
    };

    /* OUTPUT ALLOW CHORDS */
    addAndMakeVisible(outputAllowChords);
    outputAllowChords.setButtonText("Allow Chords");
    outputAllowChords.setToggleState(modelConfig.outputAllowChords, juce::dontSendNotification);
    outputAllowChords.onClick = [this] {
        sendChangeMessage();
    };

    /* OUTPUT MAX CHORD SIZE */
    addAndMakeVisible(outputMaximumChordSizeLabel);
    outputMaximumChordSizeLabel.setText("Max Chord Size:", juce::dontSendNotification);
    outputMaximumChordSizeLabel.attachToComponent(&outputMaximumChordSize, true);

    addAndMakeVisible(outputMaximumChordSize);
    outputMaximumChordSize.setInputRestrictions(2, "0123456789");
    outputMaximumChordSize.setText(juce::String(modelConfig.outputMaximumChordSize));
    outputMaximumChordSize.onTextChange = [this] {
        sendChangeMessage();
    };

    /* OUTPUT BAR SEPARATORS */
    addAndMakeVisible(outputSendBarSeparatorsLabel);
    outputSendBarSeparatorsLabel.setText("Send Bar Separator Signals:", juce::dontSendNotification);
    outputSendBarSeparatorsLabel.attachToComponent(&outputSendBarSeparators, true);

    addAndMakeVisible(outputSendBarSeparators);
    outputSendBarSeparators.setToggleState(
        modelConfig.outputSendBarSeparators || modelConfig.inputMode == InputMode::Trading, juce::dontSendNotification);
    outputSendBarSeparators.setEnabled(modelConfig.inputMode != InputMode::Trading);
    outputSendBarSeparators.onClick = [this] {
        sendChangeMessage();
    };

    /* OUTPUT BAR LENGTH */
    addAndMakeVisible(outputBarLengthLabel);
    outputBarLengthLabel.setText("Bar Length:", juce::dontSendNotification);
    outputBarLengthLabel.attachToComponent(&outputBarLength, true);

    addAndMakeVisible(outputBarLength);
    outputBarLength.setInputRestrictions(10, "0123456789");
    outputBarLength.setText(juce::String(modelConfig.outputBarLength));
    outputBarLength.onTextChange = [this] {
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

    /* OUTPUT PAUSE AFTER TIME INTERVAL */
    addAndMakeVisible(outputPauseAfterTimeIntervalLabel);
    outputPauseAfterTimeIntervalLabel.setText("Pause After Time Interval:", juce::dontSendNotification);
    outputPauseAfterTimeIntervalLabel.attachToComponent(&outputPauseAfterTimeInterval, true);

    addAndMakeVisible(outputPauseAfterTimeInterval);
    outputPauseAfterTimeInterval.setToggleState(modelConfig.outputPauseAfterTimeInterval, juce::dontSendNotification);
    outputPauseAfterTimeInterval.onClick = [this] {
        sendChangeMessage();
    };

    /* OUTPUT PAUSE AFTER TIME INTERVAL VALUE */
    addAndMakeVisible(outputPauseAfterTimeIntervalValueLabel);
    outputPauseAfterTimeIntervalValueLabel.setText("Time Interval:", juce::dontSendNotification);
    outputPauseAfterTimeIntervalValueLabel.attachToComponent(&outputPauseAfterTimeIntervalValue, true);

    addAndMakeVisible(outputPauseAfterTimeIntervalValue);
    outputPauseAfterTimeIntervalValue.setInputRestrictions(15, "0123456789");
    outputPauseAfterTimeIntervalValue.setText(juce::String(modelConfig.outputPauseAfterTimeIntervalValue));
    outputPauseAfterTimeIntervalValue.onTextChange = [this] {
        sendChangeMessage();
    };

    /* OUTPUT PAUSE AFTER DURATION */
    addAndMakeVisible(outputPauseAfterDurationLabel);
    outputPauseAfterDurationLabel.setText("Pause After Duration:", juce::dontSendNotification);
    outputPauseAfterDurationLabel.attachToComponent(&outputPauseAfterDuration, true);

    addAndMakeVisible(outputPauseAfterDuration);
    outputPauseAfterDuration.setToggleState(modelConfig.outputPauseAfterDuration, juce::dontSendNotification);
    outputPauseAfterDuration.onClick = [this] {
        sendChangeMessage();
    };

    /* OUTPUT PAUSE AFTER DURATION VALUE */
    addAndMakeVisible(outputPauseAfterDurationValueLabel);
    outputPauseAfterDurationValueLabel.setText("Duration:", juce::dontSendNotification);
    outputPauseAfterDurationValueLabel.attachToComponent(&outputPauseAfterDurationValue, true);

    addAndMakeVisible(outputPauseAfterDurationValue);
    outputPauseAfterDurationValue.setInputRestrictions(15, "0123456789");
    outputPauseAfterDurationValue.setText(juce::String(modelConfig.outputPauseAfterDurationValue));
    outputPauseAfterDurationValue.onTextChange = [this] {
        sendChangeMessage();
    };

    /* OUTPUT PAUSE AFTER TRADING SEQUENCE */
    addAndMakeVisible(outputPauseAfterTradingSequenceLabel);
    outputPauseAfterTradingSequenceLabel.setText("Pause After Trading Sequence:", juce::dontSendNotification);
    outputPauseAfterTradingSequenceLabel.attachToComponent(&outputPauseAfterTradingSequence, true);

    addAndMakeVisible(outputPauseAfterTradingSequence);
    outputPauseAfterTradingSequence.setToggleState(modelConfig.outputPauseAfterTradingSequence, juce::dontSendNotification);
    outputPauseAfterTradingSequence.onClick = [this] {
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

    setSize(700, 1150);
}

ModelConfigurationComponent::~ModelConfigurationComponent() {
}

void ModelConfigurationComponent::paint(juce::Graphics &g) {
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

void ModelConfigurationComponent::resized() {
    /* Reset visibiliy of conditional input fields */
    directInputWindowLength.setVisible(false);
    directInputManualTrigger.setVisible(false);
    directInputManualTriggerControlType.setVisible(false);
    directInputManualTriggerControlId.setVisible(false);
    directInputSendNoteOffs.setVisible(false);
    directInputSnapOnBar.setVisible(false);
    directInputStartOnInput.setVisible(false);
    directInputStartDelay.setVisible(false);
    directInputUnpauseOnInput.setVisible(false);
    directInputUnpauseDelay.setVisible(false);
    directInputManualPause.setVisible(false);
    directInputManualPauseControlType.setVisible(false);
    directInputManualPauseControlId.setVisible(false);
    directInputManualPauseControlTriggerType.setVisible(false);
    directInputManualPauseClearsFutureNotes.setVisible(false);
    directInputHoldBass.setVisible(false);
    directInputBassLow.setVisible(false);
    directInputBassHigh.setVisible(false);
    directInputInitialBass.setVisible(false);
    bufferInputPromptOnNextBar.setVisible(false);
    bufferInputForceOnNextBar.setVisible(false);
    bufferInputSize.setVisible(false);
    bufferInputLow.setVisible(false);
    bufferInputHigh.setVisible(false);
    bufferInputBassLow.setVisible(false);
    bufferInputBassHigh.setVisible(false);
    tradingInputNumberOfBars.setVisible(false);
    tradingInputModelGoesFirst.setVisible(false);
    tradingInputInitialOffset.setVisible(false);
    playbackInputStartTime.setVisible(false);
    playbackInputEndTime.setVisible(false);
    playbackInputControlSequence.setVisible(false);

    auto area = getLocalBounds().reduced(8);

    inputTitle.setBounds(area.removeFromTop(36).removeFromLeft(300).reduced(8));

    inputMode.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));

    if (inputMode.getSelectedId() == InputMode::Direct + 1) {
        directInputWindowLength.setVisible(true);
        directInputManualTrigger.setVisible(true);
        directInputManualTriggerControlType.setVisible(true);
        directInputManualTriggerControlId.setVisible(true);
        directInputSendNoteOffs.setVisible(true);
        directInputSnapOnBar.setVisible(true);
        directInputStartOnInput.setVisible(true);
        directInputStartDelay.setVisible(true);
        directInputUnpauseOnInput.setVisible(true);
        directInputUnpauseDelay.setVisible(true);
        directInputManualPause.setVisible(true);
        directInputManualPauseControlType.setVisible(true);
        /* Only show control ID and trigger type if control type is not "OSCController" (later) */
        directInputManualPauseClearsFutureNotes.setVisible(true);
        directInputHoldBass.setVisible(true);
        directInputBassLow.setVisible(true);
        directInputBassHigh.setVisible(true);
        directInputInitialBass.setVisible(true);

        directInputWindowLength.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));
        auto directInputManualTriggerArea = area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8);
        directInputManualTrigger.setBounds(directInputManualTriggerArea.removeFromLeft(60));
        directInputManualTriggerControlType.setBounds(
            directInputManualTriggerArea.removeFromLeft(100).withTrimmedRight(10)
        );
        if (directInputManualTriggerControlType.getSelectedId() != ControlType::OscController + 1) {
            directInputManualTriggerControlId.setVisible(true);
            directInputManualTriggerControlId.setBounds(directInputManualTriggerArea.removeFromLeft(50));
        }
        directInputSendNoteOffs.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));
        directInputSnapOnBar.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));
        auto directInputStartOnInputArea = area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8);
        directInputStartOnInput.setBounds(directInputStartOnInputArea.removeFromLeft(250));
        directInputStartDelay.setBounds(directInputStartOnInputArea);
        auto directInputUnpauseOnInputArea = area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8);
        directInputUnpauseOnInput.setBounds(directInputUnpauseOnInputArea.removeFromLeft(250));
        directInputUnpauseDelay.setBounds(directInputUnpauseOnInputArea);
        auto directInputManualPauseArea = area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8);
        directInputManualPause.setBounds(directInputManualPauseArea.removeFromLeft(60));
        directInputManualPauseControlType.setBounds(
            directInputManualPauseArea.removeFromLeft(100).withTrimmedRight(10)
        );
        if (directInputManualPauseControlType.getSelectedId() != ControlType::OscController + 1) {
            directInputManualPauseControlId.setVisible(true);
            directInputManualPauseControlTriggerType.setVisible(true);
            directInputManualPauseControlId.setBounds(directInputManualPauseArea.removeFromLeft(50));
            directInputManualPauseControlTriggerType.setBounds(
                directInputManualPauseArea.removeFromLeft(110).withTrimmedLeft(20)
            );
        }
        directInputManualPauseClearsFutureNotes.setBounds(directInputManualPauseArea.removeFromRight(25));
        auto directInputBassArea = area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8);
        directInputHoldBass.setBounds(directInputBassArea.removeFromLeft(150));
        directInputBassLow.setBounds(directInputBassArea.removeFromLeft(65).withTrimmedRight(25));
        directInputBassHigh.setBounds(directInputBassArea.removeFromLeft(135).withTrimmedRight(90));
        directInputInitialBass.setBounds(directInputBassArea);
    } else if (inputMode.getSelectedId() == InputMode::Buffer + 1) {
        bufferInputSize.setVisible(true);
        bufferInputLow.setVisible(true);
        bufferInputHigh.setVisible(true);
        bufferInputPromptOnNextBar.setVisible(true);
        bufferInputForceOnNextBar.setVisible(true);
        bufferInputBassLow.setVisible(true);
        bufferInputBassHigh.setVisible(true);

        auto bufferNextBarArea = area.removeFromTop(36).reduced(8);
        bufferInputPromptOnNextBar.setBounds(bufferNextBarArea.removeFromLeft(500).withTrimmedLeft(250));
        bufferInputForceOnNextBar.setBounds(bufferNextBarArea.removeFromLeft(250));
        bufferInputSize.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));

        auto bufferInputRangeArea = area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8);
        bufferInputLow.setBounds(bufferInputRangeArea.removeFromLeft(65).withTrimmedRight(25));
        bufferInputHigh.setBounds(bufferInputRangeArea.removeFromLeft(135).withTrimmedRight(90));
        bufferInputBassLow.setBounds(bufferInputRangeArea.removeFromLeft(65).withTrimmedRight(25));
        bufferInputBassHigh.setBounds(bufferInputRangeArea.removeFromLeft(45).withTrimmedRight(10));
    } else if (inputMode.getSelectedId() == InputMode::Trading + 1) {
        tradingInputNumberOfBars.setVisible(true);
        tradingInputModelGoesFirst.setVisible(true);
        tradingInputInitialOffset.setVisible(true);

        tradingInputNumberOfBars.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));
        auto tradingInputOffsetArea = area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8);
        tradingInputModelGoesFirst.setBounds(tradingInputOffsetArea.removeFromLeft(250));
        tradingInputInitialOffset.setBounds(tradingInputOffsetArea);
    } else if (inputMode.getSelectedId() == InputMode::Playback + 1) {
        playbackInputStartTime.setVisible(true);
        playbackInputEndTime.setVisible(true);
        playbackInputControlSequence.setVisible(true);

        playbackInputStartTime.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));
        playbackInputEndTime.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));
        playbackInputControlSequence.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));
    }

    inputInstrument.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));

    auto inputRangeArea = area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8);
    inputLow.setBounds(inputRangeArea.removeFromLeft(65).withTrimmedRight(25));
    inputHigh.setBounds(inputRangeArea.removeFromLeft(135).withTrimmedRight(90));

    inputDuration.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));
    inputClearsPast.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));
    inputClicks.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));
    inputInitialData.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));
    inputInitialDataRefreshRate.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));

    outputTitle.setBounds(area.removeFromTop(36).removeFromLeft(300).reduced(8));

    outputMinimumDuration.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));
    outputMaximumDuration.setBounds(area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8));
    auto timeIntervalArea = area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8);
    outputMinimumTimeInterval.setBounds(timeIntervalArea.removeFromLeft(120).withTrimmedRight(40));
    outputAllowChords.setBounds(timeIntervalArea.removeFromLeft(250).withTrimmedRight(100));
    outputMaximumChordSize.setBounds(timeIntervalArea);

    auto outputSendBarSeparatorsArea = area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8);
    outputSendBarSeparators.setBounds(outputSendBarSeparatorsArea.removeFromLeft(200).withTrimmedRight(90));
    outputBarLength.setBounds(outputSendBarSeparatorsArea.removeFromLeft(90));

    auto outputTemperaturesArea = area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8);
    outputTemperatureTime.setBounds(outputTemperaturesArea.removeFromLeft(50));
    outputTemperatureDuration.setBounds(outputTemperaturesArea.removeFromLeft(150).withTrimmedLeft(100));
    outputTemperatureNote.setBounds(outputTemperaturesArea.removeFromLeft(150).withTrimmedLeft(100));

    auto outputStartTimeArea = area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8);
    outputStartTime.setBounds(outputStartTimeArea.removeFromLeft(150));
    outputForceStartTime.setBounds(outputStartTimeArea.withTrimmedLeft(40));

    auto outputPauseAfterTimeIntervalArea = area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8);
    outputPauseAfterTimeInterval.setBounds(outputPauseAfterTimeIntervalArea.removeFromLeft(200).withTrimmedRight(90));
    outputPauseAfterTimeIntervalValue.setBounds(outputPauseAfterTimeIntervalArea);

    auto outputPauseAfterDurationArea = area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8);
    outputPauseAfterDuration.setBounds(outputPauseAfterDurationArea.removeFromLeft(200).withTrimmedRight(90));
    outputPauseAfterDurationValue.setBounds(outputPauseAfterDurationArea);

    auto outputPauseAfterTradingSequenceArea = area.removeFromTop(36).removeFromRight(getWidth() - 250).reduced(8);
    outputPauseAfterTradingSequence.setBounds(outputPauseAfterTradingSequenceArea.removeFromLeft(200).withTrimmedRight(90));

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

    // Check that if Input Mode is Trading, then Send Bar Separators must be active too.
    if (inputMode.getSelectedId() == InputMode::Trading + 1 && !outputSendBarSeparators.getToggleState()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "\"Send Bar Separators\" must be active when Trading is used.");
        return false;
    }

    // Check that directInputSendNoteOffs and directInputSnapOnBar are not both true.
    if (inputMode.getSelectedId() == InputMode::Direct + 1
        && directInputSendNoteOffs.getToggleState() && directInputSnapOnBar.getToggleState()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "\"Send Note Offs\" and \"Snap on Closest Bar\" cannot be both active.");
        return false;
    }

    // Check that directInputSendNoteOffs and directInputStartOnInput are not both true.
    if (inputMode.getSelectedId() == InputMode::Direct + 1
        && directInputSendNoteOffs.getToggleState() && directInputStartOnInput.getToggleState()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "\"Send Note Offs\" and \"Start on Input\" cannot be both active.");
        return false;
    }

    // Check that directInputSendNoteOffs and directInputUnpauseOnInput are not both true.
    if (inputMode.getSelectedId() == InputMode::Direct + 1
        && directInputSendNoteOffs.getToggleState() && directInputUnpauseOnInput.getToggleState()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "\"Send Note Offs\" and \"Unpause on Input\" cannot be both active.");
        return false;
    }

    // Check that directInputSendNoteOffs is true if directInputManualPause is true
    if (inputMode.getSelectedId() == InputMode::Direct + 1
        && directInputManualPause.getToggleState() && !directInputSendNoteOffs.getToggleState()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "\"Send Note Offs\" must be active if \"Manual Pause\" is active.");
        return false;
    }

    // Check that outputPauseAfterTradingSequence is not true if inputMode is not Trading
    if (outputPauseAfterTradingSequence.getToggleState() && inputMode.getSelectedId() != InputMode::Trading + 1) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon, "Error",
                                               "\"Pause After Trading Sequence\" can only be active when Input Mode is set to Trading.");
        return false;
    }

    modelConfig.inputMode = static_cast<InputMode>(inputMode.getSelectedId() - 1);
    modelConfig.inputInstrument = inputInstrument.getText().getIntValue();
    modelConfig.inputLow = inputLow.getText().getIntValue();
    modelConfig.inputHigh = inputHigh.getText().getIntValue();
    modelConfig.inputDuration = inputDuration.getText().getIntValue();
    modelConfig.inputClearsPast = inputClearsPast.getToggleState();
    modelConfig.inputClicks = inputClicks.getToggleState();
    modelConfig.inputInitialData = inputInitialData.getText();
    modelConfig.inputInitialDataRefreshRate = inputInitialData.getText().getIntValue();
    modelConfig.directInputWindowLength = directInputWindowLength.getText().getIntValue();
    modelConfig.directInputManualTrigger = directInputManualTrigger.getToggleState();
    modelConfig.directInputManualTriggerControlType = static_cast<ControlType>(
        directInputManualTriggerControlType.getSelectedId() - 1);
    modelConfig.directInputManualTriggerControlId = directInputManualTriggerControlId.getText().getIntValue();
    modelConfig.directInputSendNoteOffs = directInputSendNoteOffs.getToggleState();
    modelConfig.directInputSnapOnBar = directInputSnapOnBar.getToggleState();
    modelConfig.directInputStartOnInput = directInputStartOnInput.getToggleState();
    modelConfig.directInputStartDelay = directInputStartDelay.getText().getIntValue();
    modelConfig.directInputUnpauseOnInput = directInputUnpauseOnInput.getToggleState();
    modelConfig.directInputUnpauseDelay = directInputUnpauseDelay.getText().getIntValue();
    modelConfig.directInputManualPause = directInputManualPause.getToggleState();
    modelConfig.directInputManualPauseControlType = static_cast<ControlType>(
        directInputManualPauseControlType.getSelectedId() - 1);
    modelConfig.directInputManualPauseControlId = directInputManualPauseControlId.getText().getIntValue();
    modelConfig.directInputManualPauseControlTriggerType =
            static_cast<ControlTriggerType>(directInputManualPauseControlTriggerType.getSelectedId() - 1);
    modelConfig.directInputManualPauseClearsFutureNotes = directInputManualPauseClearsFutureNotes.getToggleState();
    modelConfig.directInputHoldBass = directInputHoldBass.getToggleState();
    modelConfig.directInputBassLow = directInputBassLow.getText().getIntValue();
    modelConfig.directInputBassHigh = directInputBassHigh.getText().getIntValue();
    modelConfig.directInputInitialBass = directInputInitialBass.getText().getIntValue();
    modelConfig.bufferInputPromptOnNextBar = bufferInputPromptOnNextBar.getToggleState();
    modelConfig.bufferInputForceOnNextBar = bufferInputForceOnNextBar.getToggleState();
    modelConfig.bufferInputSize = bufferInputSize.getText().getIntValue();
    modelConfig.bufferInputLow = bufferInputLow.getText().getIntValue();
    modelConfig.bufferInputHigh = bufferInputHigh.getText().getIntValue();
    modelConfig.bufferInputBassLow = bufferInputBassLow.getText().getIntValue();
    modelConfig.bufferInputBassHigh = bufferInputBassHigh.getText().getIntValue();
    modelConfig.tradingInputNumberOfBars = tradingInputNumberOfBars.getText().getIntValue();
    modelConfig.playbackInputStartTime = playbackInputStartTime.getText().getIntValue();
    modelConfig.tradingInputInitialOffset = tradingInputInitialOffset.getText().getIntValue();
    modelConfig.playbackInputEndTime = playbackInputEndTime.getText().getIntValue();
    modelConfig.playbackInputControlSequence = playbackInputControlSequence.getText();
    modelConfig.tradingInputModelGoesFirst = tradingInputModelGoesFirst.getToggleState();
    modelConfig.outputMinimumDuration = outputMinimumDuration.getText().getIntValue();
    modelConfig.outputMaximumDuration = outputMaximumDuration.getText().getIntValue();
    modelConfig.outputMinimumTimeInterval = outputMinimumTimeInterval.getText().getIntValue();
    modelConfig.outputAllowChords = outputAllowChords.getToggleState();
    modelConfig.outputMaximumChordSize = outputMaximumChordSize.getText().getIntValue();
    modelConfig.outputSendBarSeparators = outputSendBarSeparators.getToggleState();
    modelConfig.outputBarLength = outputBarLength.getText().getIntValue();
    modelConfig.outputTemperatures[0] = outputTemperatureTime.getText().getFloatValue();
    modelConfig.outputTemperatures[1] = outputTemperatureDuration.getText().getFloatValue();
    modelConfig.outputTemperatures[2] = outputTemperatureNote.getText().getFloatValue();
    modelConfig.outputStartTime = outputStartTime.getText().getIntValue();
    modelConfig.outputForceStartTime = outputForceStartTime.getToggleState();
    modelConfig.outputPauseAfterTimeInterval = outputPauseAfterTimeInterval.getToggleState();
    modelConfig.outputPauseAfterTimeIntervalValue = outputPauseAfterTimeIntervalValue.getText().getIntValue();
    modelConfig.outputPauseAfterDuration = outputPauseAfterDuration.getToggleState();
    modelConfig.outputPauseAfterDurationValue = outputPauseAfterDurationValue.getText().getIntValue();
    modelConfig.outputPauseAfterTradingSequence = outputPauseAfterTradingSequence.getToggleState();

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

