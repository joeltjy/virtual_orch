#include "JordanAI/ModelConfigurationComponent.h"

void ModelConfig::updateParameter(const juce::String &parameterName, const juce::String &parameterValue) {
    switch (compile_time_hash(parameterName.toStdString().c_str())) {
        case ModelConfigParameter::INPUT_MODE:
            if (parameterValue == "Direct") {
                DBG("CGB: inputMode set to Direct");
                inputMode = Direct;
            } else if (parameterValue == "Buffer") {
                DBG("CGB: inputMode set to Buffer");
                inputMode = Buffer;
            } else if (parameterValue == "Trading") {
                DBG("CGB: inputMode set to Trading");
                inputMode = Trading;
            } else if (parameterValue == "Playback") {
                DBG("CGB: inputMode set to Playback");
                inputMode = Playback;
            } else {
                throw std::runtime_error("Unknown InputMode value: " + parameterValue.toStdString());
            }
            break;
        case ModelConfigParameter::INPUT_INITIAL_DATA:
            DBG("CGB: inputInitialData set to " + parameterValue);
            inputInitialData = parameterValue;
            break;
        case ModelConfigParameter::DIRECT_INPUT_MANUAL_PAUSE_CONTROL_TYPE:
            if (parameterValue == "note") {
                DBG("CGB: directInputManualPauseControlType set to Note");
                directInputManualPauseControlType = ControlType::Note;
            } else if (parameterValue == "CC") {
                DBG("CGB: directInputManualPauseControlType set to CC");
                directInputManualPauseControlType = ControlType::CC;
            } else if (parameterValue == "oscController") {
                DBG("CGB: directInputManualPauseControlType set to OscController");
                directInputManualPauseControlType = ControlType::OscController;
            } else {
                throw std::runtime_error("Unknown ControlType value: " + parameterValue.toStdString());
            }
            break;
        case ModelConfigParameter::DIRECT_INPUT_MANUAL_PAUSE_CONTROL_TRIGGER_TYPE:
            if (parameterValue == "toggle") {
                DBG("CGB: directInputManualPauseControlTriggerType set to Toggle");
                directInputManualPauseControlTriggerType = ControlTriggerType::Toggle;
            } else if (parameterValue == "momentary") {
                DBG("CGB: directInputManualPauseControlTriggerType set to Momentary");
                directInputManualPauseControlTriggerType = ControlTriggerType::Momentary;
            } else {
                throw std::runtime_error("Unknown ControlTriggerType value: " + parameterValue.toStdString());
            }
            break;
        case ModelConfigParameter::PLAYBACK_INPUT_CONTROL_SEQUENCE:
            DBG("CGB: playbackInputControlSequence set to " + parameterValue);
            playbackInputControlSequence = parameterValue;
            break;
        default:
            throw std::runtime_error("Unknown string parameter: " + parameterName.toStdString());
    }
}

void ModelConfig::updateParameter(const juce::String &parameterName, const int32_t parameterValue) {
    switch (compile_time_hash(parameterName.toStdString().c_str())) {
        // ===== INPUT =====
        case ModelConfigParameter::INPUT_INSTRUMENT:
            DBG("CGB: inputInstrument set to " + std::to_string(parameterValue));
            inputInstrument = parameterValue;
            break;
        case ModelConfigParameter::INPUT_LOW:
            DBG("CGB: inputLow set to " + std::to_string(parameterValue));
            inputLow = parameterValue;
            break;
        case ModelConfigParameter::INPUT_HIGH:
            DBG("CGB: inputHigh set to " + std::to_string(parameterValue));
            inputHigh = parameterValue;
            break;
        case ModelConfigParameter::INPUT_DURATION:
            DBG("CGB: inputDuration set to " + std::to_string(parameterValue));
            inputDuration = parameterValue;
            break;
        case ModelConfigParameter::INPUT_CLEARS_PAST:
            DBG("CGB: inputClearsPast set to " + std::to_string(parameterValue));
            inputClearsPast = parameterValue;
            break;
        case ModelConfigParameter::INPUT_CLICKS:
            DBG("CGB: inputClicks set to " + std::to_string(parameterValue));
            inputClicks = parameterValue;
            break;

        // ===== DIRECT INPUT =====
        case ModelConfigParameter::DIRECT_INPUT_WINDOW_LENGTH:
            DBG("CGB: directInputWindowLength set to " + std::to_string(parameterValue));
            directInputWindowLength = parameterValue;
            break;
        case ModelConfigParameter::DIRECT_INPUT_SEND_NOTE_OFFS:
            DBG("CGB: directInputSendNoteOffs set to " + std::to_string(parameterValue));
            directInputSendNoteOffs = parameterValue;
            break;
        case ModelConfigParameter::DIRECT_INPUT_SNAP_ON_BAR:
            DBG("CGB: directInputSnapOnBar set to " + std::to_string(parameterValue));
            directInputSnapOnBar = parameterValue;
            break;
        case ModelConfigParameter::DIRECT_INPUT_START_ON_INPUT:
            DBG("CGB: directInputStartOnInput set to " + std::to_string(parameterValue));
            directInputStartOnInput = parameterValue;
            break;
        case ModelConfigParameter::DIRECT_INPUT_START_DELAY:
            DBG("CGB: directInputStartDelay set to " + std::to_string(parameterValue));
            directInputStartDelay = parameterValue;
            break;
        case ModelConfigParameter::DIRECT_INPUT_UNPAUSE_ON_INPUT:
            DBG("CGB: directInputUnpauseOnInput set to " + std::to_string(parameterValue));
            directInputUnpauseOnInput = parameterValue;
            break;
        case ModelConfigParameter::DIRECT_INPUT_UNPAUSE_DELAY:
            DBG("CGB: directInputUnpauseDelay set to " + std::to_string(parameterValue));
            directInputUnpauseDelay = parameterValue;
            break;
        case ModelConfigParameter::DIRECT_INPUT_MANUAL_PAUSE:
            DBG("CGB: directInputManualPause set to " + std::to_string(parameterValue));
            directInputManualPause = parameterValue;
            break;
        case ModelConfigParameter::DIRECT_INPUT_MANUAL_PAUSE_CONTROL_ID:
            DBG("CGB: directInputManualPauseControlId set to " + std::to_string(parameterValue));
            directInputManualPauseControlId = parameterValue;
            break;
        case ModelConfigParameter::DIRECT_INPUT_MANUAL_PAUSE_CLEARS_FUTURE_NOTES:
            DBG("CGB: directInputManualPauseClearsFutureNotes set to " + std::to_string(parameterValue));
            directInputManualPauseClearsFutureNotes = parameterValue;
            break;
        case ModelConfigParameter::DIRECT_INPUT_HOLD_BASS:
            DBG("CGB: directInputHoldBass set to " + std::to_string(parameterValue));
            directInputHoldBass = parameterValue;
            break;
        case ModelConfigParameter::DIRECT_INPUT_BASS_LOW:
            DBG("CGB: directInputBassLow set to " + std::to_string(parameterValue));
            directInputBassLow = parameterValue;
            break;
        case ModelConfigParameter::DIRECT_INPUT_BASS_HIGH:
            DBG("CGB: directInputBassHigh set to " + std::to_string(parameterValue));
            directInputBassHigh = parameterValue;
            break;
        case ModelConfigParameter::DIRECT_INPUT_INITIAL_BASS:
            DBG("CGB: directInputInitialBass set to " + std::to_string(parameterValue));
            directInputInitialBass = parameterValue;
            break;

        // ===== BUFFER INPUT =====
        case ModelConfigParameter::BUFFER_INPUT_PROMPT_ON_NEXT_BAR:
            DBG("CGB: bufferInputPromptOnNextBar set to " + std::to_string(parameterValue));
            bufferInputPromptOnNextBar = parameterValue;
            break;
        case ModelConfigParameter::BUFFER_INPUT_FORCE_ON_NEXT_BAR:
            DBG("CGB: bufferInputForceOnNextBar set to " + std::to_string(parameterValue));
            bufferInputForceOnNextBar = parameterValue;
            break;
        case ModelConfigParameter::BUFFER_INPUT_SIZE:
            DBG("CGB: bufferInputSize set to " + std::to_string(parameterValue));
            bufferInputSize = parameterValue;
            break;
        case ModelConfigParameter::BUFFER_INPUT_LOW:
            DBG("CGB: bufferInputLow set to " + std::to_string(parameterValue));
            bufferInputLow = parameterValue;
            break;
        case ModelConfigParameter::BUFFER_INPUT_HIGH:
            DBG("CGB: bufferInputHigh set to " + std::to_string(parameterValue));
            bufferInputHigh = parameterValue;
            break;
        case ModelConfigParameter::BUFFER_INPUT_BASS_LOW:
            DBG("CGB: bufferInputBassLow set to " + std::to_string(parameterValue));
            bufferInputBassLow = parameterValue;
            break;
        case ModelConfigParameter::BUFFER_INPUT_BASS_HIGH:
            DBG("CGB: bufferInputBassHigh set to " + std::to_string(parameterValue));
            bufferInputBassHigh = parameterValue;
            break;

        // ===== TRADING INPUT =====
        case ModelConfigParameter::TRADING_INPUT_NUMBER_OF_BARS:
            DBG("CGB: tradingInputNumberOfBars set to " + std::to_string(parameterValue));
            tradingInputNumberOfBars = parameterValue;
            break;
        case ModelConfigParameter::TRADING_INPUT_MODEL_GOES_FIRST:
            DBG("CGB: tradingInputModelGoesFirst set to " + std::to_string(parameterValue));
            tradingInputModelGoesFirst = parameterValue;
            break;
        case ModelConfigParameter::TRADING_INPUT_INITIAL_OFFSET:
            DBG("CGB: tradingInputInitialOffset set to " + std::to_string(parameterValue));
            tradingInputInitialOffset = parameterValue;
            break;

        // ===== PLAYBACK INPUT =====
        case ModelConfigParameter::PLAYBACK_INPUT_START_TIME:
            DBG("CGB: playbackInputStartTime set to " + std::to_string(parameterValue));
            playbackInputStartTime = parameterValue;
            break;
        case ModelConfigParameter::PLAYBACK_INPUT_END_TIME:
            DBG("CGB: playbackInputEndTime set to " + std::to_string(parameterValue));
            playbackInputEndTime = parameterValue;
            break;

        // ===== OUTPUT =====
        case ModelConfigParameter::OUTPUT_MINIMUM_DURATION:
            DBG("CGB: outputMinimumDuration set to " + std::to_string(parameterValue));
            outputMinimumDuration = parameterValue;
            break;
        case ModelConfigParameter::OUTPUT_MAXIMUM_DURATION:
            DBG("CGB: outputMaximumDuration set to " + std::to_string(parameterValue));
            outputMaximumDuration = parameterValue;
            break;
        case ModelConfigParameter::OUTPUT_MINIMUM_TIME_INTERVAL:
            DBG("CGB: outputMinimumTimeInterval set to " + std::to_string(parameterValue));
            outputMinimumTimeInterval = parameterValue;
            break;
        case ModelConfigParameter::OUTPUT_ALLOW_CHORDS:
            DBG("CGB: outputAllowChords set to " + std::to_string(parameterValue));
            outputAllowChords = parameterValue;
            break;
        case ModelConfigParameter::OUTPUT_MAXIMUM_CHORD_SIZE:
            DBG("CGB: outputMaximumChordSize set to " + std::to_string(parameterValue));
            outputMaximumChordSize = parameterValue;
            break;
        case ModelConfigParameter::OUTPUT_SEND_BAR_SEPARATORS:
            DBG("CGB: outputSendBarSeparators set to " + std::to_string(parameterValue));
            outputSendBarSeparators = parameterValue;
            break;
        case ModelConfigParameter::OUTPUT_BAR_LENGTH:
            DBG("CGB: outputBarLength set to " + std::to_string(parameterValue));
            outputBarLength = parameterValue;
            break;
        case ModelConfigParameter::OUTPUT_TEMPERATURE_TIME:
            DBG("CGB: outputTemperatureTime set to " + std::to_string(parameterValue));
            outputTemperatures[0] = parameterValue;
            break;
        case ModelConfigParameter::OUTPUT_TEMPERATURE_DURATION:
            DBG("CGB: outputTemperatureDuration set to " + std::to_string(parameterValue));
            outputTemperatures[1] = parameterValue;
            break;
        case ModelConfigParameter::OUTPUT_TEMPERATURE_NOTE:
            DBG("CGB: outputTemperatureNote set to " + std::to_string(parameterValue));
            outputTemperatures[2] = parameterValue;
            break;
        case ModelConfigParameter::OUTPUT_START_TIME:
            DBG("CGB: outputStartTime set to " + std::to_string(parameterValue));
            outputStartTime = parameterValue;
            break;
        case ModelConfigParameter::OUTPUT_FORCE_START_TIME:
            DBG("CGB: outputForceStartTime set to " + std::to_string(parameterValue));
            outputForceStartTime = parameterValue;
            break;
        case ModelConfigParameter::OUTPUT_PAUSE_AFTER_TIME_INTERVAL:
            DBG("CGB: outputPauseAfterTimeInterval set to " + std::to_string(parameterValue));
            outputPauseAfterTimeInterval = parameterValue;
            break;
        case ModelConfigParameter::OUTPUT_PAUSE_AFTER_TIME_INTERVAL_VALUE:
            DBG("CGB: outputPauseAfterTimeIntervalValue set to " + std::to_string(parameterValue));
            outputPauseAfterTimeIntervalValue = parameterValue;
            break;
        case ModelConfigParameter::OUTPUT_PAUSE_AFTER_DURATION:
            DBG("CGB: outputPauseAfterDuration set to " + std::to_string(parameterValue));
            outputPauseAfterDuration = parameterValue;
            break;
        case ModelConfigParameter::OUTPUT_PAUSE_AFTER_DURATION_VALUE:
            DBG("CGB: outputPauseAfterDurationValue set to " + std::to_string(parameterValue));
            outputPauseAfterDurationValue = parameterValue;
            break;
        case ModelConfigParameter::OUTPUT_PAUSE_AFTER_TRADING_SEQUENCE:
            DBG("CGB: outputPauseAfterTradingSequence set to " + std::to_string(parameterValue));
            outputPauseAfterTradingSequence = parameterValue;
            break;

        default:
            throw std::runtime_error("Unknown bool/int32 parameter: " + parameterName.toStdString());
    }
}

void ModelConfig::updateParameter(const juce::String &parameterName, const int32_t outputInstrumentId,
                                  const int32_t parameterValue) {
    switch (compile_time_hash(parameterName.toStdString().c_str())) {
        case ModelConfigParameter::OUTPUT_INSTRUMENT_ACTIVE:
            if (!outputInstruments.contains(outputInstrumentId)) {
                throw std::runtime_error("Output instrument ID " + std::to_string(outputInstrumentId) +
                                         " does not exist in the model configuration.");
            }
            DBG("CGB: ID " + std::to_string(outputInstrumentId) + " active set to " + std::to_string(parameterValue));
            outputInstruments[outputInstrumentId].active = parameterValue;
            break;
        case ModelConfigParameter::OUTPUT_INSTRUMENT_MONOPHONY:
            if (!outputInstruments.contains(outputInstrumentId)) {
                throw std::runtime_error("Output instrument ID " + std::to_string(outputInstrumentId) +
                                         " does not exist in the model configuration.");
            }
            DBG("CGB: ID " + std::to_string(outputInstrumentId) + " monophony set to " + std::to_string(parameterValue))
            ;
            outputInstruments[outputInstrumentId].monophony = parameterValue;
            break;
        case ModelConfigParameter::OUTPUT_INSTRUMENT_LOW:
            if (!outputInstruments.contains(outputInstrumentId)) {
                throw std::runtime_error("Output instrument ID " + std::to_string(outputInstrumentId) +
                                         " does not exist in the model configuration.");
            }
            DBG("CGB: ID " + std::to_string(outputInstrumentId) + " low set to " + std::to_string(parameterValue));
            outputInstruments[outputInstrumentId].low = parameterValue;
            break;
        case ModelConfigParameter::OUTPUT_INSTRUMENT_HIGH:
            if (!outputInstruments.contains(outputInstrumentId)) {
                throw std::runtime_error("Output instrument ID " + std::to_string(outputInstrumentId) +
                                         " does not exist in the model configuration.");
            }
            DBG("CGB: ID " + std::to_string(outputInstrumentId) + " high set to " + std::to_string(parameterValue));
            outputInstruments[outputInstrumentId].high = parameterValue;
            break;
        default:
            throw std::runtime_error("Unknown output instrument bool/int32 parameter: " + parameterName.toStdString());
    }
}
