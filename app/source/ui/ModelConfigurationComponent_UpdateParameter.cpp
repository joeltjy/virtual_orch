#include "VirtualOrch/ui/ModelConfigurationComponent.h"

void ModelConfig::updateParameter(const juce::String &parameterName, const juce::String &parameterValue) {
    switch (compile_time_hash(parameterName.toStdString().c_str())) {
        case ModelConfigParameter::INPUT_MODE:
            if (parameterValue == "Direct") {
                DBG("CGB: inputMode set to Direct");
                inputMode = Direct;
            } else if (parameterValue == "Buffer") {
                DBG("CGB: inputMode set to Buffer");
                inputMode = Buffer;
            } else {
                throw std::runtime_error("Unknown InputMode value: " + parameterValue.toStdString());
            }
            break;
        case ModelConfigParameter::INPUT_FILTER_TYPE:
            if (parameterValue.equalsIgnoreCase("passthrough")) {
                inputFilterType = InputFilterType::Passthrough;
            } else if (parameterValue.equalsIgnoreCase("pitchRangeSplit")) {
                inputFilterType = InputFilterType::PitchRangeSplit;
            } else {
                throw std::runtime_error("Unknown InputFilterType value: " + parameterValue.toStdString());
            }
            break;
        case ModelConfigParameter::INPUT_INITIAL_DATA:
            DBG("CGB: inputInitialData set to " + parameterValue);
            inputInitialData = parameterValue;
            break;
        default:
            throw std::runtime_error("Unknown string parameter: " + parameterName.toStdString());
    }
}

void ModelConfig::updateParameter(const juce::String &parameterName, const int32_t parameterValue) {
    switch (compile_time_hash(parameterName.toStdString().c_str())) {
        // ===== INPUT =====
        case ModelConfigParameter::INPUT_INSTRUMENT:
            // Reduction vocab requires instrument 0; OT retags keyboard as piano separately.
            DBG("CGB: inputInstrument forced to reduction input id (0)");
            inputInstrument = 0;
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
        case ModelConfigParameter::FILTER_CONDITIONING_LOW:
            DBG("CGB: filterConditioningLow set to " + std::to_string(parameterValue));
            filterConditioningLow = parameterValue;
            break;
        case ModelConfigParameter::FILTER_CONDITIONING_HIGH:
            DBG("CGB: filterConditioningHigh set to " + std::to_string(parameterValue));
            filterConditioningHigh = parameterValue;
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
        case ModelConfigParameter::DIRECT_INPUT_START_ON_INPUT:
            DBG("CGB: directInputStartOnInput set to " + std::to_string(parameterValue));
            directInputStartOnInput = parameterValue;
            break;
        case ModelConfigParameter::DIRECT_INPUT_START_DELAY:
            DBG("CGB: directInputStartDelay set to " + std::to_string(parameterValue));
            directInputStartDelay = parameterValue;
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

        // ===== OUTPUT =====
        case ModelConfigParameter::OUTPUT_MINIMUM_DURATION:
            DBG("CGB: outputMinimumDuration set to " + std::to_string(parameterValue));
            outputMinimumDuration = parameterValue;
            break;
        case ModelConfigParameter::OUTPUT_MAXIMUM_DURATION:
            DBG("CGB: outputMaximumDuration set to " + std::to_string(parameterValue));
            outputMaximumDuration = parameterValue;
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
        case ModelConfigParameter::OUTPUT_MAX_AHEAD_SECONDS:
            DBG("CGB: outputMaxAheadSeconds set to " + std::to_string(parameterValue));
            outputMaxAheadSeconds = parameterValue;
            break;
        case ModelConfigParameter::OUTPUT_AHEAD_THROTTLE_SECONDS:
            DBG("CGB: outputAheadThrottleSeconds set to " + std::to_string(parameterValue));
            outputAheadThrottleSeconds = parameterValue;
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
