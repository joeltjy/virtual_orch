#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/ui/ScrollableWindow.h"

enum InputMode {
    Direct,
    Buffer
};

enum class InputFilterType {
    Passthrough,
    PitchRangeSplit
};

struct OutputInstrumentConfig {
    /** Whether the instrument is active */
    bool active = false;

    /** Whether the instrument is monophonic */
    bool monophony = false;

    /** The lowest note that the instrument can play */
    int32_t low = 36;

    /** The highest note that the instrument can play */
    int32_t high = 120;
};

constexpr std::size_t compile_time_hash(const char *str) {
    std::size_t hash = 5381;
    while (*str) {
        hash = ((hash << 5) + hash) + *str++;
    }
    return hash;
}

enum ModelConfigParameter : size_t {
    INPUT_MODE = compile_time_hash("inputMode"),
    INPUT_FILTER_TYPE = compile_time_hash("inputFilterType"),
    INPUT_INSTRUMENT = compile_time_hash("inputInstrument"),
    INPUT_LOW = compile_time_hash("inputLow"),
    INPUT_HIGH = compile_time_hash("inputHigh"),
    INPUT_DURATION = compile_time_hash("inputDuration"),
    INPUT_INITIAL_DATA = compile_time_hash("inputInitialData"),
    FILTER_CONDITIONING_LOW = compile_time_hash("filterConditioningLow"),
    FILTER_CONDITIONING_HIGH = compile_time_hash("filterConditioningHigh"),

    DIRECT_INPUT_WINDOW_LENGTH = compile_time_hash("directInputWindowLength"),
    DIRECT_INPUT_SEND_NOTE_OFFS = compile_time_hash("directInputSendNoteOffs"),
    DIRECT_INPUT_START_ON_INPUT = compile_time_hash("directInputStartOnInput"),
    DIRECT_INPUT_START_DELAY = compile_time_hash("directInputStartDelay"),
    DIRECT_INPUT_HOLD_BASS = compile_time_hash("directInputHoldBass"),
    DIRECT_INPUT_BASS_LOW = compile_time_hash("directInputBassLow"),
    DIRECT_INPUT_BASS_HIGH = compile_time_hash("directInputBassHigh"),
    DIRECT_INPUT_INITIAL_BASS = compile_time_hash("directInputInitialBass"),

    BUFFER_INPUT_SIZE = compile_time_hash("bufferInputSize"),
    BUFFER_INPUT_LOW = compile_time_hash("bufferInputLow"),
    BUFFER_INPUT_HIGH = compile_time_hash("bufferInputHigh"),
    BUFFER_INPUT_BASS_LOW = compile_time_hash("bufferInputBassLow"),
    BUFFER_INPUT_BASS_HIGH = compile_time_hash("bufferInputBassHigh"),

    OUTPUT_MINIMUM_DURATION = compile_time_hash("outputMinimumDuration"),
    OUTPUT_MAXIMUM_DURATION = compile_time_hash("outputMaximumDuration"),
    OUTPUT_TEMPERATURE_TIME = compile_time_hash("outputTemperatureTime"),
    OUTPUT_TEMPERATURE_DURATION = compile_time_hash("outputTemperatureDuration"),
    OUTPUT_TEMPERATURE_NOTE = compile_time_hash("outputTemperatureNote"),
    OUTPUT_START_TIME = compile_time_hash("outputStartTime"),
    OUTPUT_FORCE_START_TIME = compile_time_hash("outputForceStartTime"),
    OUTPUT_MAX_AHEAD_SECONDS = compile_time_hash("outputMaxAheadSeconds"),
    OUTPUT_AHEAD_THROTTLE_SECONDS = compile_time_hash("outputAheadThrottleSeconds"),

    OUTPUT_INSTRUMENT_ACTIVE = compile_time_hash("outputInstrumentActive"),
    OUTPUT_INSTRUMENT_MONOPHONY = compile_time_hash("outputInstrumentMonophony"),
    OUTPUT_INSTRUMENT_LOW = compile_time_hash("outputInstrumentLow"),
    OUTPUT_INSTRUMENT_HIGH = compile_time_hash("outputInstrumentHigh")
};

struct ModelConfig {
    /** The input mode of the model. */
    InputMode inputMode = Direct;

    /** How MIDI tokens are split into token vs conditioning queues. */
    InputFilterType inputFilterType = InputFilterType::Passthrough;

    /** Fixed reduction-model instrument embedding (0). OT retags keyboard as piano (15). */
    int32_t inputInstrument = 0;

    /** The range of the input */
    int32_t inputLow = 36;
    int32_t inputHigh = 120;

    /** The length (1/100s) of each note given as input. */
    int32_t inputDuration = 400;

    /** The initial input data */
    juce::String inputInitialData = "";

    /** Notes of reduction history fed to AMT/Dense ONNX (not OT; OT uses maxContextLength, default 256). */
    int32_t reductionContextNotes = 160;

    /** PitchRangeSplit: pitches in [low, high) go to conditioning. */
    int32_t filterConditioningLow = 24;
    int32_t filterConditioningHigh = 36;

    /** DIRECT INPUT: The length (1/100s) of the window to trigger input. */
    int32_t directInputWindowLength = 10;

    /** DIRECT INPUT: Whether to send Note Off signals. */
    bool directInputSendNoteOffs = true;

    /** DIRECT INPUT: Start Music Transformer on input. */
    bool directInputStartOnInput = false;

    /** DIRECT INPUT: Delay for starting Music Transformer after input. */
    int32_t directInputStartDelay = 0;

    /** DIRECT INPUT: Hold bass input */
    bool directInputHoldBass = false;

    /** DIRECT INPUT: The range of the bass zone */
    int32_t directInputBassLow = 24;
    int32_t directInputBassHigh = 36;

    /** DIRECT INPUT: The initial held bass */
    int32_t directInputInitialBass = -1;

    /** BUFFER INPUT: The size of the buffer to store input data. */
    int32_t bufferInputSize = 6;

    /** BUFFER INPUT: The range of the buffer */
    int32_t bufferInputLow = 36;
    int32_t bufferInputHigh = 120;

    /** BUFFER INPUT: The range of the bass zone */
    int32_t bufferInputBassLow = 24;
    int32_t bufferInputBassHigh = 36;

    /** The minimum length (1/100s) of each generated note. */
    int32_t outputMinimumDuration = 1;

    /** The maximum length (1/100s) of each generated note. */
    int32_t outputMaximumDuration = 200;

    /** The temperature of the model */
    std::array<double, 3> outputTemperatures{0.2, 0.2, 0.2};

    /** The start time of the model */
    int32_t outputStartTime = 0;

    /** Force to start at start time */
    bool outputForceStartTime = false;

    /**
     * ReductionTransformer only: soft-pause generation when a note is more than
     * this many seconds ahead of the clock (0 disables). Default 5.
     */
    int32_t outputMaxAheadSeconds = 5;

    /** ReductionTransformer only: pause duration (seconds) when ahead. Default 2. */
    int32_t outputAheadThrottleSeconds = 2;

    /** The instruments that the model can generate (indexed by MIDI id). */
    std::map<int32_t, OutputInstrumentConfig> outputInstruments{};

    /** The sorted active instruments (this is a derived property used by [MusicTransformer]). */
    std::set<int32_t> sortedActiveOutputInstruments{};

    auto getOutputInstrumentsIds() const -> std::vector<int32_t> {
        std::vector<int32_t> ids;
        for (const auto &instrument: outputInstruments) {
            ids.push_back(instrument.first);
        }
        return ids;
    }

    auto loadFromPreset(const juce::var &parsedJson) -> void {
        // INPUT: Set input mode
        juce::String inputModeStr = parsedJson.getProperty("inputMode", "direct").toString();
        if (inputModeStr == "direct") {
            inputMode = InputMode::Direct;
        } else if (inputModeStr == "buffer") {
            inputMode = InputMode::Buffer;
        }

        juce::String filterTypeStr = parsedJson.getProperty("inputFilterType", "passthrough").toString();
        if (filterTypeStr == "pitchRangeSplit") {
            inputFilterType = InputFilterType::PitchRangeSplit;
        } else {
            inputFilterType = InputFilterType::Passthrough;
        }

        // Reduction model input instrument is always 0 (dense MaxInstr=5). OT labels keyboard as piano.
        inputInstrument = 0;

        // INPUT: Set Input's low range
        inputLow = parsedJson.getProperty("inputLow", 36);

        // INPUT: Set Input's high range
        inputHigh = parsedJson.getProperty("inputHigh", 120);

        // INPUT: Set duration
        inputDuration = parsedJson.getProperty("inputDuration", 400);

        // INPUT: Set initial data
        inputInitialData = parsedJson.getProperty("inputInitialData", "");

        reductionContextNotes = juce::jmax(
            1, static_cast<int32_t>(parsedJson.getProperty("contextNotes", 160)));

        filterConditioningLow = parsedJson.getProperty("filterConditioningLow", 24);
        filterConditioningHigh = parsedJson.getProperty("filterConditioningHigh", 36);

        // DIRECT INPUT: Set Direct Input's window length
        directInputWindowLength = parsedJson.getProperty("directInputWindowLength", 10);

        // DIRECT INPUT: Set Direct Input's send note offs
        directInputSendNoteOffs = parsedJson.getProperty("directInputSendNoteOffs", true);

        // DIRECT INPUT: Set Direct Input's start on input
        directInputStartOnInput = parsedJson.getProperty("directInputStartOnInput", false);

        // DIRECT INPUT: Set Direct Input's start delay
        directInputStartDelay = parsedJson.getProperty("directInputStartDelay", 0);

        // DIRECT INPUT: Set Direct Input's hold bass
        directInputHoldBass = parsedJson.getProperty("directInputHoldBass", false);

        // DIRECT INPUT: Set DIRECT Input's bass low range
        directInputBassLow = parsedJson.getProperty("directInputBassLow", 24);

        // DIRECT INPUT: Set DIRECT Input's bass high range
        directInputBassHigh = parsedJson.getProperty("directInputBassHigh", 36);

        // DIRECT INPUT: Set DIRECT Input's bass high range
        directInputInitialBass = parsedJson.getProperty("directInputInitialBass", -1);

        // BUFFER INPUT: Set Buffer Input's buffer size
        bufferInputSize = parsedJson.getProperty("bufferInputSize", 6);

        // BUFFER INPUT: Set Buffer Input's low range
        bufferInputLow = parsedJson.getProperty("bufferInputLow", 36);

        // BUFFER INPUT: Set Buffer Input's high range
        bufferInputHigh = parsedJson.getProperty("bufferInputHigh", 120);

        // BUFFER INPUT: Set Buffer Input's bass low range
        bufferInputBassLow = parsedJson.getProperty("bufferInputBassLow", 24);

        // BUFFER INPUT: Set Buffer Input's bass high range
        bufferInputBassHigh = parsedJson.getProperty("bufferInputBassHigh", 36);

        // OUTPUT: Set minimum duration
        outputMinimumDuration = parsedJson.getProperty("outputMinimumDuration", 1);

        // OUTPUT: Set maximum duration
        outputMaximumDuration = parsedJson.getProperty("outputMaximumDuration", 200);

        // OUTPUT: Set temperatures
        auto jsonOutputTemperatures = parsedJson.getProperty("outputTemperatures", juce::Array<var>{.2, .2, .2});
        outputTemperatures[0] = jsonOutputTemperatures[0];
        outputTemperatures[1] = jsonOutputTemperatures[1];
        outputTemperatures[2] = jsonOutputTemperatures[2];

        // OUTPUT: Set start time
        outputStartTime = parsedJson.getProperty("outputStartTime", 0);

        // OUTPUT: Set force start time
        outputForceStartTime = parsedJson.getProperty("outputForceStartTime", false);

        // OUTPUT: Max seconds ahead of clock before soft-pause throttle
        outputMaxAheadSeconds = parsedJson.getProperty("outputMaxAheadSeconds", 5);

        outputAheadThrottleSeconds = parsedJson.getProperty("outputAheadThrottleSeconds", 2);

        // OUTPUT: Set instruments (optional in defaultPreset — empty {} must not crash)
        auto *jsonOutputInstruments = parsedJson.getProperty("outputInstruments", var()).getArray();
        sortedActiveOutputInstruments.clear();
        if (jsonOutputInstruments != nullptr) {
            for (auto &instrument: *jsonOutputInstruments) {
                int32_t instrumentId = instrument.getProperty("id", 0);
                if (!outputInstruments.contains(instrumentId)) {
                    DBG("Instrument " + std::to_string(instrumentId) + " is not available in the model config!");
                    continue;
                }
                outputInstruments[instrumentId].active = instrument.getProperty("active", false);
                outputInstruments[instrumentId].monophony = instrument.getProperty("monophony", false);
                outputInstruments[instrumentId].low = instrument.getProperty("low", 36);
                outputInstruments[instrumentId].high = instrument.getProperty("high", 120);
                if (outputInstruments[instrumentId].active) {
                    sortedActiveOutputInstruments.emplace(instrumentId);
                }
            }
        }
    }

    void updateParameter(const juce::String &parameterName, const juce::String &parameterValue);

    void updateParameter(const juce::String &parameterName, int32_t parameterValue);

    void updateParameter(const juce::String &parameterName, int32_t outputInstrumentId, int32_t parameterValue);
};

class ModelConfigurationComponent : public ScrollableComponent {
public:
    ModelConfigurationComponent(ModelConfig &modelConfig);

    ~ModelConfigurationComponent() override;

    void paint(juce::Graphics &) override;

    void resized() override;

    void userTriedToCloseWindow() override;

    bool apply() override;

private:
    ModelConfig &modelConfig;

    /* INPUT */
    juce::Label inputTitle, inputModeLabel, inputFilterTypeLabel, inputInstrumentLabel, inputLowLabel, inputHighLabel,
            inputDurationLabel,
            inputInitialDataLabel, filterConditioningLowLabel, filterConditioningHighLabel, directInputWindowLengthLabel,
            directInputSendNoteOffsLabel, directInputStartOnInputLabel,
            directInputStartDelayLabel,
            directInputHoldBassLabel, directInputBassLowLabel, directInputBassHighLabel, directInputInitialBassLabel,
            bufferInputSizeLabel, bufferInputLowLabel,
            bufferInputHighLabel, bufferInputBassLowLabel, bufferInputBassHighLabel;
    juce::ComboBox inputMode, inputFilterType;
    juce::TextEditor inputInstrument, inputLow, inputHigh, inputDuration, inputInitialData,
            filterConditioningLow, filterConditioningHigh, directInputWindowLength,
            directInputStartDelay,
            directInputBassLow, directInputBassHigh, directInputInitialBass, bufferInputSize, bufferInputLow,
            bufferInputHigh, bufferInputBassLow, bufferInputBassHigh;
    juce::ToggleButton directInputSendNoteOffs,
            directInputStartOnInput, directInputHoldBass;

    /* OUTPUT */
    juce::Label outputTitle, outputMinimumDurationLabel, outputMaximumDurationLabel,
            outputInstrumentsLabel,
            outputTemperatureTimeLabel, outputTemperatureDurationLabel, outputTemperatureNoteLabel,
            outputStartTimeLabel, outputMaxAheadSecondsLabel, outputAheadThrottleSecondsLabel;
    juce::TextEditor outputMinimumDuration, outputMaximumDuration,
            outputTemperatureTime, outputTemperatureDuration, outputTemperatureNote, outputStartTime,
            outputMaxAheadSeconds, outputAheadThrottleSeconds;
    juce::ToggleButton outputForceStartTime;
    juce::OwnedArray<juce::ToggleButton> outputInstruments;
    juce::OwnedArray<juce::ToggleButton> outputInstrumentsMonophony;
    juce::OwnedArray<juce::TextEditor> outputInstrumentsRangeLow, outputInstrumentsRangeHigh;
    juce::OwnedArray<juce::Label> outputInstrumentsRangeLowLabel, outputInstrumentsRangeHighLabel;

    juce::TextButton okButton{"OK"}, applyButton{"Apply"}, cancelButton{"Cancel"};
};
