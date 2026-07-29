#pragma once

#include <JuceHeader.h>

#include "ScrollableWindow.h"

enum InputMode {
    Direct,
    Buffer,
    Trading,
    Playback
};

enum ControlType {
    Note,
    CC,
    OscController
};

enum ControlTriggerType {
    Toggle,
    Momentary
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
    INPUT_INSTRUMENT = compile_time_hash("inputInstrument"),
    INPUT_LOW = compile_time_hash("inputLow"),
    INPUT_HIGH = compile_time_hash("inputHigh"),
    INPUT_DURATION = compile_time_hash("inputDuration"),
    INPUT_CLEARS_PAST = compile_time_hash("inputClearsPast"),
    INPUT_CLICKS = compile_time_hash("inputClicks"),
    INPUT_INITIAL_DATA = compile_time_hash("inputInitialData"),

    DIRECT_INPUT_WINDOW_LENGTH = compile_time_hash("directInputWindowLength"),
    DIRECT_INPUT_MANUAL_TRIGGER = compile_time_hash("directInputManualTrigger"),
    DIRECT_INPUT_MANUAL_TRIGGER_CONTROL_TYPE = compile_time_hash("directInputManualTriggerControlType"),
    DIRECT_INPUT_MANUAL_TRIGGER_CONTROL_ID = compile_time_hash("directInputManualTriggerControlId"),
    DIRECT_INPUT_SEND_NOTE_OFFS = compile_time_hash("directInputSendNoteOffs"),
    DIRECT_INPUT_SNAP_ON_BAR = compile_time_hash("directInputSnapOnBar"),
    DIRECT_INPUT_START_ON_INPUT = compile_time_hash("directInputStartOnInput"),
    DIRECT_INPUT_START_DELAY = compile_time_hash("directInputStartDelay"),
    DIRECT_INPUT_UNPAUSE_ON_INPUT = compile_time_hash("directInputUnpauseOnInput"),
    DIRECT_INPUT_UNPAUSE_DELAY = compile_time_hash("directInputUnpauseDelay"),
    DIRECT_INPUT_MANUAL_PAUSE = compile_time_hash("directInputManualPause"),
    DIRECT_INPUT_MANUAL_PAUSE_CONTROL_TYPE = compile_time_hash("directInputManualPauseControlType"),
    DIRECT_INPUT_MANUAL_PAUSE_CONTROL_ID = compile_time_hash("directInputManualPauseControlId"),
    DIRECT_INPUT_MANUAL_PAUSE_CONTROL_TRIGGER_TYPE = compile_time_hash("directInputManualPauseControlTriggerType"),
    DIRECT_INPUT_MANUAL_PAUSE_CLEARS_FUTURE_NOTES = compile_time_hash("directInputManualPauseClearsFutureNotes"),
    DIRECT_INPUT_HOLD_BASS = compile_time_hash("directInputHoldBass"),
    DIRECT_INPUT_BASS_LOW = compile_time_hash("directInputBassLow"),
    DIRECT_INPUT_BASS_HIGH = compile_time_hash("directInputBassHigh"),
    DIRECT_INPUT_INITIAL_BASS = compile_time_hash("directInputInitialBass"),

    BUFFER_INPUT_PROMPT_ON_NEXT_BAR = compile_time_hash("bufferInputPromptOnNextBar"),
    BUFFER_INPUT_FORCE_ON_NEXT_BAR = compile_time_hash("bufferInputForceOnNextBar"),
    BUFFER_INPUT_SIZE = compile_time_hash("bufferInputSize"),
    BUFFER_INPUT_LOW = compile_time_hash("bufferInputLow"),
    BUFFER_INPUT_HIGH = compile_time_hash("bufferInputHigh"),
    BUFFER_INPUT_BASS_LOW = compile_time_hash("bufferInputBassLow"),
    BUFFER_INPUT_BASS_HIGH = compile_time_hash("bufferInputBassHigh"),

    TRADING_INPUT_NUMBER_OF_BARS = compile_time_hash("tradingInputNumberOfBars"),
    TRADING_INPUT_MODEL_GOES_FIRST = compile_time_hash("tradingInputModelGoesFirst"),
    TRADING_INPUT_INITIAL_OFFSET = compile_time_hash("tradingInputInitialOffset"),

    PLAYBACK_INPUT_START_TIME = compile_time_hash("playbackInputStartTime"),
    PLAYBACK_INPUT_END_TIME = compile_time_hash("playbackInputEndTime"),
    PLAYBACK_INPUT_CONTROL_SEQUENCE = compile_time_hash("playbackInputControlSequence"),

    OUTPUT_MINIMUM_DURATION = compile_time_hash("outputMinimumDuration"),
    OUTPUT_MAXIMUM_DURATION = compile_time_hash("outputMaximumDuration"),
    OUTPUT_MINIMUM_TIME_INTERVAL = compile_time_hash("outputMinimumTimeInterval"),
    OUTPUT_ALLOW_CHORDS = compile_time_hash("outputAllowChords"),
    OUTPUT_MAXIMUM_CHORD_SIZE = compile_time_hash("outputMaximumChordSize"),
    OUTPUT_SEND_BAR_SEPARATORS = compile_time_hash("outputSendBarSeparators"),
    OUTPUT_BAR_LENGTH = compile_time_hash("outputBarLength"),
    OUTPUT_TEMPERATURE_TIME = compile_time_hash("outputTemperatureTime"),
    OUTPUT_TEMPERATURE_DURATION = compile_time_hash("outputTemperatureDuration"),
    OUTPUT_TEMPERATURE_NOTE = compile_time_hash("outputTemperatureNote"),
    OUTPUT_START_TIME = compile_time_hash("outputStartTime"),
    OUTPUT_FORCE_START_TIME = compile_time_hash("outputForceStartTime"),
    OUTPUT_PAUSE_AFTER_TIME_INTERVAL = compile_time_hash("outputPauseAfterTimeInterval"),
    OUTPUT_PAUSE_AFTER_TIME_INTERVAL_VALUE = compile_time_hash("outputPauseAfterTimeIntervalValue"),
    OUTPUT_PAUSE_AFTER_DURATION = compile_time_hash("outputPauseAfterDuration"),
    OUTPUT_PAUSE_AFTER_DURATION_VALUE = compile_time_hash("outputPauseAfterDurationValue"),
    OUTPUT_PAUSE_AFTER_TRADING_SEQUENCE = compile_time_hash("outputPauseAfterTradingSequence"),

    OUTPUT_INSTRUMENT_ACTIVE = compile_time_hash("outputInstrumentActive"),
    OUTPUT_INSTRUMENT_MONOPHONY = compile_time_hash("outputInstrumentMonophony"),
    OUTPUT_INSTRUMENT_LOW = compile_time_hash("outputInstrumentLow"),
    OUTPUT_INSTRUMENT_HIGH = compile_time_hash("outputInstrumentHigh")
};

struct ModelConfig {
    /** The input mode of the model. */
    InputMode inputMode = Direct;

    /** The MIDI instrument used to condition the model. */
    int32_t inputInstrument = 0;

    /** The range of the input */
    int32_t inputLow = 36;
    int32_t inputHigh = 120;

    /** The length (1/100s) of each note given as input. */
    int32_t inputDuration = 400;

    /** Whether inputing a note clears the previous input data. */
    bool inputClearsPast = false;

    /** Whether to send click signals */
    bool inputClicks = false;

    /** The initial input data */
    juce::String inputInitialData = "";

    /** The refresh rate of the initial input data (0 means no refresh). */
    int32_t inputInitialDataRefreshRate = 0;

    /** DIRECT INPUT: The length (1/100s) of the window to trigger input. */
    int32_t directInputWindowLength = 10;

    /** DIRECT INPUT: Whether to trigger the input by pressing a note. */
    bool directInputManualTrigger = false;

    /** DIRECT INPUT: Whether to use Note or Continuous Control to trigger direct input. */
    ControlType directInputManualTriggerControlType = ControlType::Note;

    /** DIRECT INPUT: If `directInputManualTrigger` is `true`, the MIDI id of the control used to trigger the input. */
    int32_t directInputManualTriggerControlId = 24;

    /** DIRECT INPUT: Whether to send Note Off signals. */
    bool directInputSendNoteOffs = false;

    /** DIRECT INPUT: Snap input on closest bar. */
    bool directInputSnapOnBar = false;

    /** DIRECT INPUT: Start Music Transformer on input. */
    bool directInputStartOnInput = false;

    /** DIRECT INPUT: Delay for starting Music Transformer after input. */
    int32_t directInputStartDelay = 0;

    /** DIRECT INPUT: Unpause Music Transformer on input. */
    bool directInputUnpauseOnInput = false;

    /** DIRECT INPUT: Delay for unpausing Music Transformer after input. */
    int32_t directInputUnpauseDelay = 0;

    /** DIRECT INPUT: Manual Pausing and Unpausing of the Music Transformer */
    bool directInputManualPause = false;

    /** DIRECT INPUT: Wheter to use Note or Continuous Control to trigger manual pause. */
    ControlType directInputManualPauseControlType = ControlType::Note;

    /** DIRECT INPUT: The MIDI id of the control used to trigger manual pause. */
    int32_t directInputManualPauseControlId = 0;

    /** DIRECT INPUT: The trigger type of the control used to trigger manual pause. */
    ControlTriggerType directInputManualPauseControlTriggerType = ControlTriggerType::Toggle;

    /** DIRECT INPUT: Whether a pause clears future notes. */
    bool directInputManualPauseClearsFutureNotes = true;

    /** DIRECT INPUT: Hold bass input */
    bool directInputHoldBass = false;

    /** DIRECT INPUT: The range of the bass zone */
    int32_t directInputBassLow = 24;
    int32_t directInputBassHigh = 36;

    /** DIRECT INPUT: The initial held bass */
    int32_t directInputInitialBass = -1;

    /** BUFFER INPUT: Prompt on next bar. */
    bool bufferInputPromptOnNextBar = false;

    /** BUFFER INPUT: Force instruments to play on next bar. */
    bool bufferInputForceOnNextBar = false;

    /** BUFFER INPUT: The size of the buffer to store input data. */
    int32_t bufferInputSize = 6;

    /** BUFFER INPUT: The range of the buffer */
    int32_t bufferInputLow = 36;
    int32_t bufferInputHigh = 120;

    /** BUFFER INPUT: The range of the bass zone */
    int32_t bufferInputBassLow = 24;
    int32_t bufferInputBassHigh = 36;

    /** TRADING INPUT: The number of bars of input and output to alternate between. */
    int32_t tradingInputNumberOfBars = 4;

    /** TRADING INPUT: Whether the model goes first. */
    bool tradingInputModelGoesFirst = true;

    /** TRADING INPUT: The number of bars of input and output to alternate between. */
    int32_t tradingInputInitialOffset = 0;

    /** PLAYBACK INPUT: The start time of the playback. */
    int32_t playbackInputStartTime = 0;

    /** PLAYBACK INPUT: The end time of the playback. */
    int32_t playbackInputEndTime = 800;

    /** PLAYBACK INPUT: The control sequence (space-separated). */
    juce::String playbackInputControlSequence = "";

    /** The minimum length (1/100s) of each generated note. */
    int32_t outputMinimumDuration = 1;

    /** The maximum length (1/100s) of each generated note. */
    int32_t outputMaximumDuration = 999;

    /** The minimum amount of time (1/100s) between each generated note. */
    int32_t outputMinimumTimeInterval = 0;

    /** Whether to allow chords (useless when minimumTimeInterval == 0, overriden by monophony) */
    bool outputAllowChords = false;

    /** Max chord size */
    int32_t outputMaximumChordSize = 6;

    /** Whether to send Bar Separator signals */
    bool outputSendBarSeparators = false;

    /** The length (1/100s) of a bar */
    int32_t outputBarLength = 200;

    /** The temperature of the model */
    std::array<double, 3> outputTemperatures{0.2, 0.2, 0.2};

    /** The start time of the model */
    int32_t outputStartTime = 0;

    /** Force to start at start time */
    bool outputForceStartTime = false;

    /** Pause output after time inteval */
    bool outputPauseAfterTimeInterval = false;

    /** The time interval (1/100s) needed to trigger an output pause (only if outputPauseAfterTimeInterval is set to true) */
    int32_t outputPauseAfterTimeIntervalValue = 20;

    /** Pause output after duration */
    bool outputPauseAfterDuration = false;

    /** The duration (1/100s) needed to trigger an output pause (only if outputPauseAfterDuration is set to true) */
    int32_t outputPauseAfterDurationValue = 50;

    /** Pause output after trading sequence */
    bool outputPauseAfterTradingSequence = false;

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
        } else if (inputModeStr == "trading") {
            inputMode = InputMode::Trading;
        } else if (inputModeStr == "playback") {
            inputMode = InputMode::Playback;
        }

        // INPUT: Set instrument
        inputInstrument = parsedJson.getProperty("inputInstrument", 0);

        // INPUT: Set Input's low range
        inputLow = parsedJson.getProperty("inputLow", 36);

        // INPUT: Set Input's high range
        inputHigh = parsedJson.getProperty("inputHigh", 120);

        // INPUT: Set duration
        inputDuration = parsedJson.getProperty("inputDuration", 400);

        // INPUT: Set clear past
        inputClearsPast = parsedJson.getProperty("inputClearsPast", false);

        // INPUT: Set clicks
        inputClicks = parsedJson.getProperty("inputClicks", false);

        // INPUT: Set initial data
        inputInitialData = parsedJson.getProperty("inputInitialData", "");

        // INPUT: Set initial data refresh rate
        inputInitialDataRefreshRate = parsedJson.getProperty("inputInitialDataRefreshRate", 0);

        // DIRECT INPUT: Set Direct Input's window length
        directInputWindowLength = parsedJson.getProperty("directInputWindowLength", 10);

        // DIRECT INPUT: Set Direct Input's manual trigger
        directInputManualTrigger = parsedJson.getProperty("directInputManualTrigger", false);

        // DIRECT INPUT: Set Direct Input's manual trigger control type
        juce::String manualTriggerControlTypeStr = parsedJson.getProperty("directInputManualTriggerControlType",
                                                                        "note").toString();
        if (manualTriggerControlTypeStr == "note") {
            directInputManualTriggerControlType = ControlType::Note;
        } else if (manualTriggerControlTypeStr == "CC") {
            directInputManualTriggerControlType = ControlType::CC;
        } else if (manualTriggerControlTypeStr == "oscController") {
            directInputManualTriggerControlType = ControlType::OscController;
        }

        // DIRECT INPUT: Set Direct Input's manual trigger control id
        directInputManualTriggerControlId = parsedJson.getProperty("directInputManualTriggerControlId", 24);

        // DIRECT INPUT: Set Direct Input's send note offs
        directInputSendNoteOffs = parsedJson.getProperty("directInputSendNoteOffs", false);

        // DIRECT INPUT: Set Direct Input's snap on bar
        directInputSnapOnBar = parsedJson.getProperty("directInputSnapOnBar", false);

        // DIRECT INPUT: Set Direct Input's start on input
        directInputStartOnInput = parsedJson.getProperty("directInputStartOnInput", false);

        // DIRECT INPUT: Set Direct Input's start delay
        directInputStartDelay = parsedJson.getProperty("directInputStartDelay", 0);

        // DIRECT INPUT: Set Direct Input's unpause on input
        directInputUnpauseOnInput = parsedJson.getProperty("directInputUnpauseOnInput", false);

        // DIRECT INPUT: Set Direct Input's unpause delay
        directInputUnpauseDelay = parsedJson.getProperty("directInputUnpauseDelay", 0);

        // DIRECT INPUT: Set Direct Input's manual pause
        directInputManualPause = parsedJson.getProperty("directInputManualPause", false);

        // DIRECT INPUT: Set Direct Input's manual pause control type
        juce::String manualPauseControlTypeStr = parsedJson.getProperty("directInputManualPauseControlType",
                                                                        "note").toString();
        if (manualPauseControlTypeStr == "note") {
            directInputManualPauseControlType = ControlType::Note;
        } else if (manualPauseControlTypeStr == "CC") {
            directInputManualPauseControlType = ControlType::CC;
        } else if (manualPauseControlTypeStr == "oscController") {
            directInputManualPauseControlType = ControlType::OscController;
        }

        // DIRECT INPUT: Set Direct Input's manual pause control id
        directInputManualPauseControlId = parsedJson.getProperty("directInputManualPauseControlId", 0);

        // DIRECT INPUT: Set Direct Input's manual pause trigger type
        juce::String manualPauseControlTriggerTypeStr = parsedJson.getProperty(
            "directInputManualPauseControlTriggerType",
            "toggle").toString();
        if (manualPauseControlTriggerTypeStr == "toggle") {
            directInputManualPauseControlTriggerType = ControlTriggerType::Toggle;
        } else if (manualPauseControlTriggerTypeStr == "momentary") {
            directInputManualPauseControlTriggerType = ControlTriggerType::Momentary;
        }

        // DIRECT INPUT: Set Direct Input's manual pause clears future notes
        directInputManualPauseClearsFutureNotes = parsedJson.getProperty("directInputManualPauseClearsFutureNotes",
                                                                         true);

        // DIRECT INPUT: Set Direct Input's hold bass
        directInputHoldBass = parsedJson.getProperty("directInputHoldBass", false);

        // DIRECT INPUT: Set DIRECT Input's bass low range
        directInputBassLow = parsedJson.getProperty("directInputBassLow", 24);

        // DIRECT INPUT: Set DIRECT Input's bass high range
        directInputBassHigh = parsedJson.getProperty("directInputBassHigh", 36);

        // DIRECT INPUT: Set DIRECT Input's bass high range
        directInputInitialBass = parsedJson.getProperty("directInputInitialBass", -1);

        // BUFFER INPUT: Set Buffer Input's prompt on next bar
        bufferInputPromptOnNextBar = parsedJson.getProperty("bufferInputPromptOnNextBar", false);

        // BUFFER INPUT: Set Buffer Input's force on next bar
        bufferInputForceOnNextBar = parsedJson.getProperty("bufferInputForceOnNextBar", false);

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

        // TRADING INPUT: Set Number of Bars
        tradingInputNumberOfBars = parsedJson.getProperty("tradingInputNumberOfBars", 4);

        // TRADING INPUT: Set Model Goes First
        tradingInputModelGoesFirst = parsedJson.getProperty("tradingInputModelGoesFirst", true);

        // TRADING INPUT: Set Initial Offset
        tradingInputInitialOffset = parsedJson.getProperty("tradingInputInitialOffset", 4);

        // PLAYBACK INPUT: Set Start Time
        playbackInputStartTime = parsedJson.getProperty("playbackInputStartTime", 0);

        // PLAYBACK INPUT: Set Start Time
        playbackInputEndTime = parsedJson.getProperty("playbackInputEndTime", 10000);

        // PLAYBACK INPUT: Set Control Sequence
        playbackInputControlSequence = parsedJson.getProperty("playbackInputControlSequence", "");

        // OUTPUT: Set minimum duration
        outputMinimumDuration = parsedJson.getProperty("outputMinimumDuration", 1);

        // OUTPUT: Set maximum duration
        outputMaximumDuration = parsedJson.getProperty("outputMaximumDuration", 999);

        // OUTPUT: Set minimum time interval
        outputMinimumTimeInterval = parsedJson.getProperty("outputMinimumTimeInterval", 0);

        // OUTPUT: Set allow chords
        outputAllowChords = parsedJson.getProperty("outputAllowChords", false);

        // OUTPUT: Set maximum chord size
        outputMaximumChordSize = parsedJson.getProperty("outputMaximumChordSize", 6);

        // OUTPUT: Set send bar separators
        outputSendBarSeparators = parsedJson.getProperty("outputSendBarSeparators", false);

        // OUTPUT: Set bar length
        outputBarLength = parsedJson.getProperty("outputBarLength", 200);

        // OUTPUT: Set temperatures
        auto jsonOutputTemperatures = parsedJson.getProperty("outputTemperatures", juce::Array<var>{.2, .2, .2});
        outputTemperatures[0] = jsonOutputTemperatures[0];
        outputTemperatures[1] = jsonOutputTemperatures[1];
        outputTemperatures[2] = jsonOutputTemperatures[2];

        // OUTPUT: Set start time
        outputStartTime = parsedJson.getProperty("outputStartTime", 0);

        // OUTPUT: Set force start time
        outputForceStartTime = parsedJson.getProperty("outputForceStartTime", false);

        // OUTPUT: Set pause after time interval
        outputPauseAfterTimeInterval = parsedJson.getProperty("outputPauseAfterTimeInterval", false);

        // OUTPUT: Set pause after time interval value
        outputPauseAfterTimeIntervalValue = parsedJson.getProperty("outputPauseAfterTimeIntervalValue", 20);

        // OUTPUT: Set pause after duration
        outputPauseAfterDuration = parsedJson.getProperty("outputPauseAfterDuration", false);

        // OUTPUT: Set pause after duration value
        outputPauseAfterDurationValue = parsedJson.getProperty("outputPauseAfterDurationValue", 50);

        // OUTPUT: Set pause after time elapsed
        outputPauseAfterTradingSequence = parsedJson.getProperty("outputPauseAfterTradingSequence", false);

        // OUTPUT: Set instruments
        auto *jsonOutputInstruments = parsedJson.getProperty("outputInstruments", var()).getArray();
        sortedActiveOutputInstruments.clear();
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
    juce::Label inputTitle, inputModeLabel, inputInstrumentLabel, inputLowLabel, inputHighLabel, inputDurationLabel,
            inputClearsPastLabel, inputClicksLabel, inputInitialDataLabel, inputInitialDataRefreshRateLabel,
            directInputWindowLengthLabel, directInputManualTriggerLabel, directInputManualTriggerControlTypeLabel,
            directInputManualTriggerControlIdLabel, directInputSendNoteOffsLabel, directInputSnapOnBarLabel,
            directInputStartOnInputLabel, directInputStartDelayLabel, directInputUnpauseOnInputLabel,
            directInputUnpauseDelayLabel, directInputManualPauseLabel, directInputManualPauseClearsFutureNotesLabel,
            directInputHoldBassLabel, directInputBassLowLabel, directInputBassHighLabel, directInputInitialBassLabel,
            bufferInputPromptOnNextBarLabel, bufferInputForceOnNextBarLabel, bufferInputSizeLabel, bufferInputLowLabel,
            bufferInputHighLabel, bufferInputBassLowLabel, bufferInputBassHighLabel, tradingInputNumberOfBarsLabel,
            tradingInputInitialOffsetLabel, playbackInputStartTimeLabel, playbackInputEndTimeLabel,
            playbackInputControlSequenceLabel;
    juce::ComboBox inputMode, directInputManualTriggerControlType, directInputManualPauseControlType,
            directInputManualPauseControlTriggerType;
    juce::TextEditor inputInstrument, inputLow, inputHigh, inputDuration, inputInitialData, inputInitialDataRefreshRate,
            directInputWindowLength, directInputManualTriggerControlId, directInputStartDelay, directInputUnpauseDelay,
            directInputManualPauseControlId, directInputBassLow, directInputBassHigh, directInputInitialBass, bufferInputSize, bufferInputLow,
            bufferInputHigh, bufferInputBassLow, bufferInputBassHigh, tradingInputNumberOfBars,
            tradingInputInitialOffset, playbackInputStartTime, playbackInputEndTime, playbackInputControlSequence;
    juce::ToggleButton inputClearsPast, inputClicks, directInputManualTrigger, directInputManualPause,
            directInputManualPauseClearsFutureNotes, directInputSendNoteOffs, directInputSnapOnBar,
            directInputStartOnInput, directInputUnpauseOnInput, directInputHoldBass, bufferInputPromptOnNextBar,
            bufferInputForceOnNextBar, tradingInputModelGoesFirst;

    /* OUTPUT */
    juce::Label outputTitle, outputMinimumDurationLabel, outputMaximumDurationLabel, outputMinimumTimeIntervalLabel,
            outputMaximumChordSizeLabel, outputSendBarSeparatorsLabel, outputBarLengthLabel, outputInstrumentsLabel,
            outputTemperatureTimeLabel, outputTemperatureDurationLabel, outputTemperatureNoteLabel,
            outputStartTimeLabel, outputPauseAfterTimeIntervalLabel, outputPauseAfterTimeIntervalValueLabel,
            outputPauseAfterDurationLabel, outputPauseAfterDurationValueLabel, outputPauseAfterTradingSequenceLabel;
    juce::TextEditor outputMinimumDuration, outputMaximumDuration, outputMinimumTimeInterval, outputMaximumChordSize,
            outputBarLength, outputTemperatureTime, outputTemperatureDuration, outputTemperatureNote, outputStartTime,
            outputPauseAfterTimeIntervalValue, outputPauseAfterDurationValue;
    juce::ToggleButton outputSendBarSeparators, outputAllowChords, outputForceStartTime, outputPauseAfterTimeInterval,
            outputPauseAfterDuration, outputPauseAfterTradingSequence;
    juce::OwnedArray<juce::ToggleButton> outputInstruments;
    juce::OwnedArray<juce::ToggleButton> outputInstrumentsMonophony;
    juce::OwnedArray<juce::TextEditor> outputInstrumentsRangeLow, outputInstrumentsRangeHigh;
    juce::OwnedArray<juce::Label> outputInstrumentsRangeLowLabel, outputInstrumentsRangeHighLabel;

    juce::TextButton okButton{"OK"}, applyButton{"Apply"}, cancelButton{"Cancel"};
};
