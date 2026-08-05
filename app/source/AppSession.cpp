#include "VirtualOrch/AppSession.h"

#include "VirtualOrch/orchestration-models/OrchestrationModels.h"
#include "VirtualOrch/ui/UiConstants.h"

AppSession::AppSession()
    : presetStore(modelConfig),
      clock(metrics),
      musicTransformer(modelConfig),
      orchestrationModel(createOrchestrationModel("TestModel")),
      inputFilter(createInputFilter(modelConfig.inputFilterType,
                                    modelConfig,
                                    musicTransformer.inputTokenQueue,
                                    musicTransformer.inputConditioningQueue,
                                    musicTransformer.updatesFromFilter)),
      outputPlayback(clock, musicTransformer, outputProcessor, bufferOutputProcessor,
                     visualizationBufferSize),
      midiInputProcess(clock, musicTransformer, modelConfig, outputProcessor, inputFilter,
                       selectedMidiInputIdentifier, selectedLaunchpadMidiIdentifier,
                       selectedMtcClockIdentifier, mtcClockActive) {
    musicTransformer.orchestrationMidiIncoming = &orchestrationTransformer.midiInputIncoming;
    musicTransformer.orchestrationConditioningIncoming =
        &orchestrationTransformer.conditioningIncoming;
    musicTransformer.orchestrationUpdatesIncoming = &orchestrationTransformer.updatesIncoming;
    midiInputProcess.getLaunchpadGrid().instrumentUpdates =
        &orchestrationTransformer.instrumentUpdates;
    auto &pads = midiInputProcess.getLaunchpadGrid().padInstruments;
    pads[0][0] = 0;
    pads[0][1] = 1;
    pads[0][2] = 2;
    pads[0][3] = 3;
    pads[0][4] = 4;
    pads[0][5] = 5;
    pads[0][6] = 7;
    pads[1][0] = 8;
    pads[1][1] = 9;
    pads[1][2] = 10;
    pads[2][0] = 11;
    pads[2][1] = 12;
    pads[2][2] = 13;
    pads[3][0] = 14;
    pads[3][1] = 15;
    pads[3][2] = 16;
    pads[3][3] = 17;
    pads[3][4] = 6;
    pads[3][5] = 18;
    pads[4][0] = 0;
    pads[4][1] = 1;
    pads[4][2] = 2;
    pads[4][3] = 3;
    pads[4][4] = 4;
    pads[4][5] = 5;
    pads[4][6] = 7;
    pads[5][0] = 8;
    pads[5][1] = 9;
    pads[5][2] = 10;
    pads[6][0] = 11;
    pads[6][1] = 12;
    pads[6][2] = 13;
    pads[7][0] = 14;
    pads[7][1] = 15;
    pads[7][2] = 16;
    pads[7][3] = 17;
    pads[7][4] = 6;
    pads[7][5] = 18;

    orchestrationTransformer.orchestrationModel = orchestrationModel.get();
}

AppSession::~AppSession() {
    testOrchestrationTransformerThread.stop();
    musicTransformer.stopThread(-1);
    orchestrationTransformer.stopThread(-1);
    outputPlayback.stopThread(-1);
}

auto AppSession::rebuildInputFilter() -> void {
    inputFilter = createInputFilter(modelConfig.inputFilterType,
                                    modelConfig,
                                    musicTransformer.inputTokenQueue,
                                    musicTransformer.inputConditioningQueue,
                                    musicTransformer.updatesFromFilter);
}

auto AppSession::setOrchestrationModel(const juce::String &name) -> bool {
    auto model = createOrchestrationModel(name.toStdString());
    if (model == nullptr)
        return false;

    testOrchestrationTransformerThread.stop();
    orchestrationModel = std::move(model);
    orchestrationTransformer.orchestrationModel = orchestrationModel.get();
    return true;
}

auto AppSession::startGeneration() -> void {
    if (! musicTransformer.isModelLoaded()) {
        juce::AlertWindow::showMessageBoxAsync(
            juce::AlertWindow::WarningIcon,
            UiConstants::workspaceGenerationNoModelTitle,
            UiConstants::workspaceGenerationNoModelMessage);
        return;
    }

    if (modelConfig.inputMode == InputMode::Direct && modelConfig.directInputStartOnInput) {
        musicTransformer.directInputBlock = true;
    } else {
        musicTransformer.directInputBlock = false;
    }

    midiInputProcess.resetForStart();
    if (inputFilter != nullptr)
        inputFilter->reset();

    musicTransformer.startThread();
    orchestrationTransformer.startThread();
    outputPlayback.startThread();
    if (! mtcClockActive) {
        clock.startAtTime(0);
    } else {
        clock.setMtcTime(0);
    }
    if (orchestrationModel != nullptr && orchestrationModel->getName() == "TestModel")
        testOrchestrationTransformerThread.start(clock, orchestrationTransformer);
}

auto AppSession::stopGeneration() -> void {
    testOrchestrationTransformerThread.stop();
    musicTransformer.signalThreadShouldExit();
    orchestrationTransformer.signalThreadShouldExit();
    outputPlayback.signalThreadShouldExit();
    // Join before callers replace the ORT session (e.g. updateModel).
    musicTransformer.stopThread(-1);
    orchestrationTransformer.stopThread(-1);
    outputPlayback.stopThread(-1);
    clock.stop();
    outputPlayback.resetProgress();
    if (outputProcessor != nullptr)
        outputProcessor->clear();
}
