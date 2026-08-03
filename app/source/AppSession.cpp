#include "VirtualOrch/AppSession.h"

#include "VirtualOrch/ui/UiConstants.h"

AppSession::AppSession()
    : presetStore(modelConfig),
      clock(metrics),
      musicTransformer(modelConfig),
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
}

AppSession::~AppSession() {
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
}

auto AppSession::stopGeneration() -> void {
    musicTransformer.signalThreadShouldExit();
    orchestrationTransformer.signalThreadShouldExit();
    outputPlayback.signalThreadShouldExit();
    clock.stop();
    outputPlayback.resetProgress();
    if (outputProcessor != nullptr)
        outputProcessor->clear();
}
