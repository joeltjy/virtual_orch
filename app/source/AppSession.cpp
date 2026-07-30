#include "VirtualOrch/AppSession.h"

#include "VirtualOrch/ui/UiConstants.h"

AppSession::AppSession()
    : presetStore(modelConfig),
      clock(metrics),
      musicTransformer(modelConfig),
      inputFilter(createInputFilter(modelConfig.inputFilterType,
                                    modelConfig,
                                    musicTransformer.inputTokenQueue,
                                    musicTransformer.inputConditioningQueue)),
      outputPlayback(clock, musicTransformer, outputProcessor, bufferOutputProcessor,
                     visualizationBufferSize),
      midiInputProcess(clock, musicTransformer, modelConfig, outputProcessor, inputFilter,
                       selectedMidiInputIdentifier, selectedMidiInput2Identifier,
                       selectedMtcClockIdentifier, mtcClockActive) {
}

AppSession::~AppSession() {
    musicTransformer.stopThread(-1);
    outputPlayback.stopThread(-1);
}

auto AppSession::rebuildInputFilter() -> void {
    inputFilter = createInputFilter(modelConfig.inputFilterType,
                                    modelConfig,
                                    musicTransformer.inputTokenQueue,
                                    musicTransformer.inputConditioningQueue);
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
    outputPlayback.startThread();
    if (! mtcClockActive) {
        clock.startAtTime(0);
    } else {
        clock.setMtcTime(0);
    }
}

auto AppSession::stopGeneration() -> void {
    musicTransformer.signalThreadShouldExit();
    outputPlayback.signalThreadShouldExit();
    clock.stop();
    outputPlayback.resetProgress();
    if (outputProcessor != nullptr)
        outputProcessor->clear();
}
