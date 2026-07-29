#include "VirtualOrch/AppSession.h"

#include "VirtualOrch/ui/UiConstants.h"

AppSession::AppSession()
    : presetStore(modelConfig),
      clock(metrics),
      musicTransformer(modelConfig),
      outputPlayback(clock, musicTransformer, outputProcessor, bufferOutputProcessor,
                     visualizationBufferSize),
      midiInputProcess(clock, musicTransformer, modelConfig, outputProcessor,
                       selectedMidiInputIdentifier, selectedMidiInput2Identifier,
                       selectedMtcClockIdentifier, mtcClockActive) {
}

AppSession::~AppSession() {
    musicTransformer.stopThread(-1);
    outputPlayback.stopThread(-1);
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
