#include "VirtualOrch/AppSession.h"

#include "VirtualOrch/orchestration-models/OrchestrationModels.h"
#include "VirtualOrch/ui/UiConstants.h"

AppSession::AppSession()
    : presetStore(modelConfig),
      clock(metrics),
      musicTransformer(modelConfig),
      denseMusicTransformer(modelConfig),
      orchestrationModel(createOrchestrationModel("TestModel")),
      inputFilter(createInputFilter(modelConfig.inputFilterType,
                                    modelConfig,
                                    musicTransformer.inputTokenQueue,
                                    musicTransformer.inputConditioningQueue,
                                    musicTransformer.updatesFromFilter)),
      outputPlayback(clock, orchestrationTransformer, outputProcessor, bufferOutputProcessor,
                     visualizationBufferSize),
      midiInputProcess(clock,
                       [this]() -> ReductionTransformer & { return activeReduction(); },
                       modelConfig,
                       outputProcessor,
                       inputFilter,
                       selectedMidiInputIdentifier,
                       selectedLaunchpadMidiIdentifier,
                       selectedMtcClockIdentifier,
                       mtcClockActive) {
    bindActiveMusicBackend();
    midiInputProcess.getLaunchpadGrid().instrumentUpdates =
        &orchestrationTransformer.instrumentUpdates;
    midiInputProcess.getLaunchpadGrid().scene_callback = [this](int idx) {
        if (idx == 0) {
            const auto mode = orchestrationTransformer.getMode();
            orchestrationTransformer.setMode(mode == OrchestrationMode::Edit
                                                 ? OrchestrationMode::Jam
                                                 : OrchestrationMode::Edit);
        } else if (idx == 1) {
            auto &reduction = activeReduction();
            reduction.paused.set(! reduction.paused.get());
        } else if (idx == 2) {
            orchestrationTransformer.paused.set(! orchestrationTransformer.paused.get());
        }

        syncPauseTopLeds();
    };
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
    outputPlayback.isReductionPaused = [this] { return activeReduction().paused.get(); };

    musicTransformer.clock = &clock;
    denseMusicTransformer.clock = &clock;

    const auto syncLeds = [this] { syncPauseTopLeds(); };
    musicTransformer.onPausedChanged = syncLeds;
    denseMusicTransformer.onPausedChanged = syncLeds;
}

AppSession::~AppSession() {
    testOrchestrationTransformerThread.stop();
    musicTransformer.stopThread(-1);
    denseMusicTransformer.stopThread(-1);
    orchestrationTransformer.stopThread(-1);
    outputPlayback.stopThread(-1);
    if (outputProcessor != nullptr)
        outputProcessor->clear();
}

auto AppSession::activeReduction() -> ReductionTransformer & {
    return musicModelArch == MusicModelArch::Dense
               ? static_cast<ReductionTransformer &>(denseMusicTransformer)
               : static_cast<ReductionTransformer &>(musicTransformer);
}

auto AppSession::activeReduction() const -> const ReductionTransformer & {
    return musicModelArch == MusicModelArch::Dense
               ? static_cast<const ReductionTransformer &>(denseMusicTransformer)
               : static_cast<const ReductionTransformer &>(musicTransformer);
}

auto AppSession::rebuildInputFilter() -> void {
    auto &reduction = activeReduction();
    inputFilter = createInputFilter(modelConfig.inputFilterType,
                                    modelConfig,
                                    reduction.inputTokenQueue,
                                    reduction.inputConditioningQueue,
                                    reduction.updatesFromFilter);
}

auto AppSession::bindActiveMusicBackend() -> void {
    musicTransformer.orchestrationMidiIncoming = nullptr;
    musicTransformer.orchestrationConditioningIncoming = nullptr;
    musicTransformer.orchestrationReductionIncoming = nullptr;
    musicTransformer.orchestrationUpdatesIncoming = nullptr;
    denseMusicTransformer.orchestrationMidiIncoming = nullptr;
    denseMusicTransformer.orchestrationConditioningIncoming = nullptr;
    denseMusicTransformer.orchestrationReductionIncoming = nullptr;
    denseMusicTransformer.orchestrationUpdatesIncoming = nullptr;

    auto &reduction = activeReduction();
    reduction.orchestrationMidiIncoming = &orchestrationTransformer.midiInputIncoming;
    reduction.orchestrationConditioningIncoming = &orchestrationTransformer.conditioningIncoming;
    reduction.orchestrationReductionIncoming = &orchestrationTransformer.reductionIncoming;
    reduction.orchestrationUpdatesIncoming = &orchestrationTransformer.updatesIncoming;

    rebuildInputFilter();
}

auto AppSession::setMusicModelArch(MusicModelArch arch) -> void {
    auto callback = activeReduction().onInputDataChanged;

    musicModelArch = arch;
    bindActiveMusicBackend();
    setOnInputDataChanged(std::move(callback));
}

auto AppSession::isMusicModelLoaded() const -> bool {
    return activeReduction().isModelLoaded();
}

auto AppSession::isMusicThreadRunning() const -> bool {
    return activeReduction().isThreadRunning();
}

auto AppSession::getMusicDirectInputBlock() -> juce::Atomic<bool> & {
    return activeReduction().directInputBlock;
}

auto AppSession::getActiveInputData() const -> std::vector<int32_t> {
    return activeReduction().getInputData();
}

auto AppSession::setOnInputDataChanged(std::function<void(std::vector<int32_t>)> callback) -> void {
    musicTransformer.onInputDataChanged = nullptr;
    denseMusicTransformer.onInputDataChanged = nullptr;
    activeReduction().onInputDataChanged = std::move(callback);
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
    if (! isMusicModelLoaded()) {
        juce::AlertWindow::showMessageBoxAsync(
            juce::AlertWindow::WarningIcon,
            UiConstants::workspaceGenerationNoModelTitle,
            UiConstants::workspaceGenerationNoModelMessage);
        return;
    }

    if (modelConfig.inputMode == InputMode::Direct && modelConfig.directInputStartOnInput)
        getMusicDirectInputBlock() = true;
    else
        getMusicDirectInputBlock() = false;

    midiInputProcess.resetForStart();
    if (inputFilter != nullptr)
        inputFilter->reset();

    activeReduction().startThread();

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
    denseMusicTransformer.signalThreadShouldExit();
    orchestrationTransformer.signalThreadShouldExit();
    outputPlayback.signalThreadShouldExit();
    musicTransformer.stopThread(-1);
    denseMusicTransformer.stopThread(-1);
    orchestrationTransformer.stopThread(-1);
    outputPlayback.stopThread(-1);
    clock.stop();
    outputPlayback.resetProgress();
    outputPlayback.clearNoteOnHistory();
    musicTransformer.clearOutputHistory();
    denseMusicTransformer.clearOutputHistory();
    if (outputProcessor != nullptr)
        outputProcessor->clear();
}

auto AppSession::syncPauseTopLeds() -> void {
    auto &grid = midiInputProcess.getLaunchpadGrid();
    grid.setTopLed(1, activeReduction().paused.get() ? LaunchpadLighting::kRed
                                                     : LaunchpadLighting::kOff);
    grid.setTopLed(2, orchestrationTransformer.paused.get() ? LaunchpadLighting::kRed
                                                            : LaunchpadLighting::kOff);
}
