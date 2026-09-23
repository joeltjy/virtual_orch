#include "VirtualOrch/AppSession.h"

#include "VirtualOrch/orchestration-models/IodPretrained.h"
#include "VirtualOrch/orchestration-models/OrchestrationModels.h"
#include "VirtualOrch/ProjectPaths.h"
#include "VirtualOrch/ui/UiConstants.h"
#include "onnxruntime_cxx_api.h"

#include <atomic>
#include <iostream>
#include <thread>

namespace {

/**
 * Joins happen off the message thread when stopping from the UI, so a stuck worker
 * cannot freeze Stop. Destructor still joins with this timeout.
 */
constexpr int threadStopTimeoutMs = 5000;

std::atomic<bool> stopJoinInFlight{false};

auto reportThreadProblem(ModelSamplingAlert &alert, const juce::String &detail) -> void {
    alert.report("Threads", detail);
    // Also to the console: the alert line is easy to miss and gets overwritten.
    std::cerr << "[threads] " << detail << std::endl;
}

auto stopThreadOrReport(juce::Thread &thread, ModelSamplingAlert &alert) -> void {
    if (thread.stopThread(threadStopTimeoutMs))
        return;
    reportThreadProblem(alert,
                        thread.getThreadName() + " did not exit within "
                            + juce::String(threadStopTimeoutMs) + " ms");
}

/** startThread() is a no-op on a running thread, which silently yields a dead session. */
auto startThreadOrReport(juce::Thread &thread, ModelSamplingAlert &alert) -> void {
    if (thread.isThreadRunning()) {
        reportThreadProblem(alert,
                            thread.getThreadName()
                                + " was still running at start; generation will not restart");
        return;
    }
    thread.startThread();
}

} // namespace

AppSession::AppSession()
    : presetStore(modelConfig),
      clock(metrics),
      musicTransformer(modelConfig),
      reductionTransformerV1(modelConfig),
      reductionTransformerV2(modelConfig),
      orchestrationModel(createOrchestrationModel("IodPretrained")),
      inputFilter(createInputFilter(modelConfig.inputFilterType,
                                    modelConfig,
                                    musicTransformer.inputTokenQueue,
                                    musicTransformer.inputConditioningQueue,
                                    musicTransformer.updatesFromFilter)),
      outputPlayback(clock, orchestrationTransformer, outputProcessor, bufferOutputProcessor,
                     visualizationBufferSize, playbackSource),
      midiInputProcess(clock,
                       [this]() -> ReductionTransformer & { return activeReduction(); },
                       modelConfig,
                       outputProcessor,
                       inputFilter,
                       selectedMidiInputIdentifier,
                       selectedLaunchpadMidiIdentifier,
                       selectedMtcClockIdentifier,
                       mtcClockActive) {
    if (const auto vocsep = findModelCheckpoint("vocsep_reduction_best")) {
        try {
            voiceSeparation.init(vocsep->onnxFile.getFullPathName().toRawUTF8());
            std::cerr << "[vocsep] loaded " << vocsep->onnxFile.getFullPathName() << std::endl;
        } catch (const Ort::Exception &e) {
            std::cerr << "[vocsep] failed to load: " << e.what() << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "[vocsep] failed to load: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "[vocsep] checkpoint not found (need app/vocsep_reduction_best.json + "
                     "Documents/onnx_export_*/vocsep.onnx)"
                  << std::endl;
    }

    bindActiveMusicBackend();
    midiInputProcess.allowInputThru = [this] {
        return orchestrationTransformer.getMode() == OrchestrationMode::Jam;
    };
    midiInputProcess.getLaunchpadGrid().instrumentUpdates =
        &orchestrationTransformer.instrumentUpdates;
    midiInputProcess.getLaunchpadGrid().onInstrumentUpdate =
        [this](const InstrumentUpdate &update) {
            orchestrationTransformer.applyOneInstrumentUpdate(update);
        };
    orchestrationTransformer.rebuildInstrumentsFromPads = [this] {
        std::set<int32_t> user;
        std::set<int32_t> model;
        auto &grid = midiInputProcess.getLaunchpadGrid();
        for (int row = 0; row < LaunchpadGrid::kRows; ++row) {
            for (int col = 0; col < LaunchpadGrid::kCols; ++col) {
                if (grid.getEffectiveState(row, col) == 0)
                    continue;
                const auto &mapped =
                    grid.padInstruments[static_cast<size_t>(row)][static_cast<size_t>(col)];
                if (! mapped.has_value())
                    continue;
                if (row < 4)
                    user.insert(*mapped);
                else
                    model.insert(*mapped);
            }
        }
        orchestrationTransformer.replaceInstrumentSets(std::move(user), std::move(model));
    };
    midiInputProcess.getLaunchpadGrid().scene_callback = [this](int idx) {
        if (idx == 0) {
            const auto mode = orchestrationTransformer.getMode();
            orchestrationTransformer.setMode(mode == OrchestrationMode::Edit
                                                 ? OrchestrationMode::Jam
                                                 : OrchestrationMode::Edit);
        } else if (idx == 1) {
            auto &reduction = activeReduction();
            reduction.setGenerationPause(! reduction.generationPause.get());
        } else if (idx == 2) {
            orchestrationTransformer.paused.set(! orchestrationTransformer.paused.get());
        }

        syncPauseTopLeds();
    };
    // One family per row (taxonomy order): strings 0–6, woodwind 7–10, brass 11–14,
    // other 15–19. Rows 0–3 are User, rows 4–7 mirror them for Model.
    auto &pads = midiInputProcess.getLaunchpadGrid().padInstruments;
    for (int block = 0; block < 2; ++block) {
        const int base = block * 4;
        for (int col = 0; col < 7; ++col)
            pads[base + 0][col] = col; // strings 0–6
        for (int col = 0; col < 4; ++col)
            pads[base + 1][col] = 7 + col; // woodwind 7–10
        for (int col = 0; col < 4; ++col)
            pads[base + 2][col] = 11 + col; // brass 11–14
        for (int col = 0; col < 5; ++col)
            pads[base + 3][col] = 15 + col; // other 15–19
    }

    // Default: all pads off; user/model instrument sets start empty.
    orchestrationTransformer.userInstruments.clear();
    orchestrationTransformer.modelInstruments.clear();

    orchestrationTransformer.orchestrationModel = orchestrationModel.get();
    outputPlayback.isReductionPaused = [this] {
        return activeReduction().isGenerationStopped();
    };
    outputPlayback.getReductionOutputHistory = [this] {
        return activeReduction().getOutputHistory();
    };

    musicTransformer.clock = &clock;
    reductionTransformerV1.clock = &clock;
    reductionTransformerV2.clock = &clock;
    musicTransformer.samplingAlert = &modelSamplingAlert;
    reductionTransformerV1.samplingAlert = &modelSamplingAlert;
    reductionTransformerV2.samplingAlert = &modelSamplingAlert;
    if (orchestrationModel != nullptr) {
        orchestrationModel->samplingAlert = &modelSamplingAlert;
        orchestrationModel->getOutputTimings = &orchestrationTransformer.getOutputAccum;
        if (auto *iod = dynamic_cast<IodPretrained *>(orchestrationModel.get())) {
            auto orch = presetStore.settings.getOrCreateChildWithName("orchestration", nullptr);
            iod->setEnsembleEnabled(
                orch.hasProperty("vocsepEnsemble")
                    ? static_cast<bool>(orch.getProperty("vocsepEnsemble"))
                    : false);
            iod->setEnsembleAddP(
                orch.hasProperty("vocsepEnsembleAddP")
                    ? static_cast<float>(
                          static_cast<double>(orch.getProperty("vocsepEnsembleAddP")))
                    : IodPretrainedTypes::ensembleAddPDefault);
            iod->setEnsembleRemoveP(
                orch.hasProperty("vocsepEnsembleRemoveP")
                    ? static_cast<float>(
                          static_cast<double>(orch.getProperty("vocsepEnsembleRemoveP")))
                    : IodPretrainedTypes::ensembleRemovePDefault);
        }
    }

    const auto syncLeds = [this] { syncPauseTopLeds(); };
    musicTransformer.onPausedChanged = syncLeds;
    reductionTransformerV1.onPausedChanged = syncLeds;
    reductionTransformerV2.onPausedChanged = syncLeds;
}

AppSession::~AppSession() {
    testOrchestrationTransformerThread.stop();
    // Ensure any detached stop join finished before members are destroyed.
    while (stopJoinInFlight.load(std::memory_order_acquire))
        juce::Thread::sleep(10);
    musicTransformer.stopThread(threadStopTimeoutMs);
    reductionTransformerV1.stopThread(threadStopTimeoutMs);
    reductionTransformerV2.stopThread(threadStopTimeoutMs);
    orchestrationTransformer.stopThread(threadStopTimeoutMs);
    outputPlayback.stopThread(threadStopTimeoutMs);
    if (outputProcessor != nullptr)
        outputProcessor->clear();
}

auto AppSession::activeReduction() -> ReductionTransformer & {
    switch (musicModelArch) {
        case MusicModelArch::DenseV1:
            return reductionTransformerV1;
        case MusicModelArch::DenseV2:
            return reductionTransformerV2;
        case MusicModelArch::Amt:
            return musicTransformer;
    }
    return musicTransformer;
}

auto AppSession::activeReduction() const -> const ReductionTransformer & {
    switch (musicModelArch) {
        case MusicModelArch::DenseV1:
            return reductionTransformerV1;
        case MusicModelArch::DenseV2:
            return reductionTransformerV2;
        case MusicModelArch::Amt:
            return musicTransformer;
    }
    return musicTransformer;
}

auto AppSession::rebuildInputFilter() -> void {
    auto &reduction = activeReduction();
    inputFilter = createInputFilter(modelConfig.inputFilterType,
                                    modelConfig,
                                    reduction.inputTokenQueue,
                                    reduction.inputConditioningQueue,
                                    reduction.updatesFromFilter);
    inputFilter->orchestrationMidiOutgoing = &orchestrationTransformer.midiInputIncoming;
    inputFilter->orchestrationConditioningOutgoing =
        &orchestrationTransformer.conditioningIncoming;
}

auto AppSession::bindActiveMusicBackend() -> void {
    musicTransformer.orchestrationReductionIncoming = nullptr;
    musicTransformer.orchestrationUpdatesIncoming = nullptr;
    reductionTransformerV1.orchestrationReductionIncoming = nullptr;
    reductionTransformerV1.orchestrationUpdatesIncoming = nullptr;
    reductionTransformerV2.orchestrationReductionIncoming = nullptr;
    reductionTransformerV2.orchestrationUpdatesIncoming = nullptr;

    auto &reduction = activeReduction();
    reduction.orchestrationReductionIncoming = &orchestrationTransformer.reductionIncoming;
    reduction.orchestrationUpdatesIncoming = &orchestrationTransformer.updatesIncoming;

    musicTransformer.voiceSeparation = nullptr;
    reductionTransformerV1.voiceSeparation = nullptr;
    reductionTransformerV2.voiceSeparation = nullptr;
    reduction.voiceSeparation = voiceSeparation.isLoaded() ? &voiceSeparation : nullptr;

    outputPlayback.reductionOutputQueue = &reduction.outputTokenQueue;

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
    reductionTransformerV1.onInputDataChanged = nullptr;
    reductionTransformerV2.onInputDataChanged = nullptr;
    activeReduction().onInputDataChanged = std::move(callback);
}

auto AppSession::setOrchestrationModel(const juce::String &name) -> bool {
    auto model = createOrchestrationModel(name.toStdString());
    if (model == nullptr)
        return false;

    testOrchestrationTransformerThread.stop();
    orchestrationModel = std::move(model);
    orchestrationModel->samplingAlert = &modelSamplingAlert;
    orchestrationModel->getOutputTimings = &orchestrationTransformer.getOutputAccum;
    orchestrationTransformer.orchestrationModel = orchestrationModel.get();

    if (auto *iod = dynamic_cast<IodPretrained *>(orchestrationModel.get())) {
        auto orch = presetStore.settings.getOrCreateChildWithName("orchestration", nullptr);
        const bool enabled = orch.hasProperty("vocsepEnsemble")
                                 ? static_cast<bool>(orch.getProperty("vocsepEnsemble"))
                                 : false;
        const float addP =
            orch.hasProperty("vocsepEnsembleAddP")
                ? static_cast<float>(static_cast<double>(orch.getProperty("vocsepEnsembleAddP")))
                : IodPretrainedTypes::ensembleAddPDefault;
        const float removeP =
            orch.hasProperty("vocsepEnsembleRemoveP")
                ? static_cast<float>(static_cast<double>(orch.getProperty("vocsepEnsembleRemoveP")))
                : IodPretrainedTypes::ensembleRemovePDefault;
        iod->setEnsembleEnabled(enabled);
        iod->setEnsembleAddP(addP);
        iod->setEnsembleRemoveP(removeP);
    }
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

    if (stopJoinInFlight.load(std::memory_order_acquire)) {
        reportThreadProblem(modelSamplingAlert,
                            "Previous stop is still joining workers; wait a moment and Start again");
        return;
    }

    if (modelConfig.inputMode == InputMode::Direct && modelConfig.directInputStartOnInput)
        getMusicDirectInputBlock() = true;
    else
        getMusicDirectInputBlock() = false;

    midiInputProcess.resetForStart();
    if (inputFilter != nullptr)
        inputFilter->reset();

    startThreadOrReport(activeReduction(), modelSamplingAlert);
    startThreadOrReport(orchestrationTransformer, modelSamplingAlert);
    startThreadOrReport(outputPlayback, modelSamplingAlert);
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
    reductionTransformerV1.signalThreadShouldExit();
    reductionTransformerV2.signalThreadShouldExit();
    orchestrationTransformer.signalThreadShouldExit();
    outputPlayback.signalThreadShouldExit();

    // Unblock the UI immediately: clock/audio clear must not wait on ORT joins.
    musicTransformer.resetAheadThrottle();
    reductionTransformerV1.resetAheadThrottle();
    reductionTransformerV2.resetAheadThrottle();
    clock.stop();
    outputPlayback.resetProgress();
    outputPlayback.clearNoteOnHistory();
    musicTransformer.clearOutputHistory();
    reductionTransformerV1.clearOutputHistory();
    reductionTransformerV2.clearOutputHistory();
    if (outputProcessor != nullptr)
        outputProcessor->clear();

    if (stopJoinInFlight.exchange(true, std::memory_order_acq_rel))
        return;

    // Join workers off the message thread so Stop stays responsive while Dense/IC finish ORT.
    std::thread([this] {
        stopThreadOrReport(musicTransformer, modelSamplingAlert);
        stopThreadOrReport(reductionTransformerV1, modelSamplingAlert);
        stopThreadOrReport(reductionTransformerV2, modelSamplingAlert);
        stopThreadOrReport(orchestrationTransformer, modelSamplingAlert);
        stopThreadOrReport(outputPlayback, modelSamplingAlert);
        stopJoinInFlight.store(false, std::memory_order_release);
    }).detach();
}

auto AppSession::syncPauseTopLeds() -> void {
    auto &grid = midiInputProcess.getLaunchpadGrid();
    grid.setTopLed(1, activeReduction().generationPause.get() ? LaunchpadLighting::kRed
                                                              : LaunchpadLighting::kOff);
    grid.setTopLed(2, orchestrationTransformer.paused.get() ? LaunchpadLighting::kRed
                                                            : LaunchpadLighting::kOff);
}
