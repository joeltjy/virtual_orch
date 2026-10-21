#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/widgets/ActiveInstrumentsView.h"

#include "VirtualOrch/orchestration-models/IodPretrained.h"
#include "VirtualOrch/orchestration-models/OrchestrationModels.h"
#include "VirtualOrch/ProjectPaths.h"
#include "VirtualOrch/ui/UiConstants.h"
#include "onnxruntime_cxx_api.h"

#include <array>
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
      reductionTransformerPianoReduction(modelConfig),
      reductionTransformerAnticipated(modelConfig),
      reductionTransformerMelodyFirst(modelConfig),
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
                       selectedPedalMidiInputIdentifier,
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
    outputPlayback.activityLog = &activityLog;
    const auto pianoSilent = [this] {
        return activityLog.pianoSilentForMs(juce::Time::currentTimeMillis());
    };
    musicTransformer.pianoSilentForMs = pianoSilent;
    reductionTransformerV1.pianoSilentForMs = pianoSilent;
    reductionTransformerV2.pianoSilentForMs = pianoSilent;
    reductionTransformerPianoReduction.pianoSilentForMs = pianoSilent;
    reductionTransformerAnticipated.pianoSilentForMs = pianoSilent;
    reductionTransformerMelodyFirst.pianoSilentForMs = pianoSilent;
    midiInputProcess.activityLog = &activityLog;
    midiInputProcess.getLaunchpadGrid().action_callback =
        [this](const PerformanceActivity::LaunchpadAction &action) {
            consoleLog.add(PerformanceActivity::describe(action, [](int32_t local) {
                return ActiveInstrumentsView::shortNameForLocal(local);
            }));
        };
    midiInputProcess.onPedalDown = [this](int32_t pedalTimeCs) {
        outputPlayback.pedalClearRequests.push(pedalTimeCs);
    };
    midiInputProcess.allowInputThru = [this] {
        return isJamLike(orchestrationTransformer.getMode());
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
                const auto target = LaunchpadGrid::targetForRow(row);
                if (! mapped.has_value() || ! target.has_value())
                    continue;
                if (*target == InstrumentUpdateTarget::User)
                    user.insert(*mapped);
                else
                    model.insert(*mapped);
            }
        }
        orchestrationTransformer.replaceInstrumentSets(std::move(user), std::move(model));
    };
    midiInputProcess.getLaunchpadGrid().mode_callback = [this](OrchestrationMode mode) {
        setOrchestrationMode(mode);
        consoleLog.add(PerformanceActivity::describe(
            {PerformanceActivity::LaunchpadAction::Kind::Mode, 0, static_cast<int32_t>(mode),
             false, false},
            [](int32_t) { return juce::String(); }));
    };
    midiInputProcess.getLaunchpadGrid().setSelectedMode(orchestrationTransformer.getMode());
    midiInputProcess.getLaunchpadGrid().accompaniment_callback = [this] {
        applyAccompanimentCap();
    };
    midiInputProcess.getLaunchpadGrid().scene_callback = [this](int idx) {
        const auto logPause = [this](int32_t which, bool paused) {
            consoleLog.add(PerformanceActivity::describe(
                {PerformanceActivity::LaunchpadAction::Kind::Pause, 0, which, paused, false},
                [](int32_t) { return juce::String(); }));
        };
        if (idx == 0) {
            consoleLog.add(PerformanceActivity::describe(
                {PerformanceActivity::LaunchpadAction::Kind::ForceRestart, 0, 0, false, false},
                [](int32_t) { return juce::String(); }));
            forceRestart();
            return;
        }
        if (idx == 1) {
            auto &reduction = activeReduction();
            reduction.setGenerationPause(! reduction.generationPause.get());
            logPause(1, reduction.generationPause.get());
        } else if (idx == 2) {
            orchestrationTransformer.paused.set(! orchestrationTransformer.paused.get());
            logPause(2, orchestrationTransformer.paused.get());
        }

        syncPauseTopLeds();
    };
    // Same two-row instrument layout for User (input) and Model; row offset within the set.
    struct PadMapping {
        int rowOffset;
        int col;
        int32_t localId;
    };
    static constexpr std::array<PadMapping, 13> kInstrumentPads{{
        {0, 0, 6},  // string ensemble
        {0, 1, 5},  // pizzicato
        {0, 2, 4},  // tremolo
        {0, 4, 7},  // flute
        {0, 5, 8},  // oboe
        {0, 6, 9},  // clarinet
        {0, 7, 10}, // bassoon
        {1, 0, 12}, // horn
        {1, 1, 11}, // trumpet
        {1, 2, 13}, // trombone
        {1, 3, 14}, // tuba
        {1, 4, 16}, // timpani
        {1, 5, 17}, // snare
    }};

    // All instrument pads start off (User + Model).
    orchestrationTransformer.userInstruments.clear();
    orchestrationTransformer.modelInstruments.clear();
    {
        auto &grid = midiInputProcess.getLaunchpadGrid();
        for (const int base: {LaunchpadGrid::kUserRowFirst, LaunchpadGrid::kModelRowFirst})
            for (const auto &pad: kInstrumentPads)
                grid.padInstruments[static_cast<size_t>(base + pad.rowOffset)]
                                   [static_cast<size_t>(pad.col)] = pad.localId;
    }

    orchestrationTransformer.orchestrationModel = orchestrationModel.get();
    orchestrationTransformer.clock = &clock;
    outputPlayback.isReductionPaused = [this] {
        return activeReduction().isGenerationStopped();
    };
    outputPlayback.getReductionOutputHistory = [this] {
        return activeReduction().getOutputHistory();
    };

    musicTransformer.clock = &clock;
    reductionTransformerV1.clock = &clock;
    reductionTransformerV2.clock = &clock;
    reductionTransformerPianoReduction.clock = &clock;
    reductionTransformerAnticipated.clock = &clock;
    reductionTransformerMelodyFirst.clock = &clock;
    musicTransformer.samplingAlert = &modelSamplingAlert;
    reductionTransformerV1.samplingAlert = &modelSamplingAlert;
    reductionTransformerV2.samplingAlert = &modelSamplingAlert;
    reductionTransformerPianoReduction.samplingAlert = &modelSamplingAlert;
    reductionTransformerAnticipated.samplingAlert = &modelSamplingAlert;
    reductionTransformerMelodyFirst.samplingAlert = &modelSamplingAlert;
    if (orchestrationModel != nullptr) {
        orchestrationModel->samplingAlert = &modelSamplingAlert;
        orchestrationModel->getOutputTimings = &orchestrationTransformer.getOutputAccum;
        applyVocsepEnsembleSettings();
    }

    const auto syncLeds = [this] { syncPauseTopLeds(); };
    musicTransformer.onPausedChanged = syncLeds;
    reductionTransformerV1.onPausedChanged = syncLeds;
    reductionTransformerV2.onPausedChanged = syncLeds;
    reductionTransformerPianoReduction.onPausedChanged = syncLeds;
    reductionTransformerAnticipated.onPausedChanged = syncLeds;
    reductionTransformerMelodyFirst.onPausedChanged = syncLeds;
}

AppSession::~AppSession() {
    alive->store(false);
    testOrchestrationTransformerThread.stop();
    // Ensure any detached stop join finished before members are destroyed.
    while (stopJoinInFlight.load(std::memory_order_acquire))
        juce::Thread::sleep(10);
    musicTransformer.stopThread(threadStopTimeoutMs);
    reductionTransformerV1.stopThread(threadStopTimeoutMs);
    reductionTransformerV2.stopThread(threadStopTimeoutMs);
    reductionTransformerPianoReduction.stopThread(threadStopTimeoutMs);
    reductionTransformerAnticipated.stopThread(threadStopTimeoutMs);
    reductionTransformerMelodyFirst.stopThread(threadStopTimeoutMs);
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
        case MusicModelArch::DensePianoReduction:
            return reductionTransformerPianoReduction;
        case MusicModelArch::DenseAnticipatedPiano:
            return reductionTransformerAnticipated;
        case MusicModelArch::DenseMelodyFirst:
            return reductionTransformerMelodyFirst;
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
        case MusicModelArch::DensePianoReduction:
            return reductionTransformerPianoReduction;
        case MusicModelArch::DenseAnticipatedPiano:
            return reductionTransformerAnticipated;
        case MusicModelArch::DenseMelodyFirst:
            return reductionTransformerMelodyFirst;
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
    reductionTransformerPianoReduction.orchestrationReductionIncoming = nullptr;
    reductionTransformerAnticipated.orchestrationReductionIncoming = nullptr;
    reductionTransformerMelodyFirst.orchestrationReductionIncoming = nullptr;
    reductionTransformerPianoReduction.orchestrationUpdatesIncoming = nullptr;
    reductionTransformerAnticipated.orchestrationUpdatesIncoming = nullptr;
    reductionTransformerMelodyFirst.orchestrationUpdatesIncoming = nullptr;
    musicTransformer.orchestrationRestampsIncoming = nullptr;
    reductionTransformerV1.orchestrationRestampsIncoming = nullptr;
    reductionTransformerV2.orchestrationRestampsIncoming = nullptr;
    reductionTransformerPianoReduction.orchestrationRestampsIncoming = nullptr;
    reductionTransformerAnticipated.orchestrationRestampsIncoming = nullptr;
    reductionTransformerMelodyFirst.orchestrationRestampsIncoming = nullptr;

    auto &reduction = activeReduction();
    reduction.orchestrationReductionIncoming = &orchestrationTransformer.reductionIncoming;
    reduction.orchestrationUpdatesIncoming = &orchestrationTransformer.updatesIncoming;
    reduction.orchestrationRestampsIncoming = &orchestrationTransformer.restampsIncoming;

    musicTransformer.voiceSeparation = nullptr;
    reductionTransformerV1.voiceSeparation = nullptr;
    reductionTransformerV2.voiceSeparation = nullptr;
    reductionTransformerPianoReduction.voiceSeparation = nullptr;
    reductionTransformerAnticipated.voiceSeparation = nullptr;
    reductionTransformerMelodyFirst.voiceSeparation = nullptr;
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
    reductionTransformerPianoReduction.onInputDataChanged = nullptr;
    reductionTransformerAnticipated.onInputDataChanged = nullptr;
    reductionTransformerMelodyFirst.onInputDataChanged = nullptr;
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

    applyVocsepEnsembleSettings();
    return true;
}

auto AppSession::getVocsepEnsembleSettings() -> VocsepEnsembleSettings {
    auto orch = presetStore.settings.getOrCreateChildWithName("orchestration", nullptr);
    VocsepEnsembleSettings out;
    if (orch.hasProperty("vocsepEnsemble"))
        out.enabled = static_cast<bool>(orch.getProperty("vocsepEnsemble"));
    if (orch.hasProperty("vocsepEnsembleAddP"))
        out.addP = static_cast<float>(static_cast<double>(orch.getProperty("vocsepEnsembleAddP")));
    if (orch.hasProperty("vocsepEnsembleRemoveP"))
        out.removeP =
            static_cast<float>(static_cast<double>(orch.getProperty("vocsepEnsembleRemoveP")));
    return out;
}

auto AppSession::applyVocsepEnsembleSettings() -> void {
    if (orchestrationModel == nullptr)
        return;
    const auto settings = getVocsepEnsembleSettings();
    orchestrationModel->setVocsepEnsembleEnabled(settings.enabled);
    orchestrationModel->setVocsepEnsembleAddP(settings.addP);
    orchestrationModel->setVocsepEnsembleRemoveP(settings.removeP);
}

auto AppSession::setVocsepEnsembleSettings(const VocsepEnsembleSettings &settings) -> void {
    auto orch = presetStore.settings.getOrCreateChildWithName("orchestration", nullptr);
    orch.setProperty("vocsepEnsemble", settings.enabled, nullptr);
    orch.setProperty("vocsepEnsembleAddP", static_cast<double>(settings.addP), nullptr);
    orch.setProperty("vocsepEnsembleRemoveP", static_cast<double>(settings.removeP), nullptr);
    presetStore.saveSettings();
    applyVocsepEnsembleSettings();
}

auto AppSession::vocsepEnsembleAvailable() const -> bool {
    return orchestrationModel != nullptr && orchestrationModel->supportsVocsepEnsemble()
           && voiceSeparation.isLoaded();
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
    reductionTransformerPianoReduction.signalThreadShouldExit();
    reductionTransformerAnticipated.signalThreadShouldExit();
    reductionTransformerMelodyFirst.signalThreadShouldExit();
    orchestrationTransformer.signalThreadShouldExit();
    outputPlayback.signalThreadShouldExit();

    // Unblock the UI immediately: clock/audio clear must not wait on ORT joins.
    musicTransformer.resetAheadThrottle();
    reductionTransformerV1.resetAheadThrottle();
    reductionTransformerV2.resetAheadThrottle();
    reductionTransformerPianoReduction.resetAheadThrottle();
    reductionTransformerAnticipated.resetAheadThrottle();
    reductionTransformerMelodyFirst.resetAheadThrottle();
    clock.stop();
    outputPlayback.resetProgress();
    outputPlayback.clearNoteOnHistory();
    musicTransformer.clearOutputHistory();
    reductionTransformerV1.clearOutputHistory();
    reductionTransformerV2.clearOutputHistory();
    reductionTransformerPianoReduction.clearOutputHistory();
    reductionTransformerAnticipated.clearOutputHistory();
    reductionTransformerMelodyFirst.clearOutputHistory();
    if (outputProcessor != nullptr)
        outputProcessor->clear();

    if (stopJoinInFlight.exchange(true, std::memory_order_acq_rel))
        return;

    // Join workers off the message thread so Stop stays responsive while Dense/IC finish ORT.
    std::thread([this] {
        stopThreadOrReport(musicTransformer, modelSamplingAlert);
        stopThreadOrReport(reductionTransformerV1, modelSamplingAlert);
        stopThreadOrReport(reductionTransformerV2, modelSamplingAlert);
        stopThreadOrReport(reductionTransformerPianoReduction, modelSamplingAlert);
        stopThreadOrReport(reductionTransformerAnticipated, modelSamplingAlert);
        stopThreadOrReport(reductionTransformerMelodyFirst, modelSamplingAlert);
        stopThreadOrReport(orchestrationTransformer, modelSamplingAlert);
        stopThreadOrReport(outputPlayback, modelSamplingAlert);
        stopJoinInFlight.store(false, std::memory_order_release);
    }).detach();
}

auto AppSession::setOrchestrationMode(OrchestrationMode mode) -> void {
    const auto previous = orchestrationTransformer.getMode();
    orchestrationTransformer.setMode(mode);
    midiInputProcess.getLaunchpadGrid().setSelectedMode(mode);

    // Accompaniment: input never rewinds the model or clears the queue. Entering it keeps
    // every note already generated; leaving it for Edit / Jam drops every note that has not
    // started (sounding notes finish) and rewinds the model to now.
    const bool accompaniment = mode == OrchestrationMode::Accompaniment;
    for (ReductionTransformer *reduction:
         {static_cast<ReductionTransformer *>(&musicTransformer),
          static_cast<ReductionTransformer *>(&reductionTransformerV1),
          static_cast<ReductionTransformer *>(&reductionTransformerV2),
          static_cast<ReductionTransformer *>(&reductionTransformerPianoReduction),
          static_cast<ReductionTransformer *>(&reductionTransformerAnticipated),
          static_cast<ReductionTransformer *>(&reductionTransformerMelodyFirst)})
        reduction->setKeepQueueOnInput(accompaniment);
    applyAccompanimentCap();

    if (previous == OrchestrationMode::Accompaniment && ! accompaniment) {
        const auto now = static_cast<int32_t>(clock.getTime());
        outputPlayback.modeCutRequests.push(now);
        activeReduction().requestModeCut(now);
    }
}

auto AppSession::applyAccompanimentCap() -> void {
    const int cap = midiInputProcess.getLaunchpadGrid().effectivePolyphonyCap();
    musicTransformer.setMaxPolyphony(cap);
    reductionTransformerV1.setMaxPolyphony(cap);
    reductionTransformerV2.setMaxPolyphony(cap);
    reductionTransformerPianoReduction.setMaxPolyphony(cap);
    reductionTransformerAnticipated.setMaxPolyphony(cap);
    reductionTransformerMelodyFirst.setMaxPolyphony(cap);
}

auto AppSession::forceRestart() -> void {
    if (restartInFlight.exchange(true, std::memory_order_acq_rel))
        return;

    auto aliveFlag = alive;
    postToMessageThread([this, aliveFlag] {
        if (! aliveFlag->load())
            return;
        stopGeneration();

        // stopGeneration joins the workers off the message thread: wait for that, then start.
        std::thread([this, aliveFlag] {
            for (int waitedMs = 0;
                 aliveFlag->load() && stopJoinInFlight.load(std::memory_order_acquire)
                 && waitedMs < 20000;
                 waitedMs += 10)
                juce::Thread::sleep(10);
            postToMessageThread([this, aliveFlag] {
                if (! aliveFlag->load())
                    return;
                startGeneration();
                syncPauseTopLeds();
                restartsCompleted.fetch_add(1, std::memory_order_acq_rel);
                restartInFlight.store(false, std::memory_order_release);
            });
        }).detach();
    });
}

auto AppSession::syncPauseTopLeds() -> void {
    auto &grid = midiInputProcess.getLaunchpadGrid();
    grid.setTopLed(0, LaunchpadLighting::kOrange); // force restart
    grid.setTopLed(1, activeReduction().generationPause.get() ? LaunchpadLighting::kRed
                                                              : LaunchpadLighting::kOff);
    grid.setTopLed(2, orchestrationTransformer.paused.get() ? LaunchpadLighting::kRed
                                                            : LaunchpadLighting::kOff);
}
