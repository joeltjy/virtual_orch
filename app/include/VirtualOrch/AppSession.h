#pragma once

#include <JuceHeader.h>

#include <atomic>
#include <memory>

#include "VirtualOrch/Clock.h"
#include "VirtualOrch/InputFilter.h"
#include "VirtualOrch/MidiInputProcess.h"
#include "VirtualOrch/ModelSamplingAlert.h"
#include "VirtualOrch/MusicTransformer.h"
#include "VirtualOrch/OSCBufferOutputProcessor.h"
#include "VirtualOrch/OrchestrationTransformer.h"
#include "VirtualOrch/OutputPlayback.h"
#include "VirtualOrch/OutputProcessor.h"
#include "VirtualOrch/PerformanceActivity.h"
#include "VirtualOrch/PresetStore.h"
#include "VirtualOrch/reduction/ReductionTransformer.h"
#include "VirtualOrch/reduction/ReductionTransformerPianoReduction.h"
#include "VirtualOrch/reduction/ReductionTransformerV1.h"
#include "VirtualOrch/reduction/ReductionTransformerV2.h"
#include "VirtualOrch/reduction/TwoInstrumentAnticipatedReductionTransformer.h"
#include "VirtualOrch/orchestration-models/OrchestrationModel.h"
#include "VirtualOrch/TestOrchestrationTransformerThread.h"
#include "VirtualOrch/ui/ModelConfigurationComponent.h"
#include "VirtualOrch/vocsep/VoiceSeparation.h"

/**
 * Shared runtime owned by AppRootComponent and used by Settings + Workspace.
 * Declaration order matches construction / reference lifetime requirements.
 */
class AppSession {
public:
    AppSession();

    ~AppSession();

    ModelConfig modelConfig;
    PresetStore presetStore;
    Metrics metrics;
    Clock clock;

    int32_t visualizationBufferSize = 400;

    std::unique_ptr<OutputProcessor> outputProcessor;
    std::unique_ptr<OSCBufferOutputProcessor> bufferOutputProcessor;

    juce::String selectedMidiInputIdentifier;
    /** Separate keyboard used only for sustain (CC64) → generationPause. */
    juce::String selectedPedalMidiInputIdentifier;
    juce::String selectedLaunchpadMidiIdentifier;
    juce::String selectedMtcClockIdentifier;
    bool mtcClockActive = false;

    MusicModelArch musicModelArch = MusicModelArch::DenseV1;
    ModelSamplingAlert modelSamplingAlert;
    MusicTransformer musicTransformer;
    ReductionTransformerV1 reductionTransformerV1;
    ReductionTransformerV2 reductionTransformerV2;
    ReductionTransformerPianoReduction reductionTransformerPianoReduction;
    TwoInstrumentAnticipatedReductionTransformer reductionTransformerAnticipated;
    OrchestrationTransformer orchestrationTransformer;
    VoiceSeparation voiceSeparation;
    std::unique_ptr<OrchestrationModel> orchestrationModel;
    TestOrchestrationTransformerThread testOrchestrationTransformerThread;
    std::unique_ptr<InputFilter> inputFilter;
    std::atomic<PlaybackSource> playbackSource{PlaybackSource::Orchestration};
    /** What the performance-view widgets read: playback activity and the Launchpad console. */
    PerformanceActivity::ActivityLog activityLog;
    PerformanceActivity::ConsoleLog consoleLog;

    OutputPlayback outputPlayback;
    MidiInputProcess midiInputProcess;

    [[nodiscard]] auto activeReduction() -> ReductionTransformer &;

    [[nodiscard]] auto activeReduction() const -> const ReductionTransformer &;

    auto rebuildInputFilter() -> void;

    /** Point input filter + OT hooks at the active music backend. */
    auto bindActiveMusicBackend() -> void;

    auto setMusicModelArch(MusicModelArch arch) -> void;

    [[nodiscard]] auto isMusicModelLoaded() const -> bool;

    [[nodiscard]] auto isMusicThreadRunning() const -> bool;

    auto getMusicDirectInputBlock() -> juce::Atomic<bool> &;

    [[nodiscard]] auto getActiveInputData() const -> std::vector<int32_t>;

    auto setOnInputDataChanged(std::function<void(std::vector<int32_t>)> callback) -> void;

    auto setOrchestrationModel(const juce::String &name) -> bool;

    /** Global vocsep-ensemble switch and proportions (shared by every orchestration model). */
    struct VocsepEnsembleSettings {
        bool enabled = false;
        float addP = OrchestrationModel::kEnsembleAddPDefault;
        float removeP = OrchestrationModel::kEnsembleRemovePDefault;
    };

    [[nodiscard]] auto getVocsepEnsembleSettings() -> VocsepEnsembleSettings;

    /** Persist the settings and push them to the current orchestration model. */
    auto setVocsepEnsembleSettings(const VocsepEnsembleSettings &settings) -> void;

    /** Push the stored settings to the current orchestration model (after a model switch). */
    auto applyVocsepEnsembleSettings() -> void;

    /** The model supports the ensemble and vocsep (which stamps the voice ids) is loaded. */
    [[nodiscard]] auto vocsepEnsembleAvailable() const -> bool;

    auto startGeneration() -> void;

    auto stopGeneration() -> void;

    /** Set the OT Edit/Jam mode and light the matching Launchpad mode pad. */
    auto setOrchestrationMode(OrchestrationMode mode) -> void;

    /** Accompaniment mode (Jam only): push the Launchpad's polyphony cap to every reduction. */
    auto applyAccompanimentCap() -> void;

    /** Update Launchpad CC105/106 LEDs from current pause flags. */
    auto syncPauseTopLeds() -> void;

    [[nodiscard]] auto getPlaybackSource() const -> PlaybackSource {
        return playbackSource.load(std::memory_order_relaxed);
    }

    auto setPlaybackSource(PlaybackSource source) -> void {
        playbackSource.store(source, std::memory_order_relaxed);
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AppSession)
};
