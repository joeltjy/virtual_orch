#pragma once

#include <JuceHeader.h>

#include <atomic>
#include <functional>
#include <memory>

#include "VirtualOrch/Clock.h"
#include "VirtualOrch/InputFilter.h"
#include "VirtualOrch/KeyboardLaunchpad.h"
#include "VirtualOrch/MidiInputProcess.h"
#include "VirtualOrch/ModelSamplingAlert.h"
#include "VirtualOrch/MusicTransformer.h"
#include "VirtualOrch/OSCBufferOutputProcessor.h"
#include "VirtualOrch/OrchestrationTransformer.h"
#include "VirtualOrch/OutputPlayback.h"
#include "VirtualOrch/OutputProcessor.h"
#include "VirtualOrch/PerformanceActivity.h"
#include "VirtualOrch/PresetStore.h"
#include "VirtualOrch/ScriptedInputPlayer.h"
#include "VirtualOrch/reduction/ReductionTransformer.h"
#include "VirtualOrch/reduction/ReductionTransformerPianoReduction.h"
#include "VirtualOrch/reduction/ReductionTransformerV1.h"
#include "VirtualOrch/reduction/ReductionTransformerV2.h"
#include "VirtualOrch/reduction/MelodyFirstReductionTransformer.h"
#include "VirtualOrch/reduction/TwoInstrumentAnticipatedReductionTransformer.h"
#include "VirtualOrch/reduction/InformativeWaitReductionTransformer.h"
#include "VirtualOrch/reduction/HybridInformativeWaitReductionTransformer.h"
#include "VirtualOrch/reduction/HybridEventReductionTransformer.h"
#include "VirtualOrch/reduction/HybridChordTokenReductionTransformer.h"
#include "VirtualOrch/reduction/HybridGroupReductionTransformer.h"
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
    /** A TD press keeps this much of the generated future (cs) and regenerates the rest. */
    static constexpr int32_t kTdCutAheadCs = 100;
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
    /** A second piano (Additional MIDI): its notes are played exactly like the MIDI Input's. */
    juce::String selectedAdditionalMidiInputIdentifier;
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
    /** W2/W3 (legacy: rt_simultaneous_legacy_w2 / rt_simultaneous_legacy) with the WAIT sampling rules. */
    TwoInstrumentAnticipatedReductionTransformer reductionTransformerWait;
    /** HybridTransformer: v1 while the piano is silent, W2/W3 while it plays. */
    TwoInstrumentAnticipatedReductionTransformer reductionTransformerHybrid;
    InformativeWaitReductionTransformer reductionTransformerInformativeWait;
    HybridInformativeWaitReductionTransformer reductionTransformerHybridIW;
    HybridEventReductionTransformer reductionTransformerHybridEvent; // Hybrid v1.3 (event models)
    HybridChordTokenReductionTransformer reductionTransformerHybridChord; // Hybrid v1.2 (chord-token model 2)
    HybridGroupReductionTransformer reductionTransformerHybridGroup; // Hybrid v1.4 (group-token models m1_v4 / m2_v4)
    MelodyFirstReductionTransformer reductionTransformerMelodyFirst;
    OrchestrationTransformer orchestrationTransformer;
    VoiceSeparation voiceSeparation;
    /** A second vocsep for the user's notes in Edit mode (OrchestrationTransformer). */
    VoiceSeparation inputVoiceSeparation;
    std::unique_ptr<OrchestrationModel> orchestrationModel;
    TestOrchestrationTransformerThread testOrchestrationTransformerThread;
    std::unique_ptr<InputFilter> inputFilter;
    std::atomic<PlaybackSource> playbackSource{PlaybackSource::Orchestration};
    /** What the performance-view widgets read: playback activity and the Launchpad console. */
    PerformanceActivity::ActivityLog activityLog;
    PerformanceActivity::ConsoleLog consoleLog;

    OutputPlayback outputPlayback;
    MidiInputProcess midiInputProcess;

    /**
     * Scripted input: a MIDI file played as the play keyboard (Settings: MIDI Input =
     * "Scripted input (MIDI file)"). Declared after midiInputProcess, which it feeds.
     */
    ScriptedInputPlayer scriptedInput{
        [this](const juce::String &id, const juce::String &name, const juce::MidiMessage &message) {
            midiInputProcess.handleMidiFrom(id, name, message);
        }};
    [[nodiscard]] auto isScriptedInputSelected() const -> bool {
        return selectedMidiInputIdentifier == ScriptedInputPlayer::kScriptedInputIdentifier;
    }

    /** Computer keyboard as Launchpad (Settings: "Keyboard as Launchpad"); see KeyboardLaunchpad.h. */
    std::atomic<bool> keyboardLaunchpadEnabled{false};
    /**
     * A recorded session is loaded on the Replay page: it uses the process-wide simulated time
     * and log captures, so live generation must not start (startGeneration refuses).
     */
    std::atomic<bool> replayActive{false};
    KeyboardLaunchpad::Listener keyboardLaunchpad;

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

    /**
     * The UserSettings switches (OT: outer voices only shift outward, one instrument per octave shift; score viewer:
     * key signature) from / to settings.xml ("userSettings"). Loaded once at startup, saved on every change.
     */
    auto loadUserSettings() -> void;
    auto saveUserSettings() -> void;

    [[nodiscard]] auto getVocsepEnsembleSettings() -> VocsepEnsembleSettings;

    /** Persist the settings and push them to the current orchestration model. */
    auto setVocsepEnsembleSettings(const VocsepEnsembleSettings &settings) -> void;

    /** Push the stored settings to the current orchestration model (after a model switch). */
    auto applyVocsepEnsembleSettings() -> void;

    /** The model supports the ensemble and vocsep (which stamps the voice ids) is loaded. */
    [[nodiscard]] auto vocsepEnsembleAvailable() const -> bool;

    auto startGeneration() -> void;

    auto stopGeneration() -> void;

    /** Start / Stop without the worker threads: the offline simulator (LogReplay) steps them. */

    auto startGenerationForReplay(uint32_t clockCs) -> void;

    auto stopGenerationForReplay() -> void;

    /** Set the OT Edit/Jam mode and light the matching Launchpad mode pad. */
    auto setOrchestrationMode(OrchestrationMode mode) -> void;

    /**
     * Logit adjustments in a preset ("logitAdjustments"): the global BiasSettings switch and
     * silence gate, the LogitTD strength, the active T3'' / W2-W3 / Hybrid model's own switches
     * and taus ("model"; W2/W3's side for the Hybrid) and the Hybrid's v1 side ("hybridV1").
     */
    [[nodiscard]] auto logitAdjustmentsToVar() -> juce::var;
    auto applyLogitAdjustments(const juce::var &adjustments) -> void;
    /** The active T3'' / W2-W3 / Hybrid transformer, else nullptr. */
    /** The active reduction's own logit adjustments (T3'' / W2-W3 / Hybrid / Hybrid v1.1), else null. */
    [[nodiscard]] auto activeAnticipated() -> RtLogitAdjustments *;

    /** Accompaniment mode (Jam only): push the Launchpad's polyphony cap to every reduction. */
    auto applyAccompanimentCap() -> void;

    /**
     * Force restart (Launchpad top-row button 0) = stopGeneration() followed by
     * startGeneration(), exactly what the Stop and Start buttons do. The stop joins the workers
     * on another thread, so this waits for that join and then starts on the message thread.
     * Returns at once; pause flags are left as they are.
     */
    auto forceRestart() -> void;

    /** Runs a function on the message thread (tests replace it with a queue they pump). */
    std::function<void(std::function<void()>)> postToMessageThread = [](std::function<void()> f) {
        juce::MessageManager::callAsync(std::move(f));
    };

    /** Number of force restarts that have completed (tests / UI). */
    [[nodiscard]] auto getRestartsCompleted() const -> int {
        return restartsCompleted.load(std::memory_order_acquire);
    }

    /** Update Launchpad CC105/106 LEDs from current pause flags. */
    auto syncPauseTopLeds() -> void;

    [[nodiscard]] auto getPlaybackSource() const -> PlaybackSource {
        return playbackSource.load(std::memory_order_relaxed);
    }

    auto setPlaybackSource(PlaybackSource source) -> void {
        playbackSource.store(source, std::memory_order_relaxed);
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AppSession)

private:
    /** Cleared when the session is destroyed, so a pending restart never touches freed state. */
    std::shared_ptr<std::atomic<bool>> alive = std::make_shared<std::atomic<bool>>(true);
    std::atomic<bool> restartInFlight{false};
    int64_t currentRun = -1; // operation log run number, -1 = not running
    auto logRunStart() -> void;
    auto logRunStop() -> void;

public:
    /** Operation log: SESSION_START with the settings and models (origin "live" or "replay"). */
    auto logSessionStart(const juce::String &origin) -> void;

private:
    std::atomic<int> restartsCompleted{0};
};
