#pragma once

#include <JuceHeader.h>

#include <memory>

#include "VirtualOrch/Clock.h"
#include "VirtualOrch/DenseMusicTransformer.h"
#include "VirtualOrch/InputFilter.h"
#include "VirtualOrch/MidiInputProcess.h"
#include "VirtualOrch/MusicTransformer.h"
#include "VirtualOrch/OSCBufferOutputProcessor.h"
#include "VirtualOrch/OrchestrationTransformer.h"
#include "VirtualOrch/OutputPlayback.h"
#include "VirtualOrch/OutputProcessor.h"
#include "VirtualOrch/PresetStore.h"
#include "VirtualOrch/orchestration-models/OrchestrationModel.h"
#include "VirtualOrch/TestOrchestrationTransformerThread.h"
#include "VirtualOrch/ui/ModelConfigurationComponent.h"

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
    juce::String selectedLaunchpadMidiIdentifier;
    juce::String selectedMtcClockIdentifier;
    bool mtcClockActive = false;

    MusicModelArch musicModelArch = MusicModelArch::Amt;
    MusicTransformer musicTransformer;
    DenseMusicTransformer denseMusicTransformer;
    OrchestrationTransformer orchestrationTransformer;
    std::unique_ptr<OrchestrationModel> orchestrationModel;
    TestOrchestrationTransformerThread testOrchestrationTransformerThread;
    std::unique_ptr<InputFilter> inputFilter;
    OutputPlayback outputPlayback;
    MidiInputProcess midiInputProcess;

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

    auto startGeneration() -> void;

    auto stopGeneration() -> void;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AppSession)
};
