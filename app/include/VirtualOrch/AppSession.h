#pragma once

#include <JuceHeader.h>

#include <memory>

#include "VirtualOrch/Clock.h"
#include "VirtualOrch/InputFilter.h"
#include "VirtualOrch/MidiInputProcess.h"
#include "VirtualOrch/MusicTransformer.h"
#include "VirtualOrch/OSCBufferOutputProcessor.h"
#include "VirtualOrch/OrchestrationTransformer.h"
#include "VirtualOrch/OutputPlayback.h"
#include "VirtualOrch/OutputProcessor.h"
#include "VirtualOrch/PresetStore.h"
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

    MusicTransformer musicTransformer;
    OrchestrationTransformer orchestrationTransformer;
    std::unique_ptr<InputFilter> inputFilter;
    OutputPlayback outputPlayback;
    MidiInputProcess midiInputProcess;

    auto rebuildInputFilter() -> void;

    auto startGeneration() -> void;

    auto stopGeneration() -> void;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AppSession)
};
