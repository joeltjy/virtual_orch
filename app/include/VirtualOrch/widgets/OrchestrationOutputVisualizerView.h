#pragma once

#include <JuceHeader.h>

#include <vector>

#include "VirtualOrch/orchestration-models/OrchestrationModel.h"

/** Read-only table of OT output notes: onset, dur, pitch, velocity, program name. */
class OrchestrationOutputVisualizerView : public juce::Component, private juce::TableListBoxModel {
public:
    OrchestrationOutputVisualizerView();

    auto setNotes(std::vector<OrchestrationNote> notes) -> void;

    auto resized() -> void override;

private:
    enum ColumnIds : int {
        onsetColumn = 1,
        durColumn = 2,
        pitchColumn = 3,
        velocityColumn = 4,
        programColumn = 5
    };

    auto getNumRows() -> int override;

    auto paintRowBackground(juce::Graphics &g,
                            int rowNumber,
                            int width,
                            int height,
                            bool rowIsSelected) -> void override;

    auto paintCell(juce::Graphics &g,
                   int rowNumber,
                   int columnId,
                   int width,
                   int height,
                   bool rowIsSelected) -> void override;

    static auto formatSeconds(int32_t centiseconds) -> juce::String;

    static auto programNameForLocalId(int32_t localInstrumentId) -> juce::String;

    juce::TableListBox table;
    std::vector<OrchestrationNote> notes;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(OrchestrationOutputVisualizerView)
};
