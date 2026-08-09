#pragma once

#include <JuceHeader.h>

#include <vector>

#include "VirtualOrch/widgets/NoteIoRow.h"

/**
 * Read-only note table.
 */
class NoteIoTableView : public juce::Component, private juce::TableListBoxModel {
public:
    explicit NoteIoTableView(bool showChannel);

    auto setRows(std::vector<NoteIoRow> rows) -> void;

    auto resized() -> void override;

private:
    enum ColumnIds : int {
        onsetColumn = 1,
        durColumn = 2,
        pitchColumn = 3,
        velocityColumn = 4,
        programColumn = 5,
        channelColumn = 6
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

    bool showChannel = false;
    juce::TableListBox table;
    std::vector<NoteIoRow> rows;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NoteIoTableView)
};
