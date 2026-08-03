#pragma once

#include <JuceHeader.h>

#include <cstdint>
#include <set>
#include <vector>

/** Read-only two-column table: User | Model GM instrument names. */
class ActiveInstrumentsView : public juce::Component, private juce::TableListBoxModel {
public:
    ActiveInstrumentsView();

    auto setInstruments(std::set<int32_t> user, std::set<int32_t> model) -> void;

    auto resized() -> void override;

private:
    enum ColumnIds : int {
        userColumn = 1,
        modelColumn = 2
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

    juce::TableListBox table;
    std::vector<int32_t> userIds;
    std::vector<int32_t> modelIds;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ActiveInstrumentsView)
};
