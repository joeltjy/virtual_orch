#include "VirtualOrch/widgets/ActiveInstrumentsView.h"

#include "VirtualOrch/GeneralMidiInstruments.h"
#include "VirtualOrch/InstrumentConstants.h"
#include "VirtualOrch/ui/UiConstants.h"

ActiveInstrumentsView::ActiveInstrumentsView() {
    table.setModel(this);
    table.setColour(juce::ListBox::backgroundColourId, UiConstants::pianoRollBackground);
    table.getHeader().addColumn(UiConstants::workspaceOrchestrationDebugUserColumn,
                                userColumn,
                                160,
                                80,
                                400,
                                juce::TableHeaderComponent::notSortable);
    table.getHeader().addColumn(UiConstants::workspaceOrchestrationDebugModelColumn,
                                modelColumn,
                                160,
                                80,
                                400,
                                juce::TableHeaderComponent::notSortable);
    table.setHeaderHeight(22);
    table.setRowHeight(20);
    addAndMakeVisible(table);
}

auto ActiveInstrumentsView::setInstruments(std::set<int32_t> user, std::set<int32_t> model) -> void {
    userIds.assign(user.begin(), user.end());
    modelIds.assign(model.begin(), model.end());
    table.updateContent();
    table.repaint();
}

auto ActiveInstrumentsView::resized() -> void {
    table.setBounds(getLocalBounds());
}

auto ActiveInstrumentsView::getNumRows() -> int {
    return static_cast<int>(std::max(userIds.size(), modelIds.size()));
}

auto ActiveInstrumentsView::paintRowBackground(juce::Graphics &g,
                                               int /*rowNumber*/,
                                               int width,
                                               int height,
                                               bool rowIsSelected) -> void {
    g.fillAll(rowIsSelected ? UiConstants::pianoRollBackground.brighter(0.08f)
                            : UiConstants::pianoRollBackground);
    juce::ignoreUnused(width, height);
}

auto ActiveInstrumentsView::paintCell(juce::Graphics &g,
                                      int rowNumber,
                                      int columnId,
                                      int width,
                                      int height,
                                      bool /*rowIsSelected*/) -> void {
    juce::String text;
    if (columnId == userColumn && rowNumber >= 0
        && static_cast<size_t>(rowNumber) < userIds.size()) {
        const auto localId = userIds[static_cast<size_t>(rowNumber)];
        if (const auto gmId = InstrumentConstants::toGmInstrumentId(localId))
            text = GeneralMidiInstruments::nameForId(*gmId);
        else
            text = "Local " + juce::String(localId);
    } else if (columnId == modelColumn && rowNumber >= 0
               && static_cast<size_t>(rowNumber) < modelIds.size()) {
        const auto localId = modelIds[static_cast<size_t>(rowNumber)];
        if (const auto gmId = InstrumentConstants::toGmInstrumentId(localId))
            text = GeneralMidiInstruments::nameForId(*gmId);
        else
            text = "Local " + juce::String(localId);
    }

    g.setColour(UiConstants::workspaceWidgetTitleTextColour);
    g.setFont(13.0f);
    g.drawText(text, 4, 0, width - 8, height, juce::Justification::centredLeft, true);
}
