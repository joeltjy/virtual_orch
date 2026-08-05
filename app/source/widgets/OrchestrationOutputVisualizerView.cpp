#include "VirtualOrch/widgets/OrchestrationOutputVisualizerView.h"

#include "VirtualOrch/GeneralMidiInstruments.h"
#include "VirtualOrch/InstrumentConstants.h"
#include "VirtualOrch/ui/UiConstants.h"

OrchestrationOutputVisualizerView::OrchestrationOutputVisualizerView() {
    table.setModel(this);
    table.setColour(juce::ListBox::backgroundColourId, UiConstants::pianoRollBackground);
    table.getHeader().addColumn(UiConstants::workspaceOrchestrationOutputOnsetColumn,
                                onsetColumn,
                                80,
                                50,
                                160,
                                juce::TableHeaderComponent::notSortable);
    table.getHeader().addColumn(UiConstants::workspaceOrchestrationOutputDurColumn,
                                durColumn,
                                80,
                                50,
                                160,
                                juce::TableHeaderComponent::notSortable);
    table.getHeader().addColumn(UiConstants::workspaceOrchestrationOutputPitchColumn,
                                pitchColumn,
                                70,
                                40,
                                120,
                                juce::TableHeaderComponent::notSortable);
    table.getHeader().addColumn(UiConstants::workspaceOrchestrationOutputVelocityColumn,
                                velocityColumn,
                                70,
                                40,
                                120,
                                juce::TableHeaderComponent::notSortable);
    table.getHeader().addColumn(UiConstants::workspaceOrchestrationOutputProgramColumn,
                                programColumn,
                                180,
                                80,
                                400,
                                juce::TableHeaderComponent::notSortable);
    table.setHeaderHeight(22);
    table.setRowHeight(20);
    addAndMakeVisible(table);
}

auto OrchestrationOutputVisualizerView::setNotes(std::vector<OrchestrationNote> newNotes) -> void {
    notes = std::move(newNotes);
    table.updateContent();
    table.repaint();
}

auto OrchestrationOutputVisualizerView::resized() -> void {
    table.setBounds(getLocalBounds());
}

auto OrchestrationOutputVisualizerView::getNumRows() -> int {
    return static_cast<int>(notes.size());
}

auto OrchestrationOutputVisualizerView::paintRowBackground(juce::Graphics &g,
                                                           int /*rowNumber*/,
                                                           int width,
                                                           int height,
                                                           bool rowIsSelected) -> void {
    g.fillAll(rowIsSelected ? UiConstants::pianoRollBackground.brighter(0.08f)
                            : UiConstants::pianoRollBackground);
    juce::ignoreUnused(width, height);
}

auto OrchestrationOutputVisualizerView::formatSeconds(int32_t centiseconds) -> juce::String {
    return juce::String(static_cast<double>(centiseconds) / 100.0, 2);
}

auto OrchestrationOutputVisualizerView::programNameForLocalId(int32_t localInstrumentId)
    -> juce::String {
    if (const auto gmId = InstrumentConstants::toGmInstrumentId(localInstrumentId))
        return GeneralMidiInstruments::nameForId(*gmId);
    return "Local " + juce::String(localInstrumentId);
}

auto OrchestrationOutputVisualizerView::paintCell(juce::Graphics &g,
                                                  int rowNumber,
                                                  int columnId,
                                                  int width,
                                                  int height,
                                                  bool /*rowIsSelected*/) -> void {
    if (rowNumber < 0 || static_cast<size_t>(rowNumber) >= notes.size())
        return;

    const auto &note = notes[static_cast<size_t>(rowNumber)];
    juce::String text;
    switch (columnId) {
        case onsetColumn:
            text = formatSeconds(note.token.time);
            break;
        case durColumn:
            text = formatSeconds(note.token.getRealDuration());
            break;
        case pitchColumn:
            text = juce::String(note.token.getPitch());
            break;
        case velocityColumn:
            text = juce::String(note.velocity);
            break;
        case programColumn:
            text = programNameForLocalId(note.localInstrumentId);
            break;
        default:
            break;
    }

    g.setColour(UiConstants::workspaceWidgetTitleTextColour);
    g.setFont(13.0f);
    g.drawText(text, 4, 0, width - 8, height, juce::Justification::centredLeft, true);
}
