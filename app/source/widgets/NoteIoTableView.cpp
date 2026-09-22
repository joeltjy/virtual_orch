#include "VirtualOrch/widgets/NoteIoTableView.h"

#include "VirtualOrch/GeneralMidiInstruments.h"
#include "VirtualOrch/InstrumentConstants.h"
#include "VirtualOrch/InstrumentUiColours.h"
#include "VirtualOrch/ui/UiConstants.h"

NoteIoTableView::NoteIoTableView(bool showChannelIn)
    : showChannel(showChannelIn) {
    table.setModel(this);
    table.setColour(juce::ListBox::backgroundColourId, UiConstants::pianoRollBackground);
    table.getHeader().addColumn(UiConstants::workspaceNoteIoOnsetColumn,
                                onsetColumn,
                                70,
                                40,
                                140,
                                juce::TableHeaderComponent::notSortable);
    table.getHeader().addColumn(UiConstants::workspaceNoteIoDurColumn,
                                durColumn,
                                70,
                                40,
                                140,
                                juce::TableHeaderComponent::notSortable);
    table.getHeader().addColumn(UiConstants::workspaceNoteIoPitchColumn,
                                pitchColumn,
                                60,
                                40,
                                100,
                                juce::TableHeaderComponent::notSortable);
    table.getHeader().addColumn(UiConstants::workspaceNoteIoVelocityColumn,
                                velocityColumn,
                                60,
                                40,
                                100,
                                juce::TableHeaderComponent::notSortable);
    table.getHeader().addColumn(UiConstants::workspaceNoteIoProgramColumn,
                                programColumn,
                                140,
                                80,
                                400,
                                juce::TableHeaderComponent::notSortable);
    if (showChannel) {
        table.getHeader().addColumn(UiConstants::workspaceNoteIoChannelColumn,
                                    channelColumn,
                                    60,
                                    40,
                                    100,
                                    juce::TableHeaderComponent::notSortable);
    }
    table.setHeaderHeight(22);
    table.setRowHeight(20);
    addAndMakeVisible(table);
}

auto NoteIoTableView::setRows(std::vector<NoteIoRow> newRows) -> void {
    rows = std::move(newRows);
    table.updateContent();
    table.repaint();
}

auto NoteIoTableView::resized() -> void {
    table.setBounds(getLocalBounds());
}

auto NoteIoTableView::getNumRows() -> int {
    return static_cast<int>(rows.size());
}

auto NoteIoTableView::paintRowBackground(juce::Graphics &g,
                                         int /*rowNumber*/,
                                         int width,
                                         int height,
                                         bool rowIsSelected) -> void {
    g.fillAll(rowIsSelected ? UiConstants::pianoRollBackground.brighter(0.08f)
                            : UiConstants::pianoRollBackground);
    juce::ignoreUnused(width, height);
}

auto NoteIoTableView::formatSeconds(int32_t centiseconds) -> juce::String {
    return juce::String(static_cast<double>(centiseconds) / 100.0, 2);
}

auto NoteIoTableView::programNameForLocalId(int32_t localInstrumentId) -> juce::String {
    if (localInstrumentId < 0)
        return "-";
    if (const auto gmId = InstrumentConstants::toGmInstrumentId(localInstrumentId))
        return GeneralMidiInstruments::nameForId(*gmId);
    return "Local " + juce::String(localInstrumentId);
}

auto NoteIoTableView::paintCell(juce::Graphics &g,
                                int rowNumber,
                                int columnId,
                                int width,
                                int height,
                                bool /*rowIsSelected*/) -> void {
    if (rowNumber < 0 || static_cast<size_t>(rowNumber) >= rows.size())
        return;

    const auto &row = rows[static_cast<size_t>(rowNumber)];
    juce::String text;
    switch (columnId) {
        case onsetColumn:
            text = formatSeconds(row.onset);
            break;
        case durColumn:
            text = row.durationCs < 0 ? juce::String("-") : formatSeconds(row.durationCs);
            break;
        case pitchColumn:
            text = juce::String(row.pitch);
            break;
        case velocityColumn:
            text = juce::String(row.velocity);
            break;
        case programColumn:
            text = programNameForLocalId(row.localInstrumentId);
            break;
        case channelColumn:
            text = row.channel < 0 ? juce::String("-") : juce::String(row.channel);
            break;
        default:
            break;
    }

    g.setColour(columnId == programColumn ? InstrumentUiColours::forLocalInstrument(row.localInstrumentId)
                                          : UiConstants::workspaceWidgetTitleTextColour);
    g.setFont(13.0f);
    g.drawText(text, 4, 0, width - 8, height, juce::Justification::centredLeft, true);
}
