#include "VirtualOrch/widgets/ActiveInstrumentsView.h"

#include "VirtualOrch/GeneralMidiInstruments.h"
#include "VirtualOrch/InstrumentConstants.h"
#include "VirtualOrch/InstrumentUiColours.h"
#include "VirtualOrch/LaunchpadLighting.h"
#include "VirtualOrch/ui/UiConstants.h"

ActiveInstrumentsView::ActiveInstrumentsView() {
    setOpaque(true);
}

auto ActiveInstrumentsView::colourForPalette(uint8_t palette) -> juce::Colour {
    return InstrumentUiColours::fromPalette(palette);
}

auto ActiveInstrumentsView::shortNameForLocal(int32_t localId) -> juce::String {
    // Prefer app labels over GM first-word truncations (Acoustic/Taiko/Steel…).
    switch (localId) {
        case 15:
            return "Piano";
        case 17:
            return "Snare";
        case 18:
            return "Guitar";
        default:
            break;
    }

    if (const auto gmId = InstrumentConstants::toGmInstrumentId(localId)) {
        auto name = GeneralMidiInstruments::nameForId(*gmId);
        const auto space = name.indexOfChar(' ');
        if (space > 0)
            name = name.substring(0, space);
        if (name.length() > 8)
            name = name.substring(0, 8);
        return name;
    }
    return "L" + juce::String(localId);
}

auto ActiveInstrumentsView::setFromLaunchpad(const LaunchpadGrid &grid) -> void {
    rebuildCells(grid);
    repaint();
}

auto ActiveInstrumentsView::rebuildCells(const LaunchpadGrid &grid) -> void {
    for (auto &row: cells)
        for (auto &cell: row)
            cell = {};
    for (int col = 0; col < LaunchpadGrid::kTopCcCount; ++col) {
        auto &cell = cells[0][static_cast<size_t>(col)];
        const auto palette = grid.getTopLedColour(col);
        cell.fill = colourForPalette(palette);
        cell.hasContent = palette != LaunchpadLighting::kOff;
    }
    cells[0][1].label = UiConstants::workspaceActiveInstrumentsRtPauseLabel;
    cells[0][1].hasContent = true;
    cells[0][2].label = UiConstants::workspaceActiveInstrumentsOtPauseLabel;
    cells[0][2].hasContent = true;
    cells[0][8] = {};
    for (int padRow = 0; padRow < LaunchpadGrid::kRows; ++padRow) {
        const auto uiRow = static_cast<size_t>(padRow + 1);
        for (int padCol = 0; padCol < LaunchpadGrid::kCols; ++padCol) {
            auto &cell = cells[uiRow][static_cast<size_t>(padCol)];
            const auto &mapped = grid.padInstruments[static_cast<size_t>(padRow)]
                                                     [static_cast<size_t>(padCol)];
            if (! mapped.has_value()) {
                cell.fill = colourForPalette(LaunchpadLighting::kOff);
                continue;
            }

            cell.label = shortNameForLocal(*mapped);
            cell.hasContent = true;
            const auto palette =
                LaunchpadLighting::colourForGridEffective(padRow,
                                                          grid.getEffectiveState(padRow, padCol) != 0);
            cell.fill = colourForPalette(palette);
        }

        auto &side = cells[uiRow][8];
        const auto unmuted = grid.getSideUnmuted(padRow);
        const auto sidePalette = LaunchpadLighting::colourForSide(unmuted);
        side.fill = colourForPalette(sidePalette);
        side.hasContent = ! unmuted;
        side.label = unmuted ? juce::String() : "M";
    }
}

auto ActiveInstrumentsView::resized() -> void {
    auto bounds = getLocalBounds().toFloat().reduced(4.0f);
    auto labelStrip = bounds.removeFromLeft(static_cast<float>(kLabelStripWidth));

    // UI rows 1..4 (2nd–5th) = USER; rows 5..8 (6th–9th) = MODEL. Top row has no label.
    const auto cellH = bounds.getHeight() / static_cast<float>(kUiRows);
    userLabelBounds = labelStrip.withY(bounds.getY() + cellH)
                          .withHeight(cellH * 4.0f);
    modelLabelBounds = labelStrip.withY(bounds.getY() + cellH * 5.0f)
                           .withHeight(cellH * 4.0f);
    gridBounds = bounds;
}

auto ActiveInstrumentsView::paintGroupBracket(juce::Graphics &g,
                                              juce::Rectangle<float> labelArea,
                                              const juce::String &text) const -> void {
    if (labelArea.isEmpty())
        return;

    auto braceArea = labelArea.removeFromRight(10.0f).reduced(0.0f, 2.0f);
    juce::Path brace;
    brace.startNewSubPath(braceArea.getRight(), braceArea.getY());
    brace.lineTo(braceArea.getX(), braceArea.getY());
    brace.lineTo(braceArea.getX(), braceArea.getBottom());
    brace.lineTo(braceArea.getRight(), braceArea.getBottom());
    g.setColour(UiConstants::workspaceWidgetTitleTextColour.withAlpha(0.85f));
    g.strokePath(brace, juce::PathStrokeType(1.5f));

    g.setFont(juce::Font(juce::jlimit(14.0f, 22.0f, labelArea.getWidth() * 0.55f),
                         juce::Font::bold));
    g.drawFittedText(text, labelArea.toNearestInt(), juce::Justification::centred, 1);
}

auto ActiveInstrumentsView::paint(juce::Graphics &g) -> void {
    g.fillAll(UiConstants::pianoRollBackground);

    paintGroupBracket(g, userLabelBounds, UiConstants::workspaceOrchestrationDebugUserColumn);
    paintGroupBracket(g, modelLabelBounds, UiConstants::workspaceOrchestrationDebugModelColumn);

    if (gridBounds.isEmpty())
        return;

    const auto cellW = gridBounds.getWidth() / static_cast<float>(kUiCols);
    const auto cellH = gridBounds.getHeight() / static_cast<float>(kUiRows);
    const auto gap = 1.5f;

    for (int row = 0; row < kUiRows; ++row) {
        for (int col = 0; col < kUiCols; ++col) {
            const auto &cell = cells[static_cast<size_t>(row)][static_cast<size_t>(col)];
            auto cellBounds =
                juce::Rectangle<float>(gridBounds.getX() + static_cast<float>(col) * cellW,
                                       gridBounds.getY() + static_cast<float>(row) * cellH,
                                       cellW,
                                       cellH)
                    .reduced(gap);

            const bool isTopRight = row == 0 && col == 8;
            if (isTopRight) {
                g.setColour(UiConstants::pianoRollBackground.brighter(0.04f));
                g.fillRoundedRectangle(cellBounds, 3.0f);
                continue;
            }

            g.setColour(cell.fill);
            g.fillRoundedRectangle(cellBounds, 3.0f);
            g.setColour(juce::Colours::white.withAlpha(0.12f));
            g.drawRoundedRectangle(cellBounds, 3.0f, 1.0f);

            if (cell.label.isNotEmpty()) {
                const bool lit = cell.fill.getBrightness() > 0.25f;
                g.setColour(lit ? juce::Colours::black.withAlpha(0.85f)
                                : UiConstants::workspaceWidgetTitleTextColour.withAlpha(0.75f));
                g.setFont(juce::Font(
                    juce::jlimit(14.0f, 28.0f, cellBounds.getHeight() * 0.55f),
                    juce::Font::bold));
                g.drawFittedText(cell.label,
                                 cellBounds.toNearestInt().reduced(1),
                                 juce::Justification::centred,
                                 2);
            }
        }
    }
}
