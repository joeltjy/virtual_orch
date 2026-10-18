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
        case 4:
            return "Tremolo";
        case 5:
            return "Pizz";
        case 6:
            return "Str Ens";
        case 12:
            return "Horn";
        case 15:
            return "Piano";
        case 16:
            return "Timpani";
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

    const auto cellAt = [this](int padRow, int col) -> CellState & {
        return cells[static_cast<size_t>(padRow + 1)][static_cast<size_t>(col)];
    };

    // Top CC row (UI row 0): pause buttons light red while paused.
    const auto setPause = [&](int idx, const char *label) {
        auto &cell = cells[0][static_cast<size_t>(idx)];
        cell.label = label;
        cell.onPalette = LaunchpadLighting::kRed;
        cell.lit = grid.getTopLedColour(idx) != LaunchpadLighting::kOff;
    };
    setPause(1, UiConstants::workspaceActiveInstrumentsRtPauseLabel);
    setPause(2, UiConstants::workspaceActiveInstrumentsOtPauseLabel);

    // Mode row.
    const auto mode = grid.getSelectedMode();
    auto &edit = cellAt(LaunchpadGrid::kModeRow, LaunchpadGrid::kEditCol);
    edit.label = UiConstants::workspaceActiveInstrumentsEditLabel;
    edit.onPalette = LaunchpadLighting::colourForModePad(true);
    edit.lit = mode == OrchestrationMode::Edit;
    auto &jam = cellAt(LaunchpadGrid::kModeRow, LaunchpadGrid::kJamCol);
    jam.label = UiConstants::workspaceActiveInstrumentsJamLabel;
    jam.onPalette = LaunchpadLighting::colourForModePad(true);
    jam.lit = mode == OrchestrationMode::Jam;

    // TD row + AUTO_TD side button.
    const bool tdAuto = TemporalDensity::isAuto();
    for (int level = 0; level < LaunchpadGrid::kTdLevelCount; ++level) {
        auto &cell = cellAt(LaunchpadGrid::kTdRow, level);
        cell.label = UiConstants::workspaceActiveInstrumentsTdPrefix
                     + juce::String(TemporalDensity::kLevels[static_cast<size_t>(level)]);
        cell.onPalette = LaunchpadLighting::colourForTdLevel(level, true);
        cell.lit = ! tdAuto && level == grid.getManualTdLevelIndex();
    }
    auto &autoTd = cellAt(LaunchpadGrid::kTdRow, LaunchpadGrid::kCols);
    autoTd.label = UiConstants::workspaceActiveInstrumentsAutoTdLabel;
    autoTd.onPalette = LaunchpadLighting::colourForAutoTd(tdAuto);
    autoTd.lit = true;

    // Flush side button.
    auto &flush = cellAt(LaunchpadGrid::kFlushRow, LaunchpadGrid::kCols);
    flush.label = UiConstants::workspaceActiveInstrumentsFlushLabel;
    flush.onPalette = LaunchpadLighting::kFlush;
    flush.lit = true;

    // Instrument pads (input + model rows).
    for (int padRow = 0; padRow < LaunchpadGrid::kRows; ++padRow) {
        for (int padCol = 0; padCol < LaunchpadGrid::kCols; ++padCol) {
            const auto &mapped = grid.padInstruments[static_cast<size_t>(padRow)]
                                                     [static_cast<size_t>(padCol)];
            if (! mapped.has_value())
                continue;
            auto &cell = cellAt(padRow, padCol);
            cell.label = shortNameForLocal(*mapped);
            cell.onPalette = LaunchpadLighting::colourForInstrumentPad(*mapped, true);
            cell.lit = grid.getEffectiveState(padRow, padCol) != 0;
        }
    }
}

auto ActiveInstrumentsView::resized() -> void {
    auto bounds = getLocalBounds().toFloat().reduced(4.0f);
    auto labelStrip = bounds.removeFromLeft(static_cast<float>(kLabelStripWidth));

    // UI row = pad row + 1 (top CC row is UI row 0).
    const auto cellH = bounds.getHeight() / static_cast<float>(kUiRows);
    const auto setHeight = cellH * static_cast<float>(LaunchpadGrid::kInstrumentRowsPerSet);
    userLabelBounds =
        labelStrip.withY(bounds.getY() + cellH * static_cast<float>(LaunchpadGrid::kUserRowFirst + 1))
            .withHeight(setHeight);
    modelLabelBounds =
        labelStrip.withY(bounds.getY() + cellH * static_cast<float>(LaunchpadGrid::kModelRowFirst + 1))
            .withHeight(setHeight);
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

    paintGroupBracket(g, userLabelBounds, UiConstants::workspaceActiveInstrumentsInputGroupLabel);
    paintGroupBracket(g, modelLabelBounds, UiConstants::workspaceActiveInstrumentsModelGroupLabel);

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

            const auto onColour = colourForPalette(cell.onPalette);
            const auto offColour = colourForPalette(LaunchpadLighting::kOff);
            const bool labelled = cell.label.isNotEmpty();
            // Unlit buttons keep a faint tint of their lit colour so groups stay readable.
            g.setColour(cell.lit ? onColour
                                 : (labelled ? offColour.interpolatedWith(onColour, 0.22f)
                                             : offColour));
            g.fillRoundedRectangle(cellBounds, 3.0f);
            g.setColour(juce::Colours::white.withAlpha(0.12f));
            g.drawRoundedRectangle(cellBounds, 3.0f, 1.0f);

            if (labelled) {
                g.setColour(cell.lit ? juce::Colours::black.withAlpha(0.85f)
                                     : UiConstants::workspaceWidgetTitleTextColour.withAlpha(0.85f));
                const auto fontHeight = juce::jlimit(
                    14.0f, 30.0f, juce::jmin(cellBounds.getHeight() * 0.42f,
                                             cellBounds.getWidth() * 0.30f));
                g.setFont(juce::Font(fontHeight, juce::Font::bold));
                g.drawFittedText(cell.label,
                                 cellBounds.toNearestInt().reduced(2),
                                 juce::Justification::centred,
                                 2,
                                 0.6f);
            }
        }
    }
}
