#include "VirtualOrch/widgets/ActiveInstrumentsView.h"

#include "VirtualOrch/GeneralMidiInstruments.h"
#include "VirtualOrch/InstrumentConstants.h"
#include "VirtualOrch/InstrumentUiColours.h"
#include "VirtualOrch/KeyboardLaunchpad.h"
#include "VirtualOrch/LaunchpadLighting.h"
#include "VirtualOrch/MoodPresets.h"
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
    {
        auto &restart = cells[0][0];
        restart.label = UiConstants::workspaceActiveInstrumentsRestartLabel;
        restart.onPalette = LaunchpadLighting::kOrange;
        restart.lit = true;
    }
    setPause(2, UiConstants::workspaceActiveInstrumentsOtPauseLabel);
    setPause(3, UiConstants::workspaceActiveInstrumentsInputBlockLabel); // CC107, red while blocked

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
    auto &accompaniment = cellAt(LaunchpadGrid::kModeRow, LaunchpadGrid::kAccompanimentCol);
    accompaniment.label = UiConstants::workspaceActiveInstrumentsAccompanimentLabel;
    accompaniment.onPalette = LaunchpadLighting::colourForModePad(true);
    accompaniment.lit = mode == OrchestrationMode::Accompaniment;

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

    // Flush / Reset side buttons (Edit only).
    auto &flush = cellAt(LaunchpadGrid::kFlushRow, LaunchpadGrid::kCols);
    flush.label = UiConstants::workspaceActiveInstrumentsFlushLabel;
    flush.onPalette = LaunchpadLighting::kFlush;
    flush.lit = grid.sideColour(LaunchpadGrid::kFlushRow) != LaunchpadLighting::kOff;
    auto &reset = cellAt(LaunchpadGrid::kResetRow, LaunchpadGrid::kCols);
    reset.label = UiConstants::workspaceActiveInstrumentsResetLabel;
    reset.onPalette = LaunchpadLighting::kReset;
    reset.lit = grid.sideColour(LaunchpadGrid::kResetRow) != LaunchpadLighting::kOff;

    // Mood presets (bottom row): always shown, the active one brightest.
    for (int preset = 0; preset < LaunchpadGrid::kPresetCount; ++preset) {
        auto &cell = cellAt(LaunchpadGrid::kPresetRow, preset);
        cell.label = MoodPresets::all()[static_cast<size_t>(preset)].name;
        cell.onPalette = grid.presetPadColour(preset);
        cell.lit = true;
    }

    // Register (row 6): LOW / MEDIUM / HIGH pads and the AUTO side button.
    {
        static const char *const labels[] = {UiConstants::workspaceActiveInstrumentsRegisterLowLabel,
                                             UiConstants::workspaceActiveInstrumentsRegisterMediumLabel,
                                             UiConstants::workspaceActiveInstrumentsRegisterHighLabel};
        const bool registerAuto = RegisterRange::isAuto();
        for (int pad = 0; pad < LaunchpadGrid::kRegisterPadCount; ++pad) {
            const int reg = *LaunchpadGrid::registerForPad(LaunchpadGrid::kRegisterRow, pad);
            auto &cell = cellAt(LaunchpadGrid::kRegisterRow, pad);
            cell.label = labels[pad];
            cell.onPalette = LaunchpadLighting::colourForRegister(reg, true);
            cell.lit = ! registerAuto && reg == RegisterRange::get();
        }
        auto &autoReg = cellAt(LaunchpadGrid::kRegisterRow, LaunchpadGrid::kCols);
        autoReg.label = UiConstants::workspaceActiveInstrumentsAutoRegisterLabel;
        autoReg.onPalette = LaunchpadLighting::colourForAutoTd(registerAuto);
        autoReg.lit = true;
    }

    // Row 6, col 4: the monophonic pad (max polyphony 1), lit in Jam and Accompaniment while on.
    const bool polyActive = grid.isPolyphonyActive();
    for (int pad = 0; pad < LaunchpadGrid::kPolyphonyPadCount; ++pad) {
        const int col = LaunchpadGrid::kPolyphonyColFirst + pad;
        const int polyphony = *LaunchpadGrid::polyphonyForPad(LaunchpadGrid::kAccompRow, col);
        auto &cell = cellAt(LaunchpadGrid::kAccompRow, col);
        cell.label = polyphony == 1 ? juce::String("Mono") : UiConstants::workspaceActiveInstrumentsPolyphonyPrefix + juce::String(polyphony);
        cell.onPalette = LaunchpadLighting::colourForPolyphonyPad(true, true);
        cell.lit = polyActive && polyphony == grid.getMaxPolyphony();
    }

    // Computer-keyboard keys (UI row = pad row + 1, UI col 8 = side; UI row 0 = top buttons).
    for (int uiRow = 0; uiRow < kUiRows; ++uiRow)
        for (int uiCol = 0; uiCol < kUiCols; ++uiCol) {
            KeyboardLaunchpad::Target target;
            if (uiRow == 0)
                target = {-1, -1, uiCol};
            else
                target = {uiRow - 1, uiCol == LaunchpadGrid::kCols ? LaunchpadGrid::kSideCol : uiCol, -1};
            cells[static_cast<size_t>(uiRow)][static_cast<size_t>(uiCol)].key = KeyboardLaunchpad::labelForTarget(target);
        }

    // Instrument pads (your side cols 0-3, the model's 4-7): the same colours as the pads, so
    // the side the pads do not control shows grey (yours) or dim (the model's).
    controlsUserSide = grid.controlledTarget() == InstrumentUpdateTarget::User;
    for (int padRow = 0; padRow < LaunchpadGrid::kRows; ++padRow) {
        for (int padCol = 0; padCol < LaunchpadGrid::kCols; ++padCol) {
            const auto &mapped = grid.padInstruments[static_cast<size_t>(padRow)]
                                                     [static_cast<size_t>(padCol)];
            if (! mapped.has_value())
                continue;
            auto &cell = cellAt(padRow, padCol);
            cell.label = shortNameForLocal(*mapped);
            cell.lit = grid.getEffectiveState(padRow, padCol) != 0;
            cell.onPalette = cell.lit ? grid.instrumentPadColour(padRow, padCol)
                                      : LaunchpadLighting::colourForInstrumentPad(*mapped, true);
        }
    }
}

auto ActiveInstrumentsView::resized() -> void {
    auto bounds = getLocalBounds().toFloat().reduced(4.0f);
    auto labelStrip = bounds.removeFromLeft(static_cast<float>(kLabelStripWidth) * std::min(textScale, 1.6f));

    // UI row = pad row + 1 (top CC row is UI row 0).
    const auto cellH = bounds.getHeight() / static_cast<float>(kUiRows);
    instrumentLabelBounds =
        labelStrip.withY(bounds.getY() + cellH * static_cast<float>(LaunchpadGrid::kInstrumentRowFirst + 1))
            .withHeight(cellH * static_cast<float>(LaunchpadGrid::kInstrumentRowCount));
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

    g.setFont(juce::Font(juce::jlimit(14.0f, 22.0f * textScale, labelArea.getWidth() * 0.55f),
                         juce::Font::bold));
    g.drawFittedText(text, labelArea.toNearestInt(), juce::Justification::centred, 1);
}

auto ActiveInstrumentsView::paint(juce::Graphics &g) -> void {
    g.fillAll(UiConstants::pianoRollBackground);

    // The bracket names the side the pads control; that side is framed below.
    paintGroupBracket(g, instrumentLabelBounds,
                      controlsUserSide ? UiConstants::workspaceActiveInstrumentsInputGroupLabel
                                       : UiConstants::workspaceActiveInstrumentsModelGroupLabel);

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
                // Larger text fills more of the cell (drawFittedText still keeps it inside).
                const float fill = std::min(textScale, 1.7f);
                const auto fontHeight = juce::jlimit(
                    14.0f, 30.0f * textScale, juce::jmin(cellBounds.getHeight() * 0.42f * fill,
                                                         cellBounds.getWidth() * 0.30f * fill));
                g.setFont(juce::Font(fontHeight, juce::Font::bold));
                g.drawFittedText(cell.label,
                                 cellBounds.toNearestInt().reduced(2),
                                 juce::Justification::centred,
                                 2,
                                 0.6f);
            }

            if (showKeys && cell.key.isNotEmpty()) {
                // Key badge in the top-left corner, readable on lit and unlit buttons.
                const float badgeH = juce::jlimit(11.0f, 20.0f, cellBounds.getHeight() * 0.32f);
                g.setFont(juce::Font(badgeH * 0.82f, juce::Font::bold));
                const float badgeW = g.getCurrentFont().getStringWidthFloat(cell.key) + badgeH * 0.5f;
                const auto badge = juce::Rectangle<float>(cellBounds.getX() + 2.0f, cellBounds.getY() + 2.0f,
                                                          badgeW, badgeH);
                g.setColour(juce::Colours::black.withAlpha(0.72f));
                g.fillRoundedRectangle(badge, 3.0f);
                g.setColour(juce::Colour(0xff4fc3f7));
                g.drawText(cell.key, badge, juce::Justification::centred, false);
            }
        }
    }

    // Instrument rows: a divider between your side and the model's, and a frame around the
    // side the pads control in this mode.
    const auto instrumentTop = gridBounds.getY() + cellH * static_cast<float>(LaunchpadGrid::kInstrumentRowFirst + 1);
    const auto instrumentHeight = cellH * static_cast<float>(LaunchpadGrid::kInstrumentRowCount);
    const auto sideWidth = cellW * static_cast<float>(LaunchpadGrid::kColsPerSide);
    const auto dividerX = gridBounds.getX() + sideWidth;
    g.setColour(UiConstants::workspaceWidgetTitleTextColour.withAlpha(0.35f));
    g.drawLine(dividerX, instrumentTop, dividerX, instrumentTop + instrumentHeight, 1.5f);
    const auto frame = juce::Rectangle<float>(controlsUserSide ? gridBounds.getX() : dividerX, instrumentTop,
                                              sideWidth, instrumentHeight)
                           .reduced(0.5f);
    g.setColour(juce::Colours::white.withAlpha(0.9f));
    g.drawRoundedRectangle(frame, 4.0f, 2.5f);
}
