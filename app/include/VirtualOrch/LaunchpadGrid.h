#pragma once

#include <JuceHeader.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "Fifo.h"
#include "VirtualOrch/LaunchpadLighting.h"
#include "VirtualOrch/LaunchpadProgrammerMap.h"
#include "VirtualOrch/OrchestrationTransformer.h"
#include "VirtualOrch/PerformanceActivity.h"
#include "VirtualOrch/PolyphonyCap.h"
#include "VirtualOrch/TemporalDensity.h"

/** Launchpad pad state + instrument updates.
 *
 *  Grid rows (row 0 = top of the 8x8):
 *    kModeRow (0)        Edit (col 0) / Jam (col 1) / Accompaniment (col 2); the selected
 *                        one is lit
 *    kTdRow (1)          TD levels 1..32 (cols 0..5); side = AUTO_TD toggle (starts on).
 *                        AUTO on: side green, levels unlit. AUTO off: side red, the
 *                        selected level lit. A level press turns AUTO off and selects it.
 *                        Only this class writes TemporalDensity.
 *    kUserRowFirst (2)   input (User) instruments, 2 rows; side of row 2 = flush:
 *                        every input instrument that is on is switched on in the model
 *                        set (added, not replaced), then all input pads switch off.
 *    kModelRowFirst (4)  model instruments, 2 rows (same layout as User)
 *    6                   unused
 *    kAccompRow (7)      max polyphony 1..4 (cols 0..3); only works and lit in
 *                        Accompaniment mode (the choice is kept across modes)
 *  Instrument pads: On (1) / Off (0), toggled on press; unmapped pads ignore presses.
 *  Instrument pad press → one InstrumentUpdate for that pad.
 *  Presses within kBurstMs on the same pad are ignored.
 *  LEDs: instrument pads lit in their family colour when on; unmapped pads stay off.
 *  Top-row CCs use `scene_callback(idx)` with idx = cc - 104 (0..7).
 *  App wiring: 1 (CC105) RT pause, 2 (CC106) OT pause; 0 and 3..7 unused.
 *  Pause buttons light red via setTopLed while paused.
 */
class LaunchpadGrid {
public:
    static constexpr int kRows = 8;
    static constexpr int kCols = 8;
    /** Right-hand side column (MIDI notes 19,29,...,89). */
    static constexpr int kSideCol = 9;
    static constexpr int kTopCcBase = 104;
    static constexpr int kTopCcCount = 8;
    static constexpr uint32_t kBurstMs = 100;

    static constexpr int kModeRow = 0;
    static constexpr int kTdRow = 1;
    static constexpr int kUserRowFirst = 2;
    static constexpr int kModelRowFirst = 4;
    static constexpr int kInstrumentRowsPerSet = 2;
    static constexpr int kEditCol = 0;
    static constexpr int kJamCol = 1;
    static constexpr int kAccompanimentCol = 2;
    static constexpr int kFlushRow = kUserRowFirst;
    static constexpr int kTdLevelCount = static_cast<int>(TemporalDensity::kLevels.size());
    /** Bottom row: pads 0..3 pick the max polyphony 1..4 (Accompaniment mode only). */
    static constexpr int kAccompRow = kRows - 1;
    static constexpr int kPolyphonyPadCount = PolyphonyCap::kMaxCap;

    /** Max polyphony changed (the app applies the cap). */
    using AccompanimentCallback = std::function<void()>;
    using GridCallback = std::function<void(int rowIdx, int colIdx, bool on)>;
    using SceneCallback = std::function<void(int idx)>;
    using ModeCallback = std::function<void(OrchestrationMode mode)>;
    /** Programmer-mode static LED: Note On ch1, velocity = palette colour. */
    using LedCallback = std::function<void(int midiNote, uint8_t paletteColour)>;

    GridCallback grid_callback;
    AccompanimentCallback accompaniment_callback;
    /** Every accepted user action on the grid (the console widget); MIDI thread. */
    std::function<void(const PerformanceActivity::LaunchpadAction &)> action_callback;
    SceneCallback scene_callback;
    /** Mode pad pressed; the app applies the mode and calls setSelectedMode. */
    ModeCallback mode_callback;
    LedCallback set_led;

    /** (row, col) → local instrument id; leave empty slots unmapped. */
    std::array<std::array<std::optional<int32_t>, kCols>, kRows> padInstruments{};

    CircularFifo<InstrumentUpdate> *instrumentUpdates = nullptr;

    /**
     * Invoked synchronously from emitPadUpdate (MIDI thread) so OT allowed sets update
     * without waiting for the next orchestration loop / ONNX return.
     */
    std::function<void(const InstrumentUpdate &)> onInstrumentUpdate;

    [[nodiscard]] auto getGridState(int rowIdx, int colIdx) const -> bool {
        if (! isGridPad(rowIdx, colIdx))
            return false;
        return gridState[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
    }

    [[nodiscard]] auto getEffectiveState(int rowIdx, int colIdx) const -> uint8_t {
        if (! isGridPad(rowIdx, colIdx))
            return 0;
        return static_cast<uint8_t>(
            gridState[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)] ? 1 : 0);
    }

    [[nodiscard]] auto isInstrumentPad(int rowIdx, int colIdx) const -> bool {
        return isGridPad(rowIdx, colIdx)
               && padInstruments[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)]
                      .has_value();
    }

    /** User for the input rows, Model for the model rows, nullopt otherwise. */
    [[nodiscard]] static auto targetForRow(int rowIdx) -> std::optional<InstrumentUpdateTarget> {
        if (rowIdx >= kUserRowFirst && rowIdx < kUserRowFirst + kInstrumentRowsPerSet)
            return InstrumentUpdateTarget::User;
        if (rowIdx >= kModelRowFirst && rowIdx < kModelRowFirst + kInstrumentRowsPerSet)
            return InstrumentUpdateTarget::Model;
        return std::nullopt;
    }

    auto refreshPadLed(int rowIdx, int colIdx) -> void {
        if (! set_led || ! isGridPad(rowIdx, colIdx))
            return;
        const auto note = LaunchpadProgrammerMap::noteFromPad(rowIdx, colIdx);
        if (! note.has_value())
            return;

        if (const auto mode = modeForPad(rowIdx, colIdx)) {
            set_led(*note, LaunchpadLighting::colourForModePad(*mode == getSelectedMode()));
            return;
        }

        if (const auto level = tdLevelIndexForPad(rowIdx, colIdx)) {
            const bool selected = ! TemporalDensity::isAuto() && *level == getManualTdLevelIndex();
            set_led(*note, LaunchpadLighting::colourForTdLevel(*level, selected));
            return;
        }

        if (const auto polyphony = polyphonyForPad(rowIdx, colIdx)) {
            set_led(*note, LaunchpadLighting::colourForPolyphonyPad(
                               isAccompanimentActive(), *polyphony == getMaxPolyphony()));
            return;
        }

        const auto &mapped =
            padInstruments[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
        const auto colour = mapped.has_value()
                                ? LaunchpadLighting::colourForInstrumentPad(
                                      *mapped, getEffectiveState(rowIdx, colIdx) != 0)
                                : LaunchpadLighting::kOff;
        set_led(*note, colour);
    }

    auto refreshTopLed(int idx) -> void {
        if (! set_led || idx < 0 || idx >= kTopCcCount)
            return;
        set_led(kTopCcBase + idx, topLedColours[static_cast<size_t>(idx)]);
    }

    /** Mode shown on the Edit/Jam pads; does not change the orchestration mode itself. */
    auto setSelectedMode(OrchestrationMode mode) -> void {
        selectedMode.store(mode, std::memory_order_relaxed);
        refreshPadLed(kModeRow, kEditCol);
        refreshPadLed(kModeRow, kJamCol);
        refreshPadLed(kModeRow, kAccompanimentCol);
        refreshAccompanimentLeds();
    }

    [[nodiscard]] auto getSelectedMode() const -> OrchestrationMode {
        return selectedMode.load(std::memory_order_relaxed);
    }

    [[nodiscard]] static auto modeForPad(int rowIdx, int colIdx) -> std::optional<OrchestrationMode> {
        if (rowIdx != kModeRow)
            return std::nullopt;
        if (colIdx == kEditCol)
            return OrchestrationMode::Edit;
        if (colIdx == kJamCol)
            return OrchestrationMode::Jam;
        if (colIdx == kAccompanimentCol)
            return OrchestrationMode::Accompaniment;
        return std::nullopt;
    }

    /** Max polyphony 1..4 for the first pads of the bottom row. */
    [[nodiscard]] static auto polyphonyForPad(int rowIdx, int colIdx) -> std::optional<int> {
        if (rowIdx != kAccompRow || colIdx < 0 || colIdx >= kPolyphonyPadCount)
            return std::nullopt;
        return colIdx + PolyphonyCap::kMinCap;
    }

    [[nodiscard]] auto isAccompanimentActive() const -> bool {
        return getSelectedMode() == OrchestrationMode::Accompaniment;
    }
    [[nodiscard]] auto getMaxPolyphony() const -> int {
        return maxPolyphony.load(std::memory_order_relaxed);
    }

    /** Cap the reduction transformers must apply now: 0 unless in Accompaniment mode. */
    [[nodiscard]] auto effectivePolyphonyCap() const -> int {
        return isAccompanimentActive() ? getMaxPolyphony() : 0;
    }

    auto setMaxPolyphony(int cap) -> void {
        maxPolyphony.store(std::clamp(cap, PolyphonyCap::kMinCap, PolyphonyCap::kMaxCap),
                           std::memory_order_relaxed);
        refreshAccompanimentLeds();
    }

    [[nodiscard]] static auto tdLevelIndexForPad(int rowIdx, int colIdx) -> std::optional<int> {
        if (rowIdx != kTdRow || colIdx < 0 || colIdx >= kTdLevelCount)
            return std::nullopt;
        return colIdx;
    }

    /** Level used when AUTO_TD is off (index into TemporalDensity::kLevels). */
    [[nodiscard]] auto getManualTdLevelIndex() const -> int {
        return manualTdLevelIndex.load(std::memory_order_relaxed);
    }

    auto refreshSideLed(int rowIdx) -> void {
        if (! set_led || rowIdx < 0 || rowIdx >= kRows)
            return;
        const auto note = LaunchpadProgrammerMap::noteFromPad(rowIdx, kSideCol);
        if (! note.has_value())
            return;
        auto colour = LaunchpadLighting::kOff;
        if (rowIdx == kTdRow)
            colour = LaunchpadLighting::colourForAutoTd(TemporalDensity::isAuto());
        else if (rowIdx == kFlushRow)
            colour = LaunchpadLighting::kFlush;
        set_led(*note, colour);
    }

    /** Persist and light a top-row button (CC 104+idx). */
    auto setTopLed(int idx, uint8_t paletteColour) -> void {
        if (idx < 0 || idx >= kTopCcCount)
            return;
        topLedColours[static_cast<size_t>(idx)] = paletteColour;
        refreshTopLed(idx);
    }

    [[nodiscard]] auto getTopLedColour(int idx) const -> uint8_t {
        if (idx < 0 || idx >= kTopCcCount)
            return LaunchpadLighting::kOff;
        return topLedColours[static_cast<size_t>(idx)];
    }

    auto refreshAllLeds() -> void {
        for (int row = 0; row < kRows; ++row) {
            refreshSideLed(row);
            for (int col = 0; col < kCols; ++col)
                refreshPadLed(row, col);
        }
        for (int idx = 0; idx < kTopCcCount; ++idx)
            refreshTopLed(idx);
    }

    /** Set pad on/off (not a toggle). Emits InstrumentUpdate and refreshes LED. */
    auto setPadOn(int rowIdx, int colIdx, bool on) -> void {
        if (! isGridPad(rowIdx, colIdx))
            return;

        auto &state = gridState[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
        if (state == on)
            return;
        state = on;

        emitPadUpdate(rowIdx, colIdx);
        refreshPadLed(rowIdx, colIdx);
        if (grid_callback)
            grid_callback(rowIdx, colIdx, on);
    }

    /** MIDI press/release. Only presses toggle/emit. Optional nowMs for tests. */
    auto handleInput(int rowIdx, int colIdx, bool pressed,
                     std::optional<uint32_t> nowMs = std::nullopt) -> void {
        if (! pressed)
            return;

        const uint32_t now = nowMs.value_or(juce::Time::getMillisecondCounter());

        if (const auto mode = modeForPad(rowIdx, colIdx)) {
            auto &lastMs = gridLastAcceptMs[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
            if (acceptPress(lastMs, now) && mode_callback)
                mode_callback(*mode);
            return;
        }

        if (const auto level = tdLevelIndexForPad(rowIdx, colIdx)) {
            auto &lastMs = gridLastAcceptMs[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
            if (acceptPress(lastMs, now)) {
                selectTdLevel(*level);
                emitAction({PerformanceActivity::LaunchpadAction::Kind::TdLevel, 0,
                            TemporalDensity::kLevels[static_cast<size_t>(*level)], false, false});
            }
            return;
        }

        if (colIdx == kSideCol && rowIdx == kTdRow) {
            auto &lastMs = sideLastAcceptMs[static_cast<size_t>(rowIdx)];
            if (acceptPress(lastMs, now)) {
                setTdAuto(! TemporalDensity::isAuto());
                emitAction({PerformanceActivity::LaunchpadAction::Kind::TdAuto, 0, 0,
                            TemporalDensity::isAuto(), false});
            }
            return;
        }

        if (const auto polyphony = polyphonyForPad(rowIdx, colIdx)) {
            auto &lastMs = gridLastAcceptMs[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
            if (acceptPress(lastMs, now) && isAccompanimentActive()) {
                setMaxPolyphony(*polyphony);
                emitAction({PerformanceActivity::LaunchpadAction::Kind::Polyphony, 0, *polyphony,
                            false, false});
                if (accompaniment_callback)
                    accompaniment_callback();
            }
            return;
        }

        if (colIdx == kSideCol && rowIdx == kFlushRow) {
            auto &lastMs = sideLastAcceptMs[static_cast<size_t>(rowIdx)];
            if (acceptPress(lastMs, now)) {
                flushInputToModel();
                emitAction({PerformanceActivity::LaunchpadAction::Kind::Flush, 0, 0, false, false});
            }
            return;
        }

        if (isInstrumentPad(rowIdx, colIdx)) {
            auto &lastMs = gridLastAcceptMs[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
            if (! acceptPress(lastMs, now))
                return;

            auto &on = gridState[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
            on = ! on;

            emitPadUpdate(rowIdx, colIdx);
            refreshPadLed(rowIdx, colIdx);
            emitAction({PerformanceActivity::LaunchpadAction::Kind::InstrumentPad,
                        *padInstruments[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)],
                        0, on, targetForRow(rowIdx) == InstrumentUpdateTarget::Model});
            if (grid_callback)
                grid_callback(rowIdx, colIdx, on);
        }
    }

    /** Top-row CC press: scene_callback(idx), idx in 0..7 (cc - 104). */
    auto handleSceneInput(int idx) -> void {
        if (idx < 0 || idx >= kTopCcCount)
            return;

        if (scene_callback)
            scene_callback(idx);
    }

private:
    auto selectTdLevel(int levelIndex) -> void {
        manualTdLevelIndex.store(levelIndex, std::memory_order_relaxed);
        TemporalDensity::set(TemporalDensity::kLevels[static_cast<size_t>(levelIndex)]);
        refreshTdLeds();
    }

    auto setTdAuto(bool on) -> void {
        TemporalDensity::set(on ? TemporalDensity::kAuto
                                : TemporalDensity::kLevels[static_cast<size_t>(getManualTdLevelIndex())]);
        refreshTdLeds();
    }

    [[nodiscard]] auto findPadForInstrument(int firstRow, int32_t localId) const
        -> std::optional<std::pair<int, int>> {
        for (int row = firstRow; row < firstRow + kInstrumentRowsPerSet; ++row)
            for (int col = 0; col < kCols; ++col)
                if (padInstruments[static_cast<size_t>(row)][static_cast<size_t>(col)] == localId)
                    return std::pair{row, col};
        return std::nullopt;
    }

    /** Model pads are switched on before input pads switch off, so no instrument drops out. */
    auto flushInputToModel() -> void {
        std::vector<std::pair<int, int>> flushed;
        for (int row = kUserRowFirst; row < kUserRowFirst + kInstrumentRowsPerSet; ++row) {
            for (int col = 0; col < kCols; ++col) {
                if (! isInstrumentPad(row, col) || getEffectiveState(row, col) == 0)
                    continue;
                const auto localId =
                    *padInstruments[static_cast<size_t>(row)][static_cast<size_t>(col)];
                if (const auto modelPad = findPadForInstrument(kModelRowFirst, localId))
                    setPadOn(modelPad->first, modelPad->second, true);
                flushed.emplace_back(row, col);
            }
        }
        for (const auto &[row, col]: flushed)
            setPadOn(row, col, false);
    }

    auto emitAction(const PerformanceActivity::LaunchpadAction &action) -> void {
        if (action_callback)
            action_callback(action);
    }

    auto refreshAccompanimentLeds() -> void {
        for (int col = 0; col < kPolyphonyPadCount; ++col)
            refreshPadLed(kAccompRow, col);
        refreshSideLed(kAccompRow);
    }

    auto refreshTdLeds() -> void {
        for (int col = 0; col < kTdLevelCount; ++col)
            refreshPadLed(kTdRow, col);
        refreshSideLed(kTdRow);
    }

    static auto isGridPad(int row, int col) -> bool {
        return row >= 0 && row < kRows && col >= 0 && col < kCols;
    }

    /** lastMs == 0 means never accepted; updates lastMs on accept. */
    static auto acceptPress(uint32_t &lastMs, uint32_t now) -> bool {
        if (lastMs != 0 && now - lastMs < kBurstMs)
            return false;
        lastMs = now == 0 ? 1 : now;
        return true;
    }

    auto emitPadUpdate(int rowIdx, int colIdx) -> void {
        if (! isGridPad(rowIdx, colIdx))
            return;

        const auto &instrument = padInstruments[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
        const auto target = targetForRow(rowIdx);
        if (! instrument.has_value() || ! target.has_value())
            return;

        InstrumentUpdate update;
        update.localInstrumentId = *instrument;
        update.target = *target;
        update.state = getEffectiveState(rowIdx, colIdx);
        if (instrumentUpdates != nullptr)
            instrumentUpdates->push(update);
        if (onInstrumentUpdate)
            onInstrumentUpdate(update);
    }

    std::array<std::array<bool, kCols>, kRows> gridState{};
    std::atomic<OrchestrationMode> selectedMode{OrchestrationMode::Jam};

    std::atomic<int> manualTdLevelIndex{0};
    std::atomic<int> maxPolyphony{PolyphonyCap::kMinCap};

    std::array<std::array<uint32_t, kCols>, kRows> gridLastAcceptMs{};
    std::array<uint32_t, kRows> sideLastAcceptMs{};
    std::array<uint8_t, kTopCcCount> topLedColours{};
};
