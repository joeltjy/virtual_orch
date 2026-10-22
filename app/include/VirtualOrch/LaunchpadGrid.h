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
#include "VirtualOrch/RegisterRange.h"
#include "VirtualOrch/TemporalDensity.h"

/** Launchpad pad state + instrument updates.
 *
 *  Grid rows (row 0 = top of the 8x8):
 *    kModeRow (0)        Edit (col 0) / Jam (col 1) / Accompaniment (col 2); the selected
 *                        one is lit
 *    kTdRow (1)          TD levels 1..64 (cols 0..6); side = AUTO_TD toggle (starts off, at TD 4).
 *                        AUTO on: side green, levels unlit. AUTO off: side red, the
 *                        selected level lit. A level press turns AUTO off and selects it.
 *                        Only this class writes TemporalDensity.
 *    kUserRowFirst (2)   input (User) instruments, 2 rows; side of row 2 = flush:
 *                        every input instrument that is on is switched on in the model
 *                        set (added, not replaced), then all input pads switch off.
 *    kModelRowFirst (4)  model instruments, 2 rows (same layout as User)
 *    kRegisterRow (6)    register LOW / MEDIUM / HIGH (cols 0..2) + AUTO side pad; a register
 *                        other than AUTO switches TD to AUTO (TD 4 for the model)
 *    kAccompRow (7)      max polyphony 1..4 (cols 0..3); works and is lit in Jam and
 *                        Accompaniment; pressing the selected pad again clears the cap.
 *                        Entering Accompaniment with no cap starts at 1; kept across modes.
 *  Instrument pads: On (1) / Off (0), toggled on press; unmapped pads ignore presses.
 *  Instrument pad press → one InstrumentUpdate for that pad.
 *  Presses within kBurstMs on the same pad are ignored.
 *  LEDs: instrument pads lit in their family colour when on; unmapped pads stay off.
 *  Top-row CCs use `scene_callback(idx)` with idx = cc - 104 (0..7).
 *  App wiring: 0 (CC104) force restart (stop + start), 1 (CC105) RT pause, 2 (CC106) OT pause,
 *  3 (CC107) block the user's input from the reduction (ReductionInputGate); 4..7 unused.
 *  Row 6: register LOW / MEDIUM / HIGH (pads 0-2) and an AUTO side pad (RegisterRange).
 *  Pause and input-block buttons light red via setTopLed while on.
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
    /** Row 6: pads 0..2 = register LOW / MEDIUM / HIGH, the side pad = AUTO register. */
    static constexpr int kRegisterRow = kRows - 2;
    static constexpr int kRegisterPadCount = 3;
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
    /** A TD level pad or AUTO_TD was pressed (after TemporalDensity changed); the app cuts the future. */
    std::function<void()> td_callback;
    /** A register pad or AUTO was pressed (after RegisterRange changed); the app cuts the future. */
    std::function<void()> register_callback;
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

        if (const auto reg = registerForPad(rowIdx, colIdx)) {
            set_led(*note, LaunchpadLighting::colourForRegister(
                               *reg, ! RegisterRange::isAuto() && *reg == RegisterRange::get()));
            return;
        }

        if (const auto polyphony = polyphonyForPad(rowIdx, colIdx)) {
            set_led(*note, LaunchpadLighting::colourForPolyphonyPad(
                               isPolyphonyActive(), *polyphony == getMaxPolyphony()));
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
        // Accompaniment starts at polyphony 1 unless a cap was already chosen (e.g. in Jam).
        if (mode == OrchestrationMode::Accompaniment && getMaxPolyphony() == 0)
            maxPolyphony.store(PolyphonyCap::kMinCap, std::memory_order_relaxed);
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
    /** The polyphony pads work in Jam and Accompaniment mode. */
    [[nodiscard]] auto isPolyphonyActive() const -> bool {
        return isJamLike(getSelectedMode());
    }
    [[nodiscard]] auto getMaxPolyphony() const -> int {
        return maxPolyphony.load(std::memory_order_relaxed);
    }

    /**
     * Cap the reduction transformers must apply now: the selected max polyphony in Jam or
     * Accompaniment mode, 0 = no cap (nothing selected, or Edit mode).
     */
    [[nodiscard]] auto effectivePolyphonyCap() const -> int {
        return isPolyphonyActive() ? getMaxPolyphony() : 0;
    }

    auto setMaxPolyphony(int cap) -> void {
        maxPolyphony.store(std::clamp(cap, 0, PolyphonyCap::kMaxCap), std::memory_order_relaxed);
        refreshAccompanimentLeds();
    }

    /** Register value (RegisterRange::kLow..kHigh) of the first pads of the register row. */
    [[nodiscard]] static auto registerForPad(int rowIdx, int colIdx) -> std::optional<int> {
        if (rowIdx != kRegisterRow || colIdx < 0 || colIdx >= kRegisterPadCount)
            return std::nullopt;
        return RegisterRange::kLow + colIdx;
    }

    /** Register used when AUTO is off (the last one selected; MEDIUM until one is). */
    [[nodiscard]] auto getManualRegister() const -> int {
        return manualRegister.load(std::memory_order_relaxed);
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
        else if (rowIdx == kRegisterRow)
            colour = LaunchpadLighting::colourForAutoTd(RegisterRange::isAuto());
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
                if (td_callback)
                    td_callback();
            }
            return;
        }

        if (colIdx == kSideCol && rowIdx == kTdRow) {
            auto &lastMs = sideLastAcceptMs[static_cast<size_t>(rowIdx)];
            if (acceptPress(lastMs, now)) {
                setTdAuto(! TemporalDensity::isAuto());
                emitAction({PerformanceActivity::LaunchpadAction::Kind::TdAuto, 0, 0,
                            TemporalDensity::isAuto(), false});
                if (td_callback)
                    td_callback();
            }
            return;
        }

        if (const auto reg = registerForPad(rowIdx, colIdx)) {
            auto &lastMs = gridLastAcceptMs[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
            if (acceptPress(lastMs, now)) {
                selectRegister(*reg);
                emitAction({PerformanceActivity::LaunchpadAction::Kind::Register, 0, *reg, false, false});
                if (register_callback)
                    register_callback();
                tdAutoForRegister();
            }
            return;
        }

        if (colIdx == kSideCol && rowIdx == kRegisterRow) {
            auto &lastMs = sideLastAcceptMs[static_cast<size_t>(rowIdx)];
            if (acceptPress(lastMs, now)) {
                setRegisterAuto(! RegisterRange::isAuto());
                emitAction({PerformanceActivity::LaunchpadAction::Kind::RegisterAuto, 0, 0,
                            RegisterRange::isAuto(), false});
                if (register_callback)
                    register_callback();
                tdAutoForRegister();
            }
            return;
        }

        if (const auto polyphony = polyphonyForPad(rowIdx, colIdx)) {
            auto &lastMs = gridLastAcceptMs[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
            if (acceptPress(lastMs, now) && isPolyphonyActive()) {
                // pressing the selected pad again clears the cap
                const int cap = *polyphony == getMaxPolyphony() ? 0 : *polyphony;
                setMaxPolyphony(cap);
                emitAction({PerformanceActivity::LaunchpadAction::Kind::Polyphony, 0, cap, false,
                            false});
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

    auto selectRegister(int reg) -> void {
        manualRegister.store(reg, std::memory_order_relaxed);
        RegisterRange::set(reg);
        refreshRegisterLeds();
    }

    auto setRegisterAuto(bool on) -> void {
        RegisterRange::set(on ? RegisterRange::kAuto : getManualRegister());
        refreshRegisterLeds();
    }

    /** A register other than AUTO switches TD to AUTO (a TD level can still be picked after). */
    auto tdAutoForRegister() -> void {
        if (RegisterRange::isAuto() || TemporalDensity::isAuto())
            return;
        setTdAuto(true);
        emitAction({PerformanceActivity::LaunchpadAction::Kind::TdAuto, 0, 0, true, false});
        if (td_callback)
            td_callback();
    }

    auto refreshRegisterLeds() -> void {
        for (int col = 0; col < kRegisterPadCount; ++col)
            refreshPadLed(kRegisterRow, col);
        refreshSideLed(kRegisterRow);
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

    std::atomic<int> manualTdLevelIndex{2}; // TD 4, the default (TemporalDensity::kDefault)
    std::atomic<int> manualRegister{RegisterRange::kMedium};
    std::atomic<int> maxPolyphony{0}; // 0 = no cap

    std::array<std::array<uint32_t, kCols>, kRows> gridLastAcceptMs{};
    std::array<uint32_t, kRows> sideLastAcceptMs{};
    std::array<uint8_t, kTopCcCount> topLedColours{};
};
