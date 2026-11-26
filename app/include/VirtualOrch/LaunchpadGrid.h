#pragma once

#include <JuceHeader.h>
#include "VirtualOrch/AppTime.h"
#include "VirtualOrch/OpLog.h"

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
#include "VirtualOrch/MoodPresets.h"
#include "VirtualOrch/RegisterMix.h"
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
 *    kInstrumentRowFirst (2..5)  one row per family (strings, woodwinds, brass, percussion);
 *                        YOUR instruments in cols 0..3, the MODEL's in cols 4..7 (same layout).
 *                        The pads control your side in Edit and the model's side in Jam /
 *                        Accompaniment; presses on the other side are ignored. Controlled
 *                        side: family colour. Other side: yours grey, the model's dim.
 *                        Side of row 2 = FLUSH (Edit only): your instruments are added to the
 *                        model's (yours stay on). Side of row 3 = RESET (Edit only): every
 *                        model instrument switches off.
 *    kRegisterRow (6)    register LOW / MEDIUM / HIGH (cols 0..2) + AUTO side pad; a register
 *                        other than AUTO switches TD to AUTO (TD 4 for the model).
 *                        Same row, col 4 (kAccompRow): MONOPHONIC, max polyphony 1 (cols 5..7 are
 *                        free; caps 2-4 were dropped Nov 24 2026); works and is lit in every mode
 *                        (Step-in too); pressing it again clears the cap (unconstrained).
 *                        Entering Accompaniment leaves it as it is; kept across modes.
 *    kPresetRow (7)      mood presets (MoodPresets), cols 0..7: a press sets the controlled
 *                        side's instruments, TD and register. Active preset bright, changed
 *                        since dimmer, others dimmest; pressing it again re-applies it.
 *  Instrument pads: On (1) / Off (0), toggled on press; unmapped pads ignore presses.
 *  Instrument pad press → one InstrumentUpdate for that pad.
 *  Presses within kBurstMs on the same pad are ignored.
 *  Top-row CCs use `scene_callback(idx)` with idx = cc - 104 (0..7).
 *  App wiring: 0 (CC104) force restart (stop + start), 1 (CC105) RT pause, 2 (CC106) OT pause,
 *  3 (CC107) block the user's input from the reduction (ReductionInputGate); 4..7 unused.
 *  Pause and input-block buttons light red via setTopLed while on.
 */
class LaunchpadGrid {
public:
    /**
     * Which pad layout this is, written to the operation log (SESSION_START launchpad_layout) so
     * PAD_PRESS row / col can be read without knowing the date. 1 = the layout before Nov 2026
     * (input / model instrument rows); 2 = two-sided (your instruments cols 0-3, the model's 4-7,
     * presets on row 7). See NEW_LAUNCHPAD_LAYOUT.txt.
     */
    static constexpr int kLayoutVersion = 2;
    static constexpr int kRows = 8;
    static constexpr int kCols = 8;
    /** Right-hand side column (MIDI notes 19,29,...,89). */
    static constexpr int kSideCol = 9;
    static constexpr int kTopCcBase = 104;
    static constexpr int kTopCcCount = 8;
    static constexpr uint32_t kBurstMs = 100;

    static constexpr int kModeRow = 0;
    static constexpr int kTdRow = 1;
    /** Instrument rows: strings, woodwinds, brass, percussion. */
    static constexpr int kInstrumentRowFirst = 2;
    static constexpr int kInstrumentRowCount = 4;
    /** Each side's columns: yours 0..3, the model's 4..7. */
    static constexpr int kColsPerSide = 4;
    static constexpr int kUserColFirst = 0;
    static constexpr int kModelColFirst = kColsPerSide;
    static constexpr int kEditCol = 0;
    static constexpr int kJamCol = 1;
    static constexpr int kAccompanimentCol = 2;
    static constexpr int kFlushRow = kInstrumentRowFirst;
    static constexpr int kResetRow = kInstrumentRowFirst + 1;
    static constexpr int kTdLevelCount = static_cast<int>(TemporalDensity::kLevels.size());
    /** Row 6: pads 0..2 = register LOW / MEDIUM / HIGH, the side pad = AUTO register. */
    static constexpr int kRegisterRow = 6;
    static constexpr int kRegisterPadCount = 3;
    /** Row 6, col 4: monophonic (max polyphony 1) on / off, in every mode. */
    static constexpr int kAccompRow = kRegisterRow;
    static constexpr int kPolyphonyColFirst = 4;
    static constexpr int kPolyphonyPadCount = 1;
    /** Bottom row: the mood presets. */
    static constexpr int kPresetRow = 7;
    static constexpr int kPresetCount = MoodPresets::kCount;

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
    /**
     * A mood preset was applied (index into MoodPresets::all(), after instruments, TD and
     * register changed); the app cuts the future once, as for TD / register.
     */
    std::function<void(int presetIndex)> preset_callback;
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

    [[nodiscard]] static auto isInstrumentRow(int rowIdx) -> bool {
        return rowIdx >= kInstrumentRowFirst && rowIdx < kInstrumentRowFirst + kInstrumentRowCount;
    }

    /** User for cols 0..3 of the instrument rows, Model for cols 4..7, nullopt otherwise. */
    [[nodiscard]] static auto targetForPad(int rowIdx, int colIdx) -> std::optional<InstrumentUpdateTarget> {
        if (! isInstrumentRow(rowIdx) || colIdx < 0 || colIdx >= kCols)
            return std::nullopt;
        return colIdx < kModelColFirst ? InstrumentUpdateTarget::User : InstrumentUpdateTarget::Model;
    }

    /** The side the instrument pads control: yours in Edit, the model's in Jam / Accompaniment. */
    [[nodiscard]] auto controlledTarget() const -> InstrumentUpdateTarget {
        return getSelectedMode() == OrchestrationMode::Edit ? InstrumentUpdateTarget::User
                                                            : InstrumentUpdateTarget::Model;
    }

    /**
     * Whether presses on this side's instrument pads do anything: both sides in Edit
     * (Step-in), only the model's in Jam / Accompaniment.
     */
    [[nodiscard]] auto isSidePressable(InstrumentUpdateTarget side) const -> bool {
        return layout1.load(std::memory_order_relaxed) || getSelectedMode() == OrchestrationMode::Edit
               || side == controlledTarget();
    }

    /**
     * Log replay of sessions recorded with layout 1 (kLayoutVersion): presses are read as layout-1
     * pads (layout1Pad) and behave as they did then: both sides respond in every mode, and FLUSH
     * works in every mode and switches your pads off afterwards. Never set in the live app.
     */
    std::atomic<bool> layout1{false};

    /**
     * The layout-2 pad with the meaning a layout-1 pad had, or nullopt for a pad that did nothing
     * then. Layout 1: row 2 = your StrEns Pizz Tremolo - Flute Oboe Clarinet Bassoon, row 3 = your
     * Horn Trumpet Trombone Tuba Timpani, rows 4-5 = the same for the model, row 7 cols 0-3 = max
     * polyphony 1-4; modes, TD, register and the side buttons of rows 1, 2 and 6 are unchanged.
     */
    [[nodiscard]] static auto layout1Pad(int rowIdx, int colIdx) -> std::optional<std::pair<int, int>> {
        if (colIdx == kSideCol)
            return rowIdx == kTdRow || rowIdx == kFlushRow || rowIdx == kRegisterRow
                       ? std::optional<std::pair<int, int>>(std::pair{rowIdx, colIdx})
                       : std::nullopt;
        if (rowIdx >= 2 && rowIdx <= 5) {
            const int sideCol = rowIdx >= 4 ? kModelColFirst : kUserColFirst;
            const bool firstRow = rowIdx == 2 || rowIdx == 4;
            if (firstRow && colIdx >= 0 && colIdx <= 2)
                return std::pair{kInstrumentRowFirst, sideCol + colIdx};          // strings
            if (firstRow && colIdx >= 4 && colIdx <= 7)
                return std::pair{kInstrumentRowFirst + 1, sideCol + colIdx - 4};  // woodwinds
            if (! firstRow && colIdx >= 0 && colIdx <= 3)
                return std::pair{kInstrumentRowFirst + 2, sideCol + colIdx};      // brass
            if (! firstRow && colIdx == 4)
                return std::pair{kInstrumentRowFirst + 3, sideCol};               // timpani
            return std::nullopt;
        }
        if (rowIdx == 7)
            return colIdx >= 0 && colIdx < kPolyphonyPadCount
                       ? std::optional<std::pair<int, int>>(std::pair{kAccompRow, kPolyphonyColFirst + colIdx})
                       : std::nullopt;
        return std::pair{rowIdx, colIdx}; // modes, TD, register
    }

    /** LED colour of an instrument pad (also used by the on-screen view). */
    [[nodiscard]] auto instrumentPadColour(int rowIdx, int colIdx) const -> uint8_t {
        const auto target = targetForPad(rowIdx, colIdx);
        if (! isInstrumentPad(rowIdx, colIdx) || ! target.has_value())
            return LaunchpadLighting::kOff;
        return LaunchpadLighting::colourForSidedInstrumentPad(
            *padInstruments[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)],
            getEffectiveState(rowIdx, colIdx) != 0, *target == controlledTarget(),
            *target == InstrumentUpdateTarget::User);
    }

    /** Mood preset index of a bottom-row pad. */
    [[nodiscard]] static auto presetForPad(int rowIdx, int colIdx) -> std::optional<int> {
        if (rowIdx != kPresetRow || colIdx < 0 || colIdx >= kPresetCount)
            return std::nullopt;
        return colIdx;
    }

    /** The preset applied last (cleared by a mode change), or -1. */
    [[nodiscard]] auto getActivePreset() const -> int { return activePreset.load(std::memory_order_relaxed); }
    /** Something the preset sets changed by hand since it was applied. */
    [[nodiscard]] auto isActivePresetModified() const -> bool {
        return activePresetModified.load(std::memory_order_relaxed);
    }

    [[nodiscard]] auto presetPadColour(int presetIndex) const -> uint8_t {
        if (presetIndex < 0 || presetIndex >= kPresetCount)
            return LaunchpadLighting::kOff;
        return LaunchpadLighting::colourForPresetPad(
            MoodPresets::all()[static_cast<size_t>(presetIndex)].colour,
            presetIndex == getActivePreset(), isActivePresetModified());
    }

    /** Side button colour (also used by the on-screen view). */
    [[nodiscard]] auto sideColour(int rowIdx) const -> uint8_t {
        const bool edit = getSelectedMode() == OrchestrationMode::Edit;
        if (rowIdx == kTdRow)
            return LaunchpadLighting::colourForAutoTd(TemporalDensity::isAuto());
        if (rowIdx == kFlushRow)
            return LaunchpadLighting::colourForEditOnlyButton(edit, LaunchpadLighting::kFlush);
        if (rowIdx == kResetRow)
            return LaunchpadLighting::colourForEditOnlyButton(edit, LaunchpadLighting::kReset);
        if (rowIdx == kRegisterRow)
            return LaunchpadLighting::colourForAutoTd(RegisterRange::isAuto());
        return LaunchpadLighting::kOff;
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

        if (const auto preset = presetForPad(rowIdx, colIdx)) {
            set_led(*note, presetPadColour(*preset));
            return;
        }

        set_led(*note, instrumentPadColour(rowIdx, colIdx));
    }

    auto refreshTopLed(int idx) -> void {
        if (! set_led || idx < 0 || idx >= kTopCcCount)
            return;
        set_led(kTopCcBase + idx, topLedColours[static_cast<size_t>(idx)]);
    }

    /** Mode shown on the Edit/Jam pads; does not change the orchestration mode itself. */
    auto setSelectedMode(OrchestrationMode mode) -> void {
        selectedMode.store(mode, std::memory_order_relaxed);
        // Accompaniment no longer turns Mono on by itself (Nov 24 2026): the cap is only the Mono pad's.
        refreshPadLed(kModeRow, kEditCol);
        refreshPadLed(kModeRow, kJamCol);
        refreshPadLed(kModeRow, kAccompanimentCol);
        refreshAccompanimentLeds();
        // The pads now control the other side: relight both sides, Flush / Reset; a preset
        // applied to the old side is no longer the active one.
        activePreset.store(-1, std::memory_order_relaxed);
        RegisterMix::Target::set(std::nullopt);
        refreshInstrumentLeds();
        refreshPresetLeds();
        refreshSideLed(kFlushRow);
        refreshSideLed(kResetRow);
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

    /** Max polyphony 1 for col 4 of row 6 (the monophonic toggle). */
    [[nodiscard]] static auto polyphonyForPad(int rowIdx, int colIdx) -> std::optional<int> {
        if (rowIdx != kAccompRow || colIdx < kPolyphonyColFirst
            || colIdx >= kPolyphonyColFirst + kPolyphonyPadCount)
            return std::nullopt;
        return colIdx - kPolyphonyColFirst + PolyphonyCap::kMinCap;
    }

    [[nodiscard]] auto isAccompanimentActive() const -> bool {
        return getSelectedMode() == OrchestrationMode::Accompaniment;
    }
    /** The Mono pad works in every mode (Step-in too, Nov 24 2026). */
    [[nodiscard]] auto isPolyphonyActive() const -> bool { return true; }
    [[nodiscard]] auto getMaxPolyphony() const -> int {
        return maxPolyphony.load(std::memory_order_relaxed);
    }

    /**
     * Cap the reduction transformers must apply now: the selected max polyphony (Mono = 1), in any
     * mode; 0 = no cap.
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
        set_led(*note, sideColour(rowIdx));
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

        const uint32_t now = nowMs.value_or(AppTime::counterMs());
        struct PadScope {
            int64_t previous = OpLog::currentPad;
            ~PadScope() { OpLog::currentPad = previous; }
        } padScope;
        if (OpLog::on()) {
            OpLog::currentPad = OpLog::nextPadId();
            OpLog::emit(OpLog::Src::Midi, "PAD_PRESS",
                        {{"p", OpLog::id('p', OpLog::currentPad)}, {"row", rowIdx}, {"col", colIdx},
                         {"source", OpLog::padKey != nullptr ? "keyboard" : "launchpad"},
                         {"key", OpLog::padKey != nullptr ? OpLog::Value(std::string(OpLog::padKey)) : OpLog::Value()}});
        }
        if (layout1.load(std::memory_order_relaxed)) {
            // PAD_PRESS above keeps the layout-1 pad as pressed; act on its layout-2 equivalent
            const auto pad = layout1Pad(rowIdx, colIdx);
            if (! pad.has_value())
                return;
            rowIdx = pad->first;
            colIdx = pad->second;
        }

        if (const auto mode = modeForPad(rowIdx, colIdx)) {
            auto &lastMs = gridLastAcceptMs[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
            if (acceptPress(lastMs, now) && mode_callback)
                mode_callback(*mode);
            return;
        }

        if (const auto level = tdLevelIndexForPad(rowIdx, colIdx)) {
            auto &lastMs = gridLastAcceptMs[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
            if (acceptPress(lastMs, now)) {
                markPresetModified();
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
                markPresetModified();
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
                markPresetModified();
                RegisterMix::Target::set(std::nullopt); // a register picked by hand replaces the preset's mix
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
                markPresetModified();
                RegisterMix::Target::set(std::nullopt); // a register picked by hand replaces the preset's mix
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
            if (acceptPress(lastMs, now)
                && (getSelectedMode() == OrchestrationMode::Edit || layout1.load(std::memory_order_relaxed))) {
                flushInputToModel();
                emitAction({PerformanceActivity::LaunchpadAction::Kind::Flush, 0, 0, false, false});
            }
            return;
        }

        if (colIdx == kSideCol && rowIdx == kResetRow) {
            auto &lastMs = sideLastAcceptMs[static_cast<size_t>(rowIdx)];
            if (acceptPress(lastMs, now) && getSelectedMode() == OrchestrationMode::Edit) {
                resetModelInstruments();
                emitAction({PerformanceActivity::LaunchpadAction::Kind::Reset, 0, 0, false, false});
            }
            return;
        }

        if (const auto preset = presetForPad(rowIdx, colIdx)) {
            auto &lastMs = gridLastAcceptMs[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
            if (acceptPress(lastMs, now)) {
                applyPreset(*preset);
                emitAction({PerformanceActivity::LaunchpadAction::Kind::Preset, 0, *preset,
                            false, controlledTarget() == InstrumentUpdateTarget::Model});
                if (preset_callback)
                    preset_callback(*preset);
            }
            return;
        }

        if (isInstrumentPad(rowIdx, colIdx)) {
            auto &lastMs = gridLastAcceptMs[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
            // Jam / Accompaniment: only the model's side responds. Edit: both sides.
            const auto side = *targetForPad(rowIdx, colIdx);
            if (! isSidePressable(side))
                return;
            if (! acceptPress(lastMs, now))
                return;
            // Presets set the controlled side, so only a press there changes the preset.
            if (side == controlledTarget())
                markPresetModified();

            auto &on = gridState[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
            on = ! on;

            emitPadUpdate(rowIdx, colIdx);
            refreshPadLed(rowIdx, colIdx);
            emitAction({PerformanceActivity::LaunchpadAction::Kind::InstrumentPad,
                        *padInstruments[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)],
                        0, on, targetForPad(rowIdx, colIdx) == InstrumentUpdateTarget::Model});
            if (grid_callback)
                grid_callback(rowIdx, colIdx, on);
        }
    }

    /** Top-row CC press: scene_callback(idx), idx in 0..7 (cc - 104). */
    auto handleSceneInput(int idx) -> void {
        if (idx < 0 || idx >= kTopCcCount)
            return;
        struct PadScope {
            int64_t previous = OpLog::currentPad;
            ~PadScope() { OpLog::currentPad = previous; }
        } padScope;
        if (OpLog::on()) {
            OpLog::currentPad = OpLog::nextPadId();
            OpLog::emit(OpLog::Src::Midi, "PAD_PRESS",
                        {{"p", OpLog::id('p', OpLog::currentPad)}, {"top", idx},
                         {"source", OpLog::padKey != nullptr ? "keyboard" : "launchpad"},
                         {"key", OpLog::padKey != nullptr ? OpLog::Value(std::string(OpLog::padKey)) : OpLog::Value()}});
        }

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

    [[nodiscard]] static auto sideColFirst(InstrumentUpdateTarget target) -> int {
        return target == InstrumentUpdateTarget::User ? kUserColFirst : kModelColFirst;
    }

    [[nodiscard]] auto findPadForInstrument(InstrumentUpdateTarget side, int32_t localId) const
        -> std::optional<std::pair<int, int>> {
        const int first = sideColFirst(side);
        for (int row = kInstrumentRowFirst; row < kInstrumentRowFirst + kInstrumentRowCount; ++row)
            for (int col = first; col < first + kColsPerSide; ++col)
                if (padInstruments[static_cast<size_t>(row)][static_cast<size_t>(col)] == localId)
                    return std::pair{row, col};
        return std::nullopt;
    }

    /** Calls fn(row, col, localId) for every mapped instrument pad of one side. */
    template <typename Fn>
    auto forEachSidePad(InstrumentUpdateTarget side, Fn &&fn) const -> void {
        const int first = sideColFirst(side);
        for (int row = kInstrumentRowFirst; row < kInstrumentRowFirst + kInstrumentRowCount; ++row)
            for (int col = first; col < first + kColsPerSide; ++col)
                if (isInstrumentPad(row, col))
                    fn(row, col, *padInstruments[static_cast<size_t>(row)][static_cast<size_t>(col)]);
    }

    /** Your instruments that are on are added to the model's; yours stay on (layout 1: off). */
    auto flushInputToModel() -> void {
        std::vector<std::pair<int, int>> flushed;
        forEachSidePad(InstrumentUpdateTarget::User, [this, &flushed](int row, int col, int32_t localId) {
            if (getEffectiveState(row, col) == 0)
                return;
            if (const auto modelPad = findPadForInstrument(InstrumentUpdateTarget::Model, localId))
                setPadOn(modelPad->first, modelPad->second, true);
            flushed.emplace_back(row, col);
        });
        if (layout1.load(std::memory_order_relaxed))
            for (const auto &[row, col]: flushed)
                setPadOn(row, col, false);
    }

    /** Every model instrument switches off. */
    auto resetModelInstruments() -> void {
        forEachSidePad(InstrumentUpdateTarget::Model,
                       [this](int row, int col, int32_t) { setPadOn(row, col, false); });
    }

    /**
     * The controlled side's instruments become exactly the preset's; then the register and
     * TD (TD after the register, so the register's switch to AUTO TD does not apply).
     */
    auto applyPreset(int presetIndex) -> void {
        const auto &preset = MoodPresets::all()[static_cast<size_t>(presetIndex)];
        forEachSidePad(controlledTarget(), [this, &preset](int row, int col, int32_t localId) {
            setPadOn(row, col, MoodPresets::contains(preset, localId));
        });
        if (RegisterMix::Target::presetsUseMix() && preset.registerMix.has_value()) {
            setRegisterAuto(true); // C4 +/- 2.5 octaves; the mix steers within it
            RegisterMix::Target::set(preset.registerMix);
        } else {
            RegisterMix::Target::set(std::nullopt);
            if (preset.registerValue == RegisterRange::kAuto)
                setRegisterAuto(true);
            else
                selectRegister(preset.registerValue);
        }
        if (preset.tdLevelIndex < 0)
            setTdAuto(true);
        else
            selectTdLevel(preset.tdLevelIndex);
        activePreset.store(presetIndex, std::memory_order_relaxed);
        activePresetModified.store(false, std::memory_order_relaxed);
        refreshPresetLeds();
    }

    auto markPresetModified() -> void {
        if (getActivePreset() < 0 || isActivePresetModified())
            return;
        activePresetModified.store(true, std::memory_order_relaxed);
        refreshPresetLeds();
    }

    auto refreshPresetLeds() -> void {
        for (int col = 0; col < kPresetCount; ++col)
            refreshPadLed(kPresetRow, col);
    }

    auto refreshInstrumentLeds() -> void {
        for (int row = kInstrumentRowFirst; row < kInstrumentRowFirst + kInstrumentRowCount; ++row)
            for (int col = 0; col < kCols; ++col)
                refreshPadLed(row, col);
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
        for (int col = kPolyphonyColFirst; col < kPolyphonyColFirst + kPolyphonyPadCount; ++col)
            refreshPadLed(kAccompRow, col);
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
        const auto target = targetForPad(rowIdx, colIdx);
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
    std::atomic<int> activePreset{-1};
    std::atomic<bool> activePresetModified{false};

    std::array<std::array<uint32_t, kCols>, kRows> gridLastAcceptMs{};
    std::array<uint32_t, kRows> sideLastAcceptMs{};
    std::array<uint8_t, kTopCcCount> topLedColours{};
};
