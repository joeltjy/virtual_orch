#pragma once

#include <JuceHeader.h>

#include <array>
#include <cstdint>
#include <functional>
#include <optional>

#include "Fifo.h"
#include "VirtualOrch/LaunchpadLighting.h"
#include "VirtualOrch/LaunchpadProgrammerMap.h"
#include "VirtualOrch/OrchestrationTransformer.h"

/** Launchpad pad/side state + instrument updates.
 *
 *  Middle 8x8: On (1) / Off (0), toggled on press.
 *  Side column (col = kSideCol = 9): Unmute (1) / Mute (0), toggled on press.
 *  Effective pad state = grid[row][col] * side[row].
 *  Grid press → one InstrumentUpdate for that pad (if mapped).
 *  Side press → InstrumentUpdate for every mapped pad in that row.
 *  Target: 1-indexed row <= 4 (top) → User; row >= 5 (bottom) → Model.
 *  Presses within kBurstMs on the same pad/side are ignored.
 *  LEDs: mapped 8x8 pads follow effective state (row colours); unmapped pads stay off.
 *  Side: muted=red, unmuted=off.
 *  Top-row CCs use `scene_callback(idx)` with idx = cc - 104 (0..7).
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

    using GridCallback = std::function<void(int rowIdx, int colIdx, bool on)>;
    using SideCallback = std::function<void(int rowIdx, bool unmuted)>;
    using SceneCallback = std::function<void(int idx)>;
    /** Programmer-mode static LED: Note On ch1, velocity = palette colour. */
    using LedCallback = std::function<void(int midiNote, uint8_t paletteColour)>;

    GridCallback grid_callback;
    SideCallback side_callback;
    SceneCallback scene_callback;
    LedCallback set_led;

    /** (row, col) → local instrument id; leave empty slots unmapped. */
    std::array<std::array<std::optional<int32_t>, kCols>, kRows> padInstruments{};

    CircularFifo<InstrumentUpdate> *instrumentUpdates = nullptr;

    [[nodiscard]] auto getGridState(int rowIdx, int colIdx) const -> bool {
        if (! isGridPad(rowIdx, colIdx))
            return false;
        return gridState[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
    }

    /** true = Unmute, false = Mute. Defaults to Unmute. */
    [[nodiscard]] auto getSideUnmuted(int rowIdx) const -> bool {
        if (rowIdx < 0 || rowIdx >= kRows)
            return false;
        return sideUnmuted[static_cast<size_t>(rowIdx)];
    }

    [[nodiscard]] auto getEffectiveState(int rowIdx, int colIdx) const -> uint8_t {
        if (! isGridPad(rowIdx, colIdx))
            return 0;
        const auto on = gridState[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
        const auto unmuted = sideUnmuted[static_cast<size_t>(rowIdx)];
        return static_cast<uint8_t>((on && unmuted) ? 1 : 0);
    }

    auto refreshPadLed(int rowIdx, int colIdx) -> void {
        if (! set_led || ! isGridPad(rowIdx, colIdx))
            return;
        const auto note = LaunchpadProgrammerMap::noteFromPad(rowIdx, colIdx);
        if (! note.has_value())
            return;

        const bool mapped =
            padInstruments[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)].has_value();
        const auto colour = mapped ? LaunchpadLighting::colourForGridEffective(
                                         rowIdx, getEffectiveState(rowIdx, colIdx) != 0)
                                   : LaunchpadLighting::kOff;
        set_led(*note, colour);
    }

    auto refreshSideLed(int rowIdx) -> void {
        if (! set_led || rowIdx < 0 || rowIdx >= kRows)
            return;
        const auto note = LaunchpadProgrammerMap::noteFromPad(rowIdx, kSideCol);
        if (! note.has_value())
            return;
        set_led(*note, LaunchpadLighting::colourForSide(sideUnmuted[static_cast<size_t>(rowIdx)]));
    }

    auto refreshAllLeds() -> void {
        for (int row = 0; row < kRows; ++row) {
            refreshSideLed(row);
            for (int col = 0; col < kCols; ++col)
                refreshPadLed(row, col);
        }
    }

    /** MIDI press/release. Only presses toggle/emit. Optional nowMs for tests. */
    auto handleInput(int rowIdx, int colIdx, bool pressed,
                     std::optional<uint32_t> nowMs = std::nullopt) -> void {
        if (! pressed)
            return;

        const uint32_t now = nowMs.value_or(juce::Time::getMillisecondCounter());

        if (isGridPad(rowIdx, colIdx)) {
            auto &lastMs = gridLastAcceptMs[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
            if (! acceptPress(lastMs, now))
                return;

            auto &on = gridState[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
            on = ! on;

            emitPadUpdate(rowIdx, colIdx);
            refreshPadLed(rowIdx, colIdx);
            if (grid_callback)
                grid_callback(rowIdx, colIdx, on);
            return;
        }

        if (colIdx == kSideCol && rowIdx >= 0 && rowIdx < kRows) {
            auto &lastMs = sideLastAcceptMs[static_cast<size_t>(rowIdx)];
            if (! acceptPress(lastMs, now))
                return;

            auto &unmuted = sideUnmuted[static_cast<size_t>(rowIdx)];
            unmuted = ! unmuted;

            for (int col = 0; col < kCols; ++col) {
                emitPadUpdate(rowIdx, col);
                refreshPadLed(rowIdx, col);
            }
            refreshSideLed(rowIdx);

            if (side_callback)
                side_callback(rowIdx, unmuted);
        }
    }

    /** Top-row scene CC press: scene_callback(idx), idx in 0..7. */
    auto handleSceneInput(int idx) -> void {
        if (idx < 0 || idx >= kTopCcCount)
            return;

        if (scene_callback)
            scene_callback(idx);
    }

private:
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

    /** 1-indexed row: <= 4 (top) User; >= 5 (bottom) Model. */
    static auto targetForRow(int rowIdx) -> InstrumentUpdateTarget {
        const int oneIndexed = rowIdx + 1;
        return oneIndexed <= 4 ? InstrumentUpdateTarget::User : InstrumentUpdateTarget::Model;
    }

    auto emitPadUpdate(int rowIdx, int colIdx) -> void {
        if (instrumentUpdates == nullptr || ! isGridPad(rowIdx, colIdx))
            return;

        const auto &instrument = padInstruments[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
        if (! instrument.has_value())
            return;

        InstrumentUpdate update;
        update.localInstrumentId = *instrument;
        update.target = targetForRow(rowIdx);
        update.state = getEffectiveState(rowIdx, colIdx);
        instrumentUpdates->push(update);
    }

    std::array<std::array<bool, kCols>, kRows> gridState{};
    std::array<bool, kRows> sideUnmuted = [] {
        std::array<bool, kRows> unmuted{};
        unmuted.fill(true);
        return unmuted;
    }();

    std::array<std::array<uint32_t, kCols>, kRows> gridLastAcceptMs{};
    std::array<uint32_t, kRows> sideLastAcceptMs{};
};
