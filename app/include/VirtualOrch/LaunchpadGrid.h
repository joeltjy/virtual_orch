#pragma once

#include <JuceHeader.h>

#include <array>
#include <functional>
#include <iostream>

/** Launchpad pad/side state + callbacks.
 *
 *  Grid pads (col 0-7) and right-side notes (col = kSideCol = 9): on each **press**,
 *  bool flips and `callback(row, col, current_state)` runs. Releases ignored.
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

    using InputCallback = std::function<void(int rowIdx, int colIdx, bool currentState)>;
    using SceneCallback = std::function<void(int idx)>;

    InputCallback callback;
    SceneCallback scene_callback;

    [[nodiscard]] auto getState(int rowIdx, int colIdx) const -> bool {
        if (!inBounds(rowIdx, colIdx))
            return false;
        return padState[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
    }

    /** MIDI press/release. Only presses (pressed == true) flip state and fire callback. */
    auto handleInput(int rowIdx, int colIdx, bool pressed) -> void {
        if (!pressed)
            return;
        if (!inBounds(rowIdx, colIdx))
            return;

        auto &state = padState[static_cast<size_t>(rowIdx)][static_cast<size_t>(colIdx)];
        state = !state;

        const juce::String line = "[launchpad] callback row=" + juce::String(rowIdx)
                                  + " col=" + juce::String(colIdx)
                                  + " current_state=" + juce::String(state ? 1 : 0);
        std::cout << line << std::endl;
        if (callback)
            callback(rowIdx, colIdx, state);
    }

    /** Top-row scene CC press: scene_callback(idx), idx in 0..7. */
    auto handleSceneInput(int idx) -> void {
        if (idx < 0 || idx >= kTopCcCount)
            return;

        const juce::String line = "[launchpad] scene_callback idx=" + juce::String(idx);
        std::cout << line << std::endl;
        if (scene_callback)
            scene_callback(idx);
    }

private:
    static auto inBounds(int row, int col) -> bool {
        if (row < 0 || row >= kRows)
            return false;
        return (col >= 0 && col < kCols) || col == kSideCol;
    }

    // Cols 0-7 = grid; index 9 = side column (index 8 unused).
    static constexpr int kStateCols = kSideCol + 1;
    std::array<std::array<bool, kStateCols>, kRows> padState{};
};
