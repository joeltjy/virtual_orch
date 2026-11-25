#pragma once

#include <JuceHeader.h>

#include <array>
#include <cstdint>
#include <optional>

#include "VirtualOrch/LaunchpadGrid.h"
#include "VirtualOrch/Sympathetic.h"

/**
 * Launchpad view on UI: 9x9 (top CC row + 8x8 grid + side column), every button
 * labelled. UI row = pad row + 1; UI col 8 = side column.
 */
class ActiveInstrumentsView : public juce::Component {
public:
    ActiveInstrumentsView();

    auto setFromLaunchpad(const LaunchpadGrid &grid, uint32_t tangibleMask = Sympathetic::kTangibleInactive) -> void;

    auto paint(juce::Graphics &g) -> void override;
    auto resized() -> void override;

    [[nodiscard]] static auto shortNameForLocal(int32_t localId) -> juce::String;

    /** Draw each button's computer-keyboard key (Keyboard as Launchpad) in its corner. */
    auto setShowKeys(bool show) -> void {
        if (show != showKeys) {
            showKeys = show;
            repaint();
        }
    }

    /** Larger labels for an audience (the cells stay the same; text fills more of them). */
    auto setTextScale(float scale) -> void { textScale = scale; resized(); repaint(); }

private:
    static constexpr int kUiRows = 9;
    static constexpr int kUiCols = 9;
    static constexpr int kLabelStripWidth = 64;
    float textScale = 1.0f;
    bool showKeys = false;

    struct CellState {
        int instrument = -1;
        juce::String label;
        /** Palette colour the button shows when lit. */
        uint8_t onPalette = 0;
        bool lit = false;
        /** Computer-keyboard key for this button (KeyboardLaunchpad), or empty. */
        juce::String key;
    };

    [[nodiscard]] static auto colourForPalette(uint8_t palette) -> juce::Colour;

    auto rebuildCells(const LaunchpadGrid &grid, uint32_t tangibleMask) -> void;
    auto paintGroupBracket(juce::Graphics &g,
                           juce::Rectangle<float> labelArea,
                           const juce::String &text) const -> void;

    std::array<std::array<CellState, kUiCols>, kUiRows> cells{};
    juce::Rectangle<float> gridBounds;
    juce::Rectangle<float> instrumentLabelBounds;
    /** The pads control your side (Edit), else the model's: framed on screen. */
    bool controlsUserSide = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ActiveInstrumentsView)
};
