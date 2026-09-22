#pragma once

#include <JuceHeader.h>

#include <array>
#include <cstdint>
#include <optional>

#include "VirtualOrch/LaunchpadGrid.h"

/** Launchpad view on UI. */
class ActiveInstrumentsView : public juce::Component {
public:
    ActiveInstrumentsView();

    auto setFromLaunchpad(const LaunchpadGrid &grid) -> void;

    auto paint(juce::Graphics &g) -> void override;
    auto resized() -> void override;

    [[nodiscard]] static auto shortNameForLocal(int32_t localId) -> juce::String;

private:
    static constexpr int kUiRows = 9;
    static constexpr int kUiCols = 9;
    static constexpr int kLabelStripWidth = 64;

    struct CellState {
        juce::String label;
        juce::Colour fill = juce::Colours::transparentBlack;
        bool hasContent = false;
    };

    [[nodiscard]] static auto colourForPalette(uint8_t palette) -> juce::Colour;

    auto rebuildCells(const LaunchpadGrid &grid) -> void;
    auto paintGroupBracket(juce::Graphics &g,
                           juce::Rectangle<float> labelArea,
                           const juce::String &text) const -> void;

    std::array<std::array<CellState, kUiCols>, kUiRows> cells{};
    juce::Rectangle<float> gridBounds;
    juce::Rectangle<float> userLabelBounds;
    juce::Rectangle<float> modelLabelBounds;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ActiveInstrumentsView)
};
