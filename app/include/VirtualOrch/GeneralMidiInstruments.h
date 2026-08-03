#pragma once

#include <JuceHeader.h>

#include <cstdint>

namespace GeneralMidiInstruments {

inline constexpr int32_t kNumInstruments = 128;

/** English GM name for id in [0, 127]; otherwise "Instrument N". */
[[nodiscard]] auto nameForId(int32_t id) -> juce::String;

} // namespace GeneralMidiInstruments
