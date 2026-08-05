#pragma once

#include <JuceHeader.h>

/** Directory containing checked-in .onnx / .json model files (the app/ folder). */
[[nodiscard]] inline auto projectModelsDir() -> juce::File {
#if defined(VIRTUAL_ORCH_APP_DIR)
    return juce::File{VIRTUAL_ORCH_APP_DIR};
#else
    return juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
        .getChildFile("virtual-orch")
        .getChildFile("Models");
#endif
}
