#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/widgets/WorkspaceWidget.h"

#include <vector>

/**
 * Live read-only view of MusicTransformer inputData (prompt tokens).
 */
class PromptWidget : public WorkspaceWidget {
public:
    PromptWidget();

    ~PromptWidget() override = default;

    auto resized() -> void override;

    auto setInputData(const std::vector<int32_t> &data) -> void;

private:
    juce::TextEditor display;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PromptWidget)
};
