#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/widgets/WorkspaceWidget.h"

#include <vector>

class AppSession;

/**
 * Read-only list of the lookback context the active reduction feeds ONNX each step.
 * Appends throttled snapshots to Documents/virtual-orch/Logs/rt_model_input.log
 * when that lookback changes (rotated when the file grows large).
 */
class ReductionModelInputWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit ReductionModelInputWidget(AppSession &session);

    ~ReductionModelInputWidget() override;

    auto resized() -> void override;

private:
    auto timerCallback() -> void override;

    auto refreshFromSession() -> void;

    AppSession &session;
    juce::TextEditor display;
    /** Packed lookback shown in the editor (avoid TextEditor::getText compares). */
    std::vector<int32_t> lastDisplayedLookback;
    /** Last lookback written to the log file. */
    std::vector<int32_t> lastLoggedLookback;
    /** Pending lookback waiting for the log throttle. */
    std::vector<int32_t> pendingLogLookback;
    bool logPending = false;
    uint32_t lastLogFlushMs = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ReductionModelInputWidget)
};
