#pragma once

#include <JuceHeader.h>

#include <memory>

#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/**
 * Live τ / τ′ bars for ReductionTransformerV2 pitch bias.
 * Timer re-evaluates the schedule from clock now so values ramp during silence.
 */
class ReductionTauWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit ReductionTauWidget(AppSession &session);

    ~ReductionTauWidget() override;

    auto resized() -> void override;

private:
    struct Impl;

    auto timerCallback() -> void override;

    auto refreshFromSession() -> void;

    AppSession &session;
    std::unique_ptr<Impl> impl;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ReductionTauWidget)
};
