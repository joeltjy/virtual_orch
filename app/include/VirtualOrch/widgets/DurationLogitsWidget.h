#pragma once

#include <JuceHeader.h>

#include <cstdint>
#include <memory>

#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/** Line graph of dense RT duration-field logits (post-mask), refreshed on a timer. */
class DurationLogitsWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit DurationLogitsWidget(AppSession &session);

    ~DurationLogitsWidget() override;

    auto resized() -> void override;

private:
    struct Impl;

    auto timerCallback() -> void override;

    auto refreshFromSession() -> void;

    AppSession &session;
    std::unique_ptr<Impl> impl;
    uint64_t lastSequence = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DurationLogitsWidget)
};
