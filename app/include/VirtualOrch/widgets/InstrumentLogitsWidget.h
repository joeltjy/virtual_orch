#pragma once

#include <JuceHeader.h>

#include <cstdint>
#include <memory>

#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/** Bar chart of post-bias singleton family-combo logits per local instrument. */
class InstrumentLogitsWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit InstrumentLogitsWidget(AppSession &session);

    ~InstrumentLogitsWidget() override;

    auto resized() -> void override;

private:
    struct Impl;

    auto timerCallback() -> void override;

    auto refreshFromSession() -> void;

    AppSession &session;
    std::unique_ptr<Impl> impl;
    uint64_t lastSequence = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(InstrumentLogitsWidget)
};
