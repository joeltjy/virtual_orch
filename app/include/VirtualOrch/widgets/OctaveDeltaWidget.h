#pragma once

#include <JuceHeader.h>

#include <cstdint>
#include <memory>

#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/** Live counts of IOD octave Δ ∈ {-1, 0, +1} from decoded notes. */
class OctaveDeltaWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit OctaveDeltaWidget(AppSession &session);

    ~OctaveDeltaWidget() override;

    auto resized() -> void override;

private:
    struct Impl;

    auto timerCallback() -> void override;

    auto refreshFromSession() -> void;

    AppSession &session;
    std::unique_ptr<Impl> impl;
    uint64_t lastSequence = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(OctaveDeltaWidget)
};
