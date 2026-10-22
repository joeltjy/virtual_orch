#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/**
 * Which reduction model is generating right now. For the HybridTransformer: v1 (T3'') or W2/W3,
 * why (start / pedal released / playing 1 s / silent 0.5 s) and how far the user is towards the
 * 1 s switch to W2/W3; for any other reduction type, that type's model.
 */
class ActiveModelWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit ActiveModelWidget(AppSession &session);
    ~ActiveModelWidget() override;

    auto resized() -> void override;

private:
    class View : public juce::Component {
    public:
        explicit View(AppSession &s) : session(s) {}
        auto paint(juce::Graphics &g) -> void override;
        AppSession &session;
    };

    auto timerCallback() -> void override;

    AppSession &session;
    View view{session};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ActiveModelWidget)
};
