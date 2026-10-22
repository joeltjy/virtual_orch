#pragma once

#include <JuceHeader.h>

#include <vector>

#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/** The last Launchpad actions in words, newest at the bottom. */
class ConsoleView : public juce::Component {
public:
    static constexpr size_t kLines = 5;
    float textScale = 1.0f; // larger text shows fewer, bigger lines

    auto setLines(std::vector<juce::String> next) -> void;
    auto paint(juce::Graphics &g) -> void override;

private:
    std::vector<juce::String> lines;
};

class UserSelectionWidget : public WorkspaceWidget, private juce::Timer {
public:
    explicit UserSelectionWidget(AppSession &session);
    ~UserSelectionWidget() override;

    auto resized() -> void override;

protected:
    auto textScaleChanged() -> void override { view.textScale = getTextScale(); view.repaint(); }

private:
    auto timerCallback() -> void override;

    AppSession &session;
    ConsoleView view;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(UserSelectionWidget)
};
