#pragma once

#include <JuceHeader.h>

#include <vector>

#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/** The last Launchpad actions in words, newest at the bottom. */
class ConsoleView : public juce::Component {
public:
    static constexpr size_t kLines = 5;

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

private:
    auto timerCallback() -> void override;

    AppSession &session;
    ConsoleView view;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(UserSelectionWidget)
};
