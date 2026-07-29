#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/MainComponent.h"
#include "VirtualOrch/ui/WorkspacePage.h"

/**
 * Top-level content: Settings | Workspace. Owns the shared AppSession.
 */
class AppRootComponent : public juce::Component {
public:
    AppRootComponent();

    ~AppRootComponent() override = default;

    auto paint(juce::Graphics &g) -> void override;

    auto resized() -> void override;

    auto getSession() -> AppSession & { return session; }

private:
    AppSession session;
    juce::TabbedComponent tabs{juce::TabbedButtonBar::TabsAtTop};
    std::unique_ptr<MainComponent> settingsPage;
    std::unique_ptr<WorkspacePage> workspacePage;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AppRootComponent)
};
