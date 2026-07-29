#include "VirtualOrch/ui/AppRootComponent.h"

#include "VirtualOrch/ui/UiConstants.h"

AppRootComponent::AppRootComponent() {
    settingsPage = std::make_unique<MainComponent>(session);
    workspacePage = std::make_unique<WorkspacePage>(session);

    // Capture before addTab: TabbedComponent parents the page and can zero its bounds.
    const int contentWidth = settingsPage->getWidth() > 0
                                 ? settingsPage->getWidth()
                                 : UiConstants::appRootSettingsContentWidth;
    const int contentHeight = settingsPage->getHeight() > 0
                                  ? settingsPage->getHeight()
                                  : UiConstants::appRootSettingsContentHeight;

    tabs.addTab("Settings", UiConstants::appRootTabColour, settingsPage.get(), false);
    tabs.addTab("Workspace", UiConstants::appRootTabColour, workspacePage.get(), false);
    addAndMakeVisible(tabs);

    setSize(contentWidth, contentHeight + tabs.getTabBarDepth());
}

auto AppRootComponent::paint(juce::Graphics &g) -> void {
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

auto AppRootComponent::resized() -> void {
    tabs.setBounds(getLocalBounds());
}
