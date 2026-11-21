#include "VirtualOrch/ui/AppRootComponent.h"

#include "VirtualOrch/ui/UiConstants.h"

AppRootComponent::AppRootComponent() {
    session.logSessionStart("live");
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

    // Keyboard as Launchpad: keys reach this listener from any child that does not use them
    // (text fields keep their typing); the window gets it too for keys with no focused child.
    addKeyListener(&session.keyboardLaunchpad);
    setWantsKeyboardFocus(true);
}

AppRootComponent::~AppRootComponent() {
    removeKeyListener(&session.keyboardLaunchpad);
    if (listenedWindow != nullptr)
        listenedWindow->removeKeyListener(&session.keyboardLaunchpad);
}

auto AppRootComponent::parentHierarchyChanged() -> void {
    auto *top = getTopLevelComponent();
    if (top == this || top == listenedWindow.getComponent())
        return;
    if (listenedWindow != nullptr)
        listenedWindow->removeKeyListener(&session.keyboardLaunchpad);
    listenedWindow = top;
    if (top != nullptr)
        top->addKeyListener(&session.keyboardLaunchpad);
}

auto AppRootComponent::paint(juce::Graphics &g) -> void {
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

auto AppRootComponent::resized() -> void {
    tabs.setBounds(getLocalBounds());
}
