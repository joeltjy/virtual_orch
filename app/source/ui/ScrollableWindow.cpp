#include "VirtualOrch/ui/ScrollableWindow.h"

ScrollableWindow::ScrollableWindow(const juce::Component::SafePointer<ScrollableComponent> &contentComponent,
                                   const juce::Component::SafePointer<ParentComponent> &parentComponent,
                                   int32_t windowWidth,
                                   int32_t windowHeight) : content(contentComponent), width(windowWidth),
                                                           height(windowHeight) {
    addAndMakeVisible(contentComponent);
    contentComponent->addChangeListener(this);

    addAndMakeVisible(viewport);
    viewport.setViewedComponent(contentComponent, true);
    viewport.setScrollBarsShown(true, false);

    addAndMakeVisible(border);

    addToDesktop(juce::ComponentPeer::windowHasCloseButton |
                 juce::ComponentPeer::windowHasTitleBar);

    /* BUTTONS */
    addAndMakeVisible(okButton);
    okButton.onClick = [this, contentComponent, parentComponent] {
        if (contentComponent->apply()) {
            if (applyButton.isEnabled()) {
                sendChangeMessage();
            }
            userTriedToCloseWindow();
        }
    };

    addAndMakeVisible(applyButton);
    applyButton.setEnabled(false);
    applyButton.onClick = [this, contentComponent, parentComponent] {
        if (contentComponent->apply()) {
            sendChangeMessage();
            applyButton.setEnabled(false);
        }
    };

    addAndMakeVisible(cancelButton);
    cancelButton.onClick = [this] {
        userTriedToCloseWindow();
    };

    centreWithSize(width, height);
    setVisible(true);
}

ScrollableWindow::~ScrollableWindow() {
}

void ScrollableWindow::paint(juce::Graphics &g) {
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

void ScrollableWindow::resized() {
    auto area = getLocalBounds();

    auto viewportArea = area.removeFromTop(height - 70);
    viewport.setBounds(viewportArea.reduced(8));
    border.setBounds(viewportArea);
    content->setSize(content->getWidth() - 10, content->getHeight() - 10);

    auto buttonArea = area.removeFromBottom(70).reduced(8);

    cancelButton.setBounds(buttonArea.removeFromRight(100).reduced(8));
    applyButton.setBounds(buttonArea.removeFromRight(100).reduced(8));
    okButton.setBounds(buttonArea.removeFromRight(100).reduced(8));
}

void ScrollableWindow::userTriedToCloseWindow() {
    dispatchPendingMessages();
    delete this;
}

void ScrollableWindow::changeListenerCallback(ChangeBroadcaster *source) {
    applyButton.setEnabled(true);
}
