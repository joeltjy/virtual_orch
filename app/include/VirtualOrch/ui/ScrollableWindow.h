#pragma once

#include <JuceHeader.h>

class ScrollableComponent : public juce::Component, public juce::ChangeBroadcaster {
public:
    virtual bool apply() = 0;
};

class ParentComponent : public juce::Component, public juce::ChangeListener {
};

class ScrollableWindow : public juce::Component, juce::ChangeListener, public juce::ChangeBroadcaster {
public:
    ScrollableWindow(const juce::Component::SafePointer<ScrollableComponent> &contentComponent,
                     const juce::Component::SafePointer<ParentComponent> &parentComponent, int32_t windowWidth,
                     int32_t windowHeight);

    ~ScrollableWindow() override;

    void paint(juce::Graphics &) override;

    void resized() override;

    void userTriedToCloseWindow() override;

    void changeListenerCallback(ChangeBroadcaster *source) override;

private:
    juce::Component::SafePointer<ScrollableComponent> content;
    juce::GroupComponent border;
    int32_t width, height;

    juce::Viewport viewport;

    juce::TextButton okButton{"OK"}, applyButton{"Apply"}, cancelButton{"Cancel"};
};
