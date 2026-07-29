#pragma once

#include <JuceHeader.h>

#include <functional>

/**
 * Framed workspace panel: title bar + content area.
 * Supports title-bar drag, border/corner resize, and close.
 */
class WorkspaceWidget : public juce::Component {
public:
    WorkspaceWidget(juce::String titleText, juce::String widgetType);

    ~WorkspaceWidget() override = default;

    auto paint(juce::Graphics &g) -> void override;

    auto resized() -> void override;

    auto hitTest(int x, int y) -> bool override;

    auto mouseDown(const juce::MouseEvent &event) -> void override;

    auto mouseDrag(const juce::MouseEvent &event) -> void override;

    auto mouseUp(const juce::MouseEvent &event) -> void override;

    auto getContentComponent() -> juce::Component & { return content; }

    auto getTitle() const -> juce::String { return titleLabel.getText(); }

    auto getWidgetType() const -> const juce::String & { return widgetType; }

    /** Serializes type + bounds for ViewStore. */
    [[nodiscard]] auto toVar() const -> juce::var;

    std::function<void(WorkspaceWidget *)> onCloseRequested;

protected:
    juce::Label titleLabel;
    juce::Component content;
    juce::TextButton closeButton;
    juce::String widgetType;

private:
    auto isInTitleBar(juce::Point<int> localPos) const -> bool;

    auto getTitleBarBounds() const -> juce::Rectangle<int>;

    juce::ComponentDragger dragger;
    juce::ComponentBoundsConstrainer constrainer;
    std::unique_ptr<juce::ResizableBorderComponent> resizeBorder;
    bool isDragging = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(WorkspaceWidget)
};
