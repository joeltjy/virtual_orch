#include "VirtualOrch/widgets/WorkspaceWidget.h"

#include "VirtualOrch/ui/UiConstants.h"

WorkspaceWidget::WorkspaceWidget(juce::String titleText, juce::String type)
    : widgetType(std::move(type)) {
    setOpaque(false);
    addMouseListener(this, true);

    constrainer.setSizeLimits(
        UiConstants::workspaceWidgetMinWidth,
        UiConstants::workspaceWidgetMinHeight,
        UiConstants::workspaceWidgetMaxWidth,
        UiConstants::workspaceWidgetMaxHeight);
    constrainer.setMinimumOnscreenAmounts(
        UiConstants::workspaceWidgetTitleBarHeight,
        UiConstants::workspaceWidgetMinVisibleEdge,
        UiConstants::workspaceWidgetMinVisibleEdge,
        UiConstants::workspaceWidgetMinVisibleEdge);

    titleLabel.setText(std::move(titleText), juce::dontSendNotification);
    titleLabel.setJustificationType(juce::Justification::centredLeft);
    titleLabel.setColour(juce::Label::textColourId, UiConstants::workspaceWidgetTitleTextColour);
    titleLabel.setInterceptsMouseClicks(false, false);
    addAndMakeVisible(titleLabel);

    closeButton.setButtonText(UiConstants::workspaceWidgetCloseButtonText);
    closeButton.setColour(juce::TextButton::textColourOffId, UiConstants::workspaceWidgetCloseButtonColour);
    closeButton.setColour(juce::TextButton::buttonColourId, UiConstants::workspaceWidgetTitleBarBackground);
    closeButton.onClick = [this] {
        if (onCloseRequested != nullptr)
            onCloseRequested(this);
    };
    addAndMakeVisible(closeButton);

    content.setOpaque(false);
    addAndMakeVisible(content);

    resizeBorder = std::make_unique<juce::ResizableBorderComponent>(this, &constrainer);
    resizeBorder->setBorderThickness(
        juce::BorderSize<int>(UiConstants::workspaceWidgetResizeHandleThickness));
    addAndMakeVisible(*resizeBorder);
}

auto WorkspaceWidget::paint(juce::Graphics &g) -> void {
    const auto outer = getLocalBounds().toFloat();
    const auto radius = UiConstants::workspaceWidgetCornerRadius;
    const auto border = (float) UiConstants::workspaceWidgetBorderThickness;

    g.setColour(UiConstants::workspaceWidgetBorderColour);
    g.fillRoundedRectangle(outer, radius);

    auto inner = outer.reduced(border);
    const auto innerRadius = juce::jmax(0.0f, radius - border);

    g.setColour(UiConstants::workspaceWidgetContentBackground);
    g.fillRoundedRectangle(inner, innerRadius);

    juce::Path titlePath;
    titlePath.addRoundedRectangle(
        inner.getX(),
        inner.getY(),
        inner.getWidth(),
        (float) UiConstants::workspaceWidgetTitleBarHeight,
        innerRadius,
        innerRadius,
        true,
        true,
        false,
        false);
    g.setColour(UiConstants::workspaceWidgetTitleBarBackground);
    g.fillPath(titlePath);
}

auto WorkspaceWidget::hitTest(int x, int y) -> bool {
    juce::Path outline;
    outline.addRoundedRectangle(getLocalBounds().toFloat(), UiConstants::workspaceWidgetCornerRadius);
    return outline.contains((float) x, (float) y);
}

auto WorkspaceWidget::getTitleBarBounds() const -> juce::Rectangle<int> {
    return getLocalBounds()
        .reduced(UiConstants::workspaceWidgetBorderThickness)
        .withHeight(UiConstants::workspaceWidgetTitleBarHeight);
}

auto WorkspaceWidget::resized() -> void {
    auto bounds = getLocalBounds().reduced(UiConstants::workspaceWidgetBorderThickness);

    auto titleBar = bounds.removeFromTop(UiConstants::workspaceWidgetTitleBarHeight);
    // Keep the outer resize-handle strip free so close stays clickable.
    titleBar.removeFromRight(UiConstants::workspaceWidgetResizeHandleThickness);
    const auto closeBounds = titleBar.removeFromRight(UiConstants::workspaceWidgetCloseButtonWidth);
    closeButton.setBounds(closeBounds);

    titleLabel.setBounds(titleBar.withTrimmedLeft(UiConstants::workspaceWidgetTitlePaddingX));

    content.setBounds(bounds);

    if (resizeBorder != nullptr)
        resizeBorder->setBounds(getLocalBounds());
}

auto WorkspaceWidget::isInTitleBar(juce::Point<int> localPos) const -> bool {
    auto titleBar = getTitleBarBounds();
    titleBar.removeFromRight(UiConstants::workspaceWidgetResizeHandleThickness
                             + UiConstants::workspaceWidgetCloseButtonWidth);
    const auto edge = UiConstants::workspaceWidgetResizeHandleThickness;
    // Leave edge/corner strips for ResizableBorderComponent.
    return titleBar.reduced(edge, 0).contains(localPos);
}

auto WorkspaceWidget::mouseDown(const juce::MouseEvent &event) -> void {
    toFront(false);

    if (event.eventComponent == &closeButton)
        return;

    if (! isInTitleBar(event.getEventRelativeTo(this).getPosition()))
        return;

    isDragging = true;
    dragger.startDraggingComponent(this, event);
}

auto WorkspaceWidget::mouseDrag(const juce::MouseEvent &event) -> void {
    if (! isDragging)
        return;

    dragger.dragComponent(this, event, &constrainer);
}

auto WorkspaceWidget::mouseUp(const juce::MouseEvent &) -> void {
    isDragging = false;
}

auto WorkspaceWidget::toVar() const -> juce::var {
    auto *obj = new juce::DynamicObject();
    obj->setProperty("type", widgetType);
    obj->setProperty("x", getX());
    obj->setProperty("y", getY());
    obj->setProperty("w", getWidth());
    obj->setProperty("h", getHeight());
    return juce::var(obj);
}
