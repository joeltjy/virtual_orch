#include "VirtualOrch/ui/WorkspaceCanvas.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"
#include "VirtualOrch/widgets/PromptWidget.h"
#include "VirtualOrch/widgets/GenerationWidget.h"
#include "VirtualOrch/widgets/TransportWidget.h"

#include <algorithm>

WorkspaceCanvas::WorkspaceCanvas() {
    setOpaque(true);
}

WorkspaceCanvas::~WorkspaceCanvas() {
    suppressLayoutNotifications = true;
    for (auto &widget : widgets)
        widget->removeComponentListener(this);

    if (session != nullptr)
        session->musicTransformer.onInputDataChanged = nullptr;
}

auto WorkspaceCanvas::paint(juce::Graphics &g) -> void {
    g.fillAll(UiConstants::workspaceCanvasBackground);
}

auto WorkspaceCanvas::setSession(AppSession *sessionToUse) -> void {
    if (session != nullptr)
        session->musicTransformer.onInputDataChanged = nullptr;

    session = sessionToUse;
    rebindInputDataCallback();
    refreshPromptWidgetsFromSession();
}

auto WorkspaceCanvas::notifyLayoutChanged() -> void {
    if (suppressLayoutNotifications || onLayoutChanged == nullptr)
        return;
    onLayoutChanged();
}

auto WorkspaceCanvas::componentMovedOrResized(juce::Component &, bool wasMoved, bool wasResized)
    -> void {
    if (wasMoved || wasResized)
        notifyLayoutChanged();
}

auto WorkspaceCanvas::bindWidget(WorkspaceWidget &widget) -> void {
    widget.onCloseRequested = [this](WorkspaceWidget *toClose) {
        juce::Component::SafePointer<WorkspaceCanvas> safeThis(this);
        juce::MessageManager::callAsync([safeThis, toClose] {
            if (safeThis != nullptr)
                safeThis->removeWidget(toClose);
        });
    };
    widget.addComponentListener(this);
}

auto WorkspaceCanvas::createWidgetForType(const juce::String &type) const
    -> std::unique_ptr<WorkspaceWidget> {
    if (type == UiConstants::workspaceWidgetTypePrompt)
        return std::make_unique<PromptWidget>();

    if (type == UiConstants::workspaceWidgetTypeGeneration) {
        if (session == nullptr)
            return nullptr;
        return std::make_unique<GenerationWidget>(*session);
    }

    if (type == UiConstants::workspaceWidgetTypeTransport) {
        if (session == nullptr)
            return nullptr;
        return std::make_unique<TransportWidget>(*session);
    }

    // Legacy views may still contain stubs.
    if (type == UiConstants::workspaceWidgetTypeStub)
        return std::make_unique<WorkspaceWidget>(UiConstants::workspaceStubWidgetTitle,
                                                 UiConstants::workspaceWidgetTypeStub);

    return nullptr;
}

auto WorkspaceCanvas::nextCascadedBounds() -> juce::Rectangle<int> {
    const int offsetIndex = nextCascadeIndex++;
    return {UiConstants::workspaceStubWidgetX
                + offsetIndex * UiConstants::workspaceWidgetCascadeOffsetX,
            UiConstants::workspaceStubWidgetY
                + offsetIndex * UiConstants::workspaceWidgetCascadeOffsetY,
            UiConstants::workspaceStubWidgetWidth,
            UiConstants::workspaceStubWidgetHeight};
}

auto WorkspaceCanvas::addWidget(std::unique_ptr<WorkspaceWidget> widget,
                                juce::Rectangle<int> bounds) -> void {
    widget->setBounds(bounds);
    bindWidget(*widget);
    addAndMakeVisible(widget.get());
    widget->toFront(false);
    widgets.push_back(std::move(widget));
    rebindInputDataCallback();
    refreshPromptWidgetsFromSession();
    notifyLayoutChanged();
}

auto WorkspaceCanvas::addPromptWidget() -> void {
    auto widget = createWidgetForType(UiConstants::workspaceWidgetTypePrompt);
    addWidget(std::move(widget), nextCascadedBounds());
}

auto WorkspaceCanvas::addGenerationWidget() -> void {
    auto widget = createWidgetForType(UiConstants::workspaceWidgetTypeGeneration);
    if (widget == nullptr)
        return;
    addWidget(std::move(widget), nextCascadedBounds());
}

auto WorkspaceCanvas::addTransportWidget() -> void {
    auto widget = createWidgetForType(UiConstants::workspaceWidgetTypeTransport);
    if (widget == nullptr)
        return;
    addWidget(std::move(widget), nextCascadedBounds());
}

auto WorkspaceCanvas::addDefaultWidgets() -> void {
    suppressLayoutNotifications = true;
    clearWidgets();
    addPromptWidget();
    addGenerationWidget();
    addTransportWidget();
    suppressLayoutNotifications = false;
}

auto WorkspaceCanvas::removeWidget(WorkspaceWidget *widget) -> void {
    const auto it = std::find_if(widgets.begin(), widgets.end(), [widget](const auto &owned) {
        return owned.get() == widget;
    });

    if (it != widgets.end()) {
        (*it)->removeComponentListener(this);
        widgets.erase(it);
        rebindInputDataCallback();
        notifyLayoutChanged();
    }
}

auto WorkspaceCanvas::clearWidgets() -> void {
    for (auto &widget : widgets)
        widget->removeComponentListener(this);
    widgets.clear();
    nextCascadeIndex = 0;
}

auto WorkspaceCanvas::rebindInputDataCallback() -> void {
    if (session == nullptr)
        return;

    juce::Component::SafePointer<WorkspaceCanvas> safeThis(this);
    session->musicTransformer.onInputDataChanged = [safeThis](std::vector<int32_t> data) {
        if (safeThis == nullptr)
            return;

        for (auto &widget : safeThis->widgets) {
            if (widget != nullptr
                && widget->getWidgetType() == UiConstants::workspaceWidgetTypePrompt) {
                static_cast<PromptWidget *>(widget.get())->setInputData(data);
            }
        }
    };
}

auto WorkspaceCanvas::refreshPromptWidgetsFromSession() -> void {
    if (session == nullptr)
        return;

    const auto &data = session->musicTransformer.getInputData();
    for (auto &widget : widgets) {
        if (widget != nullptr && widget->getWidgetType() == UiConstants::workspaceWidgetTypePrompt)
            static_cast<PromptWidget *>(widget.get())->setInputData(data);
    }
}

auto WorkspaceCanvas::toVar() const -> juce::var {
    juce::Array<juce::var> widgetArray;
    for (const auto &widget : widgets)
        widgetArray.add(widget->toVar());

    auto *root = new juce::DynamicObject();
    root->setProperty("widgets", juce::var(widgetArray));
    return juce::var(root);
}

auto WorkspaceCanvas::fromVar(const juce::var &viewJson) -> bool {
    if (! viewJson.isObject())
        return false;

    const auto *root = viewJson.getDynamicObject();
    if (root == nullptr)
        return false;

    const auto widgetsVar = root->getProperty("widgets");
    if (! widgetsVar.isArray())
        return false;

    suppressLayoutNotifications = true;
    clearWidgets();

    for (const auto &entry : *widgetsVar.getArray()) {
        if (! entry.isObject())
            continue;

        const auto *obj = entry.getDynamicObject();
        if (obj == nullptr)
            continue;

        const auto type = obj->getProperty("type").toString();
        auto widget = createWidgetForType(type);
        if (widget == nullptr)
            continue;

        const int x = (int) obj->getProperty("x");
        const int y = (int) obj->getProperty("y");
        const int w = juce::jmax((int) obj->getProperty("w"), UiConstants::workspaceWidgetMinWidth);
        const int h = juce::jmax((int) obj->getProperty("h"), UiConstants::workspaceWidgetMinHeight);
        addWidget(std::move(widget), {x, y, w, h});
    }

    nextCascadeIndex = (int) widgets.size();
    suppressLayoutNotifications = false;
    return true;
}
