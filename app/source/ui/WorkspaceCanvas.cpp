#include "VirtualOrch/ui/WorkspaceCanvas.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"
#include "VirtualOrch/views/DefaultView.h"
#include "VirtualOrch/views/InputView.h"
#include "VirtualOrch/views/OrchestrationDebugView.h"
#include "VirtualOrch/widgets/ActiveInstrumentsWidget.h"
#include "VirtualOrch/widgets/ConditioningPianoRollWidget.h"
#include "VirtualOrch/widgets/GenerationWidget.h"
#include "VirtualOrch/widgets/DurationLogitsWidget.h"
#include "VirtualOrch/widgets/InstrumentLogitsWidget.h"
#include "VirtualOrch/widgets/LogitAdjustmentsWidget.h"
#include "VirtualOrch/widgets/OctaveDeltaWidget.h"
#include "VirtualOrch/widgets/ReductionTemperatureWidget.h"
#include "VirtualOrch/widgets/ReductionTauWidget.h"
#include "VirtualOrch/widgets/ModeWidget.h"
#include "VirtualOrch/widgets/OrchestrationConditioningPianoRollWidget.h"
#include "VirtualOrch/widgets/OrchestrationMidiPianoRollWidget.h"
#include "VirtualOrch/widgets/OrchestrationOutputVisualizerWidget.h"
#include "VirtualOrch/widgets/OrchestrationReductionPianoRollWidget.h"
#include "VirtualOrch/widgets/PlaybackOutputWidget.h"
#include "VirtualOrch/widgets/PromptWidget.h"
#include "VirtualOrch/widgets/ReductionTransformerOutputWidget.h"
#include "VirtualOrch/widgets/ReductionModelInputWidget.h"
#include "VirtualOrch/widgets/TokenInputPianoRollWidget.h"
#include "VirtualOrch/widgets/TransportWidget.h"
#include "VirtualOrch/widgets/VocsepOutputWidget.h"

#include <algorithm>

WorkspaceCanvas::WorkspaceCanvas() {
    setOpaque(true);
}

WorkspaceCanvas::~WorkspaceCanvas() {
    suppressLayoutNotifications = true;
    for (auto &widget : widgets)
        widget->removeComponentListener(this);

    if (session != nullptr)
        session->setOnInputDataChanged(nullptr);
}

auto WorkspaceCanvas::paint(juce::Graphics &g) -> void {
    g.fillAll(UiConstants::workspaceCanvasBackground);
}

auto WorkspaceCanvas::resized() -> void {
    applyDesignLayoutToCanvas();
}

auto WorkspaceCanvas::setSession(AppSession *sessionToUse) -> void {
    if (session != nullptr)
        session->setOnInputDataChanged(nullptr);

    session = sessionToUse;
    rebindInputDataCallback();
    refreshPromptWidgetsFromSession();
}

auto WorkspaceCanvas::notifyLayoutChanged() -> void {
    if (suppressLayoutNotifications || onLayoutChanged == nullptr)
        return;
    onLayoutChanged();
}

auto WorkspaceCanvas::scaleDesignToCanvas(juce::Rectangle<int> design) const
    -> juce::Rectangle<int> {
    const int canvasW = getWidth();
    const int canvasH = getHeight();
    if (canvasW <= 0 || canvasH <= 0)
        return design;

    const float sx = (float) canvasW / (float) UiConstants::workspaceLayoutReferenceWidth;
    const float sy = (float) canvasH / (float) UiConstants::workspaceLayoutReferenceHeight;

    return {juce::roundToInt((float) design.getX() * sx),
            juce::roundToInt((float) design.getY() * sy),
            juce::jmax(1, juce::roundToInt((float) design.getWidth() * sx)),
            juce::jmax(1, juce::roundToInt((float) design.getHeight() * sy))};
}

auto WorkspaceCanvas::scaleCanvasToDesign(juce::Rectangle<int> canvasBounds) const
    -> juce::Rectangle<int> {
    const int canvasW = getWidth();
    const int canvasH = getHeight();
    if (canvasW <= 0 || canvasH <= 0)
        return canvasBounds;

    const float sx = (float) UiConstants::workspaceLayoutReferenceWidth / (float) canvasW;
    const float sy = (float) UiConstants::workspaceLayoutReferenceHeight / (float) canvasH;

    return {juce::roundToInt((float) canvasBounds.getX() * sx),
            juce::roundToInt((float) canvasBounds.getY() * sy),
            juce::jmax(1, juce::roundToInt((float) canvasBounds.getWidth() * sx)),
            juce::jmax(1, juce::roundToInt((float) canvasBounds.getHeight() * sy))};
}

auto WorkspaceCanvas::applyDesignLayoutToCanvas() -> void {
    const bool wasSuppressed = suppressLayoutNotifications;
    suppressLayoutNotifications = true;

    for (size_t i = 0; i < widgets.size(); ++i)
        widgets[i]->setBounds(scaleDesignToCanvas(designBounds[i]));

    suppressLayoutNotifications = wasSuppressed;
}

auto WorkspaceCanvas::syncDesignBoundsFromWidget(WorkspaceWidget &widget) -> void {
    for (size_t i = 0; i < widgets.size(); ++i) {
        if (widgets[i].get() == &widget) {
            designBounds[i] = scaleCanvasToDesign(widget.getBounds());
            return;
        }
    }
}

auto WorkspaceCanvas::componentMovedOrResized(juce::Component &component, bool wasMoved,
                                              bool wasResized) -> void {
    if (suppressLayoutNotifications)
        return;

    if (! (wasMoved || wasResized))
        return;

    if (auto *widget = dynamic_cast<WorkspaceWidget *>(&component))
        syncDesignBoundsFromWidget(*widget);

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

    if (type == UiConstants::workspaceWidgetTypeTokenPianoRoll) {
        if (session == nullptr)
            return nullptr;
        return std::make_unique<TokenInputPianoRollWidget>(*session);
    }

    if (type == UiConstants::workspaceWidgetTypeConditioningPianoRoll) {
        if (session == nullptr)
            return nullptr;
        return std::make_unique<ConditioningPianoRollWidget>(*session);
    }

    if (type == UiConstants::workspaceWidgetTypeOrchestrationMidiPianoRoll) {
        if (session == nullptr)
            return nullptr;
        return std::make_unique<OrchestrationMidiPianoRollWidget>(*session);
    }

    if (type == UiConstants::workspaceWidgetTypeOrchestrationConditioningPianoRoll) {
        if (session == nullptr)
            return nullptr;
        return std::make_unique<OrchestrationConditioningPianoRollWidget>(*session);
    }

    if (type == UiConstants::workspaceWidgetTypeOrchestrationReductionPianoRoll) {
        if (session == nullptr)
            return nullptr;
        return std::make_unique<OrchestrationReductionPianoRollWidget>(*session);
    }

    if (type == UiConstants::workspaceWidgetTypeActiveInstruments) {
        if (session == nullptr)
            return nullptr;
        return std::make_unique<ActiveInstrumentsWidget>(*session);
    }

    if (type == UiConstants::workspaceWidgetTypeOrchestrationOutputVisualizer) {
        if (session == nullptr)
            return nullptr;
        return std::make_unique<OrchestrationOutputVisualizerWidget>(*session);
    }

    if (type == UiConstants::workspaceWidgetTypePlaybackOutput) {
        if (session == nullptr)
            return nullptr;
        return std::make_unique<PlaybackOutputWidget>(*session);
    }

    if (type == UiConstants::workspaceWidgetTypeReductionTransformerOutput) {
        if (session == nullptr)
            return nullptr;
        return std::make_unique<ReductionTransformerOutputWidget>(*session);
    }

    if (type == UiConstants::workspaceWidgetTypeReductionModelInput) {
        if (session == nullptr)
            return nullptr;
        return std::make_unique<ReductionModelInputWidget>(*session);
    }

    if (type == UiConstants::workspaceWidgetTypeVocsepOutput) {
        if (session == nullptr)
            return nullptr;
        return std::make_unique<VocsepOutputWidget>(*session);
    }

    if (type == UiConstants::workspaceWidgetTypeMode) {
        if (session == nullptr)
            return nullptr;
        return std::make_unique<ModeWidget>(*session);
    }

    if (type == UiConstants::workspaceWidgetTypeInstrumentLogits) {
        if (session == nullptr)
            return nullptr;
        return std::make_unique<InstrumentLogitsWidget>(*session);
    }
    if (type == UiConstants::workspaceWidgetTypeDurationLogits) {
        if (session == nullptr)
            return nullptr;
        return std::make_unique<DurationLogitsWidget>(*session);
    }

    if (type == UiConstants::workspaceWidgetTypeOctaveDelta) {
        if (session == nullptr)
            return nullptr;
        return std::make_unique<OctaveDeltaWidget>(*session);
    }
    if (type == UiConstants::workspaceWidgetTypeLogitAdjustments) {
        if (session == nullptr)
            return nullptr;
        return std::make_unique<LogitAdjustmentsWidget>(*session);
    }

    if (type == UiConstants::workspaceWidgetTypeReductionTemperature) {
        if (session == nullptr)
            return nullptr;
        return std::make_unique<ReductionTemperatureWidget>(*session);
    }

    if (type == UiConstants::workspaceWidgetTypeReductionTau) {
        if (session == nullptr)
            return nullptr;
        return std::make_unique<ReductionTauWidget>(*session);
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
                                juce::Rectangle<int> design) -> void {
    designBounds.push_back(design);
    widget->setBounds(scaleDesignToCanvas(design));
    bindWidget(*widget);
    addAndMakeVisible(widget.get());
    widget->toFront(false);
    widgets.push_back(std::move(widget));
    rebindInputDataCallback();
    refreshPromptWidgetsFromSession();
    notifyLayoutChanged();
}

auto WorkspaceCanvas::beginViewLayout() -> void {
    suppressLayoutNotifications = true;
    clearWidgets();
}

auto WorkspaceCanvas::endViewLayout() -> void {
    suppressLayoutNotifications = false;
    notifyLayoutChanged();
}

auto WorkspaceCanvas::placeWidgetOfType(const juce::String &type, juce::Rectangle<int> design)
    -> void {
    auto widget = createWidgetForType(type);
    if (widget == nullptr)
        return;
    addWidget(std::move(widget), design);
}

auto WorkspaceCanvas::hasWidgetOfType(const juce::String &type) const -> bool {
    for (const auto &widget: widgets) {
        if (widget != nullptr && widget->getWidgetType() == type)
            return true;
    }
    return false;
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

auto WorkspaceCanvas::addTokenPianoRollWidget() -> void {
    auto widget = createWidgetForType(UiConstants::workspaceWidgetTypeTokenPianoRoll);
    if (widget == nullptr)
        return;
    addWidget(std::move(widget), nextCascadedBounds());
}

auto WorkspaceCanvas::addConditioningPianoRollWidget() -> void {
    auto widget = createWidgetForType(UiConstants::workspaceWidgetTypeConditioningPianoRoll);
    if (widget == nullptr)
        return;
    addWidget(std::move(widget), nextCascadedBounds());
}

auto WorkspaceCanvas::addOrchestrationMidiPianoRollWidget() -> void {
    auto widget = createWidgetForType(UiConstants::workspaceWidgetTypeOrchestrationMidiPianoRoll);
    if (widget == nullptr)
        return;
    addWidget(std::move(widget), nextCascadedBounds());
}

auto WorkspaceCanvas::addOrchestrationConditioningPianoRollWidget() -> void {
    auto widget =
        createWidgetForType(UiConstants::workspaceWidgetTypeOrchestrationConditioningPianoRoll);
    if (widget == nullptr)
        return;
    addWidget(std::move(widget), nextCascadedBounds());
}

auto WorkspaceCanvas::addOrchestrationReductionPianoRollWidget() -> void {
    auto widget =
        createWidgetForType(UiConstants::workspaceWidgetTypeOrchestrationReductionPianoRoll);
    if (widget == nullptr)
        return;
    addWidget(std::move(widget), nextCascadedBounds());
}

auto WorkspaceCanvas::addActiveInstrumentsWidget() -> void {
    auto widget = createWidgetForType(UiConstants::workspaceWidgetTypeActiveInstruments);
    if (widget == nullptr)
        return;
    addWidget(std::move(widget), nextCascadedBounds());
}

auto WorkspaceCanvas::addOrchestrationOutputVisualizerWidget() -> void {
    auto widget = createWidgetForType(UiConstants::workspaceWidgetTypeOrchestrationOutputVisualizer);
    if (widget == nullptr)
        return;
    addWidget(std::move(widget), nextCascadedBounds());
}

auto WorkspaceCanvas::addPlaybackOutputWidget() -> void {
    auto widget = createWidgetForType(UiConstants::workspaceWidgetTypePlaybackOutput);
    if (widget == nullptr)
        return;
    addWidget(std::move(widget), nextCascadedBounds());
}

auto WorkspaceCanvas::addReductionTransformerOutputWidget() -> void {
    auto widget = createWidgetForType(UiConstants::workspaceWidgetTypeReductionTransformerOutput);
    if (widget == nullptr)
        return;
    addWidget(std::move(widget), nextCascadedBounds());
}

auto WorkspaceCanvas::addReductionModelInputWidget() -> void {
    auto widget = createWidgetForType(UiConstants::workspaceWidgetTypeReductionModelInput);
    if (widget == nullptr)
        return;
    addWidget(std::move(widget), nextCascadedBounds());
}

auto WorkspaceCanvas::addModeWidget() -> void {
    auto widget = createWidgetForType(UiConstants::workspaceWidgetTypeMode);
    if (widget == nullptr)
        return;
    addWidget(std::move(widget), nextCascadedBounds());
}

auto WorkspaceCanvas::addInstrumentLogitsWidget() -> void {
    auto widget = createWidgetForType(UiConstants::workspaceWidgetTypeInstrumentLogits);
    if (widget == nullptr)
        return;
    addWidget(std::move(widget), nextCascadedBounds());
}

auto WorkspaceCanvas::addVocsepOutputWidget() -> void {
    auto widget = createWidgetForType(UiConstants::workspaceWidgetTypeVocsepOutput);
    if (widget == nullptr)
        return;
    addWidget(std::move(widget), nextCascadedBounds());
}

auto WorkspaceCanvas::addDurationLogitsWidget() -> void {
    auto widget = createWidgetForType(UiConstants::workspaceWidgetTypeDurationLogits);
    if (widget == nullptr)
        return;
    addWidget(std::move(widget), nextCascadedBounds());
}

auto WorkspaceCanvas::addReductionTemperatureWidget() -> void {
    auto widget = createWidgetForType(UiConstants::workspaceWidgetTypeReductionTemperature);
    if (widget == nullptr)
        return;
    addWidget(std::move(widget), nextCascadedBounds());
}

auto WorkspaceCanvas::addReductionTauWidget() -> void {
    auto widget = createWidgetForType(UiConstants::workspaceWidgetTypeReductionTau);
    if (widget == nullptr)
        return;
    addWidget(std::move(widget), nextCascadedBounds());
}

auto WorkspaceCanvas::addOctaveDeltaWidget() -> void {
    auto widget = createWidgetForType(UiConstants::workspaceWidgetTypeOctaveDelta);
    if (widget == nullptr)
        return;
    addWidget(std::move(widget), nextCascadedBounds());
}

auto WorkspaceCanvas::addLogitAdjustmentsWidget() -> void {
    auto widget = createWidgetForType(UiConstants::workspaceWidgetTypeLogitAdjustments);
    if (widget == nullptr)
        return;
    addWidget(std::move(widget), nextCascadedBounds());
}

auto WorkspaceCanvas::addDefaultWidgets() -> void {
    views::DefaultView::applyTo(*this);
}

auto WorkspaceCanvas::addInputViewWidgets() -> void {
    views::InputView::applyTo(*this);
}

auto WorkspaceCanvas::addOrchestrationDebugViewWidgets() -> void {
    views::OrchestrationDebugView::applyTo(*this);
}

auto WorkspaceCanvas::removeWidget(WorkspaceWidget *widget) -> void {
    const auto it = std::find_if(widgets.begin(), widgets.end(), [widget](const auto &owned) {
        return owned.get() == widget;
    });

    if (it != widgets.end()) {
        const auto index = (size_t) std::distance(widgets.begin(), it);
        (*it)->removeComponentListener(this);
        widgets.erase(it);
        if (index < designBounds.size())
            designBounds.erase(designBounds.begin() + (std::ptrdiff_t) index);
        rebindInputDataCallback();
        notifyLayoutChanged();
    }
}

auto WorkspaceCanvas::clearWidgets() -> void {
    for (auto &widget : widgets)
        widget->removeComponentListener(this);
    widgets.clear();
    designBounds.clear();
    nextCascadeIndex = 0;
}

auto WorkspaceCanvas::rebindInputDataCallback() -> void {
    if (session == nullptr)
        return;

    juce::Component::SafePointer<WorkspaceCanvas> safeThis(this);
    session->setOnInputDataChanged([safeThis](std::vector<int32_t> data) {
        if (safeThis == nullptr)
            return;

        const size_t stride = isDenseMusicArch(safeThis->session->musicModelArch)
                                  ? size_t{4}
                                  : size_t{3};
        for (auto &widget : safeThis->widgets) {
            if (widget != nullptr
                && widget->getWidgetType() == UiConstants::workspaceWidgetTypePrompt) {
                static_cast<PromptWidget *>(widget.get())->setInputData(data, stride);
            }
        }
    });
}

auto WorkspaceCanvas::refreshPromptWidgetsFromSession() -> void {
    if (session == nullptr)
        return;

    const auto data = session->getActiveInputData();
    const size_t stride =
        isDenseMusicArch(session->musicModelArch) ? size_t{4} : size_t{3};
    for (auto &widget : widgets) {
        if (widget != nullptr && widget->getWidgetType() == UiConstants::workspaceWidgetTypePrompt)
            static_cast<PromptWidget *>(widget.get())->setInputData(data, stride);
    }
}

auto WorkspaceCanvas::toVar() const -> juce::var {
    juce::Array<juce::var> widgetArray;
    for (size_t i = 0; i < widgets.size(); ++i) {
        auto *obj = new juce::DynamicObject();
        obj->setProperty("type", widgets[i]->getWidgetType());
        const auto &b = designBounds[i];
        obj->setProperty("x", b.getX());
        obj->setProperty("y", b.getY());
        obj->setProperty("w", b.getWidth());
        obj->setProperty("h", b.getHeight());
        widgetArray.add(juce::var(obj));
    }

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
    applyDesignLayoutToCanvas();
    suppressLayoutNotifications = false;
    return true;
}
