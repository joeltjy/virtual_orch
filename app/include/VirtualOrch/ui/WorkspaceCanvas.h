#pragma once

#include <JuceHeader.h>

#include "VirtualOrch/widgets/WorkspaceWidget.h"

#include <functional>
#include <vector>

class AppSession;

/**
 * Hosts workspace widgets. Widget positions are stored in a fixed design space
 * (UiConstants::workspaceLayoutReference*) and scaled to the live canvas size.
 */
class WorkspaceCanvas : public juce::Component, private juce::ComponentListener {
public:
    WorkspaceCanvas();

    ~WorkspaceCanvas() override;

    auto paint(juce::Graphics &g) -> void override;

    auto resized() -> void override;

    auto setSession(AppSession *sessionToUse) -> void;

    auto addPromptWidget() -> void;

    auto addGenerationWidget() -> void;

    auto addTransportWidget() -> void;

    auto addTokenPianoRollWidget() -> void;

    auto addConditioningPianoRollWidget() -> void;

    auto addOrchestrationMidiPianoRollWidget() -> void;

    auto addOrchestrationConditioningPianoRollWidget() -> void;

    auto addOrchestrationReductionPianoRollWidget() -> void;

    auto addActiveInstrumentsWidget() -> void;

    auto addOrchestrationOutputVisualizerWidget() -> void;

    auto addPlaybackOutputWidget() -> void;

    auto addReductionTransformerOutputWidget() -> void;

    auto addModeWidget() -> void;

    /** Clear canvas and suppress layout notifications while seeding a view. */
    auto beginViewLayout() -> void;

    auto endViewLayout() -> void;

    /** Create + place a widget; bounds are in design-space pixels. */
    auto placeWidgetOfType(const juce::String &type, juce::Rectangle<int> designBounds) -> void;

    /** Prompt + Generation + Transport widgets for Default view. */
    auto addDefaultWidgets() -> void;

    /** Token + Conditioning piano roll widgets for InputView. */
    auto addInputViewWidgets() -> void;

    /** OT rolls + Active Instruments for OrchestrationDebug view. */
    auto addOrchestrationDebugViewWidgets() -> void;

    auto getWidgetCount() const -> int { return (int) widgets.size(); }

    [[nodiscard]] auto toVar() const -> juce::var;

    auto fromVar(const juce::var &viewJson) -> bool;

    std::function<void()> onLayoutChanged;

private:
    auto removeWidget(WorkspaceWidget *widget) -> void;

    auto clearWidgets() -> void;

    auto bindWidget(WorkspaceWidget &widget) -> void;

    auto addWidget(std::unique_ptr<WorkspaceWidget> widget, juce::Rectangle<int> designBounds)
        -> void;

    auto nextCascadedBounds() -> juce::Rectangle<int>;

    auto rebindInputDataCallback() -> void;

    auto refreshPromptWidgetsFromSession() -> void;

    auto notifyLayoutChanged() -> void;

    auto componentMovedOrResized(juce::Component &component, bool wasMoved, bool wasResized)
        -> void override;

    [[nodiscard]] auto createWidgetForType(const juce::String &type) const
        -> std::unique_ptr<WorkspaceWidget>;

    [[nodiscard]] auto scaleDesignToCanvas(juce::Rectangle<int> designBounds) const
        -> juce::Rectangle<int>;

    [[nodiscard]] auto scaleCanvasToDesign(juce::Rectangle<int> canvasBounds) const
        -> juce::Rectangle<int>;

    auto applyDesignLayoutToCanvas() -> void;

    auto syncDesignBoundsFromWidget(WorkspaceWidget &widget) -> void;

    AppSession *session = nullptr;
    std::vector<std::unique_ptr<WorkspaceWidget>> widgets;
    std::vector<juce::Rectangle<int>> designBounds;
    int nextCascadeIndex = 0;
    bool suppressLayoutNotifications = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(WorkspaceCanvas)
};
