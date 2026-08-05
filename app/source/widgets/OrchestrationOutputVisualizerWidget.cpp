#include "VirtualOrch/widgets/OrchestrationOutputVisualizerWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

OrchestrationOutputVisualizerWidget::OrchestrationOutputVisualizerWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceOrchestrationOutputVisualizerTitle,
                      UiConstants::workspaceWidgetTypeOrchestrationOutputVisualizer),
      session(sessionIn) {
    getContentComponent().addAndMakeVisible(table);
    refreshFromSession();
    startTimer(UiConstants::pianoRollTimerIntervalMs);
}

OrchestrationOutputVisualizerWidget::~OrchestrationOutputVisualizerWidget() {
    stopTimer();
}

auto OrchestrationOutputVisualizerWidget::resized() -> void {
    WorkspaceWidget::resized();
    table.setBounds(getContentComponent().getLocalBounds());
}

auto OrchestrationOutputVisualizerWidget::timerCallback() -> void {
    refreshFromSession();
}

auto OrchestrationOutputVisualizerWidget::refreshFromSession() -> void {
    const auto snap = session.orchestrationTransformer.getDebugSnapshot();
    table.setNotes(snap.outputHistory);
}
