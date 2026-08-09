#include "VirtualOrch/widgets/OrchestrationOutputVisualizerWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

OrchestrationOutputVisualizerWidget::OrchestrationOutputVisualizerWidget(AppSession &sessionIn)
    : NoteIoWidget(sessionIn,
                   UiConstants::workspaceOrchestrationOutputVisualizerTitle,
                   UiConstants::workspaceWidgetTypeOrchestrationOutputVisualizer) {
    setNoteColours(UiConstants::pianoRollHistoryNoteColour, UiConstants::pianoRollPendingNoteColour);
    refreshFromSession();
}

auto OrchestrationOutputVisualizerWidget::refreshFromSession() -> void {
    setPitchRange(session.modelConfig.inputLow, session.modelConfig.inputHigh);
    const auto snap = session.orchestrationTransformer.getDebugSnapshot();
    presentOrchestrationNotes(snap.outputHistory);
}
