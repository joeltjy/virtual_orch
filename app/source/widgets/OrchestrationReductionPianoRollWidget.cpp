#include "VirtualOrch/widgets/OrchestrationReductionPianoRollWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

OrchestrationReductionPianoRollWidget::OrchestrationReductionPianoRollWidget(AppSession &sessionIn)
    : NoteIoWidget(sessionIn,
                   UiConstants::workspaceOrchestrationReductionPianoRollTitle,
                   UiConstants::workspaceWidgetTypeOrchestrationReductionPianoRoll) {
    setNoteColours(UiConstants::pianoRollHistoryNoteColour, UiConstants::pianoRollPendingNoteColour);
    refreshFromSession();
}

auto OrchestrationReductionPianoRollWidget::refreshFromSession() -> void {
    setPitchRange(session.modelConfig.inputLow, session.modelConfig.inputHigh);
    const auto snap = session.orchestrationTransformer.getDebugSnapshotSince(windowCutoffCs());
    presentTokens(snap.reductionHistory, snap.reductionPending);
}
