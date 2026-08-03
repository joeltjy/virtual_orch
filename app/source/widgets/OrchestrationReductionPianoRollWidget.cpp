#include "VirtualOrch/widgets/OrchestrationReductionPianoRollWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

OrchestrationReductionPianoRollWidget::OrchestrationReductionPianoRollWidget(AppSession &sessionIn)
    : SessionPianoRollWidget(sessionIn,
                             UiConstants::workspaceOrchestrationReductionPianoRollTitle,
                             UiConstants::workspaceWidgetTypeOrchestrationReductionPianoRoll) {
    pianoRoll.setNoteColours(UiConstants::pianoRollHistoryNoteColour,
                             UiConstants::pianoRollPendingNoteColour);
    refreshFromSession();
}

auto OrchestrationReductionPianoRollWidget::refreshFromSession() -> void {
    pianoRoll.setPitchRange(session.modelConfig.inputLow, session.modelConfig.inputHigh);
    const auto snap = session.orchestrationTransformer.getDebugSnapshot();
    pianoRoll.setNotes(snap.reductionHistory);
    pianoRoll.setPendingNotes(snap.reductionPending);
}
