#include "VirtualOrch/widgets/OrchestrationConditioningPianoRollWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

OrchestrationConditioningPianoRollWidget::OrchestrationConditioningPianoRollWidget(
    AppSession &sessionIn)
    : NoteIoWidget(sessionIn,
                   UiConstants::workspaceOrchestrationConditioningPianoRollTitle,
                   UiConstants::workspaceWidgetTypeOrchestrationConditioningPianoRoll) {
    setNoteColours(UiConstants::pianoRollHistoryNoteColour, UiConstants::pianoRollPendingNoteColour);
    refreshFromSession();
}

auto OrchestrationConditioningPianoRollWidget::refreshFromSession() -> void {
    if (session.modelConfig.inputFilterType == InputFilterType::PitchRangeSplit)
        setPitchRange(session.modelConfig.filterConditioningLow,
                      session.modelConfig.filterConditioningHigh - 1);
    else
        setPitchRange(session.modelConfig.inputLow, session.modelConfig.inputHigh);

    const auto snap = session.orchestrationTransformer.getDebugSnapshotSince(windowCutoffCs());
    presentTokens(snap.conditioningHistory, snap.conditioningPending);
}
