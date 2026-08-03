#include "VirtualOrch/widgets/OrchestrationConditioningPianoRollWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

OrchestrationConditioningPianoRollWidget::OrchestrationConditioningPianoRollWidget(
    AppSession &sessionIn)
    : SessionPianoRollWidget(sessionIn,
                             UiConstants::workspaceOrchestrationConditioningPianoRollTitle,
                             UiConstants::workspaceWidgetTypeOrchestrationConditioningPianoRoll) {
    pianoRoll.setNoteColours(UiConstants::pianoRollHistoryNoteColour,
                             UiConstants::pianoRollPendingNoteColour);
    refreshFromSession();
}

auto OrchestrationConditioningPianoRollWidget::refreshFromSession() -> void {
    if (session.modelConfig.inputFilterType == InputFilterType::PitchRangeSplit)
        pianoRoll.setPitchRange(session.modelConfig.filterConditioningLow,
                                session.modelConfig.filterConditioningHigh - 1);
    else
        pianoRoll.setPitchRange(session.modelConfig.inputLow, session.modelConfig.inputHigh);

    const auto snap = session.orchestrationTransformer.getDebugSnapshot();
    pianoRoll.setNotes(snap.conditioningHistory);
    pianoRoll.setPendingNotes(snap.conditioningPending);
}
