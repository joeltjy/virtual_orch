#include "VirtualOrch/widgets/OrchestrationMidiPianoRollWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

OrchestrationMidiPianoRollWidget::OrchestrationMidiPianoRollWidget(AppSession &sessionIn)
    : SessionPianoRollWidget(sessionIn,
                             UiConstants::workspaceOrchestrationMidiPianoRollTitle,
                             UiConstants::workspaceWidgetTypeOrchestrationMidiPianoRoll) {
    pianoRoll.setNoteColours(UiConstants::pianoRollHistoryNoteColour,
                             UiConstants::pianoRollPendingNoteColour);
    refreshFromSession();
}

auto OrchestrationMidiPianoRollWidget::refreshFromSession() -> void {
    pianoRoll.setPitchRange(session.modelConfig.inputLow, session.modelConfig.inputHigh);
    const auto snap = session.orchestrationTransformer.getDebugSnapshot();
    pianoRoll.setNotes(snap.midiHistory);
    pianoRoll.setPendingNotes(snap.midiPending);
}
