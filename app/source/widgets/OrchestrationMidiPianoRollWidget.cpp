#include "VirtualOrch/widgets/OrchestrationMidiPianoRollWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

OrchestrationMidiPianoRollWidget::OrchestrationMidiPianoRollWidget(AppSession &sessionIn)
    : NoteIoWidget(sessionIn,
                   UiConstants::workspaceOrchestrationMidiPianoRollTitle,
                   UiConstants::workspaceWidgetTypeOrchestrationMidiPianoRoll) {
    setNoteColours(UiConstants::pianoRollHistoryNoteColour, UiConstants::pianoRollPendingNoteColour);
    refreshFromSession();
}

auto OrchestrationMidiPianoRollWidget::refreshFromSession() -> void {
    setPitchRange(session.modelConfig.inputLow, session.modelConfig.inputHigh);
    const auto snap = session.orchestrationTransformer.getDebugSnapshotSince(windowCutoffCs());
    presentTokens(snap.midiHistory, snap.midiPending);
}
