#include "VirtualOrch/widgets/ReductionTransformerOutputWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

ReductionTransformerOutputWidget::ReductionTransformerOutputWidget(AppSession &sessionIn)
    : NoteIoWidget(sessionIn,
                   UiConstants::workspaceReductionTransformerOutputTitle,
                   UiConstants::workspaceWidgetTypeReductionTransformerOutput) {
    setNoteColours(UiConstants::pianoRollHistoryNoteColour, UiConstants::pianoRollPendingNoteColour);
    refreshFromSession();
}

auto ReductionTransformerOutputWidget::refreshFromSession() -> void {
    setPitchRange(session.modelConfig.inputLow, session.modelConfig.inputHigh);
    presentTokens(session.activeReduction().getOutputHistory());
}
