#include "VirtualOrch/widgets/PlaybackOutputWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

PlaybackOutputWidget::PlaybackOutputWidget(AppSession &sessionIn)
    : NoteIoWidget(sessionIn,
                   UiConstants::workspacePlaybackOutputTitle,
                   UiConstants::workspaceWidgetTypePlaybackOutput,
                   true) {
    setNoteColours(UiConstants::pianoRollHistoryNoteColour, UiConstants::pianoRollPendingNoteColour);
    refreshFromSession();
}

auto PlaybackOutputWidget::refreshFromSession() -> void {
    setPitchRange(session.modelConfig.inputLow, session.modelConfig.inputHigh);
    presentPlaybackRecords(session.outputPlayback.getNoteOnHistorySince(windowCutoffCs()));
}
