#include "VirtualOrch/widgets/TokenInputPianoRollWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/NoteWindow.h"
#include "VirtualOrch/ui/UiConstants.h"

TokenInputPianoRollWidget::TokenInputPianoRollWidget(AppSession &sessionIn)
    : NoteIoWidget(sessionIn,
                   UiConstants::workspaceTokenPianoRollTitle,
                   UiConstants::workspaceWidgetTypeTokenPianoRoll) {
    refreshFromSession();
}

auto TokenInputPianoRollWidget::refreshFromSession() -> void {
    setPitchRange(session.modelConfig.inputLow, session.modelConfig.inputHigh);
    if (session.inputFilter != nullptr)
        presentTokens(
            NoteWindow::filteredTokens(session.inputFilter->getPastInput(), windowCutoffCs()));
    else
        presentTokens({});
}
