#include "VirtualOrch/widgets/TokenInputPianoRollWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

TokenInputPianoRollWidget::TokenInputPianoRollWidget(AppSession &sessionIn)
    : SessionPianoRollWidget(sessionIn,
                             UiConstants::workspaceTokenPianoRollTitle,
                             UiConstants::workspaceWidgetTypeTokenPianoRoll) {
    refreshFromSession();
}

auto TokenInputPianoRollWidget::refreshFromSession() -> void {
    pianoRoll.setPitchRange(session.modelConfig.inputLow, session.modelConfig.inputHigh);
    if (session.inputFilter != nullptr)
        pianoRoll.setNotes(session.inputFilter->getPastInput());
    else
        pianoRoll.setNotes({});
}
