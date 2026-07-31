#include "VirtualOrch/widgets/TokenInputPianoRollWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

TokenInputPianoRollWidget::TokenInputPianoRollWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceTokenPianoRollTitle,
                      UiConstants::workspaceWidgetTypeTokenPianoRoll),
      session(sessionIn),
      pianoRoll(sessionIn.clock) {
    getContentComponent().addAndMakeVisible(pianoRoll);
    refreshFromSession();
    startTimer(UiConstants::pianoRollTimerIntervalMs);
}

TokenInputPianoRollWidget::~TokenInputPianoRollWidget() {
    stopTimer();
}

auto TokenInputPianoRollWidget::resized() -> void {
    WorkspaceWidget::resized();
    pianoRoll.setBounds(getContentComponent().getLocalBounds());
}

auto TokenInputPianoRollWidget::timerCallback() -> void {
    refreshFromSession();
}

auto TokenInputPianoRollWidget::refreshFromSession() -> void {
    pianoRoll.setPitchRange(session.modelConfig.inputLow, session.modelConfig.inputHigh);
    if (session.inputFilter != nullptr)
        pianoRoll.setNotes(session.inputFilter->getPastInput());
    else
        pianoRoll.setNotes({});
}
