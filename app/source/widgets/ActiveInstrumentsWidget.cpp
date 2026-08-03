#include "VirtualOrch/widgets/ActiveInstrumentsWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

ActiveInstrumentsWidget::ActiveInstrumentsWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceActiveInstrumentsTitle,
                      UiConstants::workspaceWidgetTypeActiveInstruments),
      session(sessionIn) {
    getContentComponent().addAndMakeVisible(table);
    refreshFromSession();
    startTimer(UiConstants::pianoRollTimerIntervalMs);
}

ActiveInstrumentsWidget::~ActiveInstrumentsWidget() {
    stopTimer();
}

auto ActiveInstrumentsWidget::resized() -> void {
    WorkspaceWidget::resized();
    table.setBounds(getContentComponent().getLocalBounds());
}

auto ActiveInstrumentsWidget::timerCallback() -> void {
    refreshFromSession();
}

auto ActiveInstrumentsWidget::refreshFromSession() -> void {
    const auto snap = session.orchestrationTransformer.getDebugSnapshot();
    table.setInstruments(snap.userInstruments, snap.modelInstruments);
}
