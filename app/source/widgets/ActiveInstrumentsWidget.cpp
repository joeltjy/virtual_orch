#include "VirtualOrch/widgets/ActiveInstrumentsWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

ActiveInstrumentsWidget::ActiveInstrumentsWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceActiveInstrumentsTitle,
                      UiConstants::workspaceWidgetTypeActiveInstruments),
      session(sessionIn) {
    getContentComponent().addAndMakeVisible(grid);
    refreshFromSession();
    startTimer(UiConstants::pianoRollTimerIntervalMs);
}

ActiveInstrumentsWidget::~ActiveInstrumentsWidget() {
    stopTimer();
}

auto ActiveInstrumentsWidget::resized() -> void {
    WorkspaceWidget::resized();
    grid.setBounds(getContentComponent().getLocalBounds());
}

auto ActiveInstrumentsWidget::timerCallback() -> void {
    refreshFromSession();
}

auto ActiveInstrumentsWidget::refreshFromSession() -> void {
    grid.setFromLaunchpad(session.midiInputProcess.getLaunchpadGrid());
}
