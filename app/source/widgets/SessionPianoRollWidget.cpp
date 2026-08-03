#include "VirtualOrch/widgets/SessionPianoRollWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

SessionPianoRollWidget::SessionPianoRollWidget(AppSession &sessionIn,
                                               juce::String titleText,
                                               juce::String widgetType)
    : WorkspaceWidget(std::move(titleText), std::move(widgetType)),
      session(sessionIn),
      pianoRoll(sessionIn.clock) {
    getContentComponent().addAndMakeVisible(pianoRoll);
    startTimer(UiConstants::pianoRollTimerIntervalMs);
}

SessionPianoRollWidget::~SessionPianoRollWidget() {
    stopTimer();
}

auto SessionPianoRollWidget::resized() -> void {
    WorkspaceWidget::resized();
    pianoRoll.setBounds(getContentComponent().getLocalBounds());
}

auto SessionPianoRollWidget::timerCallback() -> void {
    refreshFromSession();
}
