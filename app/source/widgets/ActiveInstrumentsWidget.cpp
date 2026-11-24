#include "VirtualOrch/widgets/ActiveInstrumentsWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

ActiveInstrumentsWidget::ActiveInstrumentsWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceActiveInstrumentsTitle,
                      UiConstants::workspaceWidgetTypeActiveInstruments),
      session(sessionIn) {
    getContentComponent().addAndMakeVisible(grid);
    getContentComponent().addChildComponent(tangibleStatus);
    tangibleStatus.setColour(juce::Label::textColourId, UiConstants::workspaceWidgetTitleTextColour);
    refreshFromSession();
    startTimer(UiConstants::pianoRollTimerIntervalMs);
}

ActiveInstrumentsWidget::~ActiveInstrumentsWidget() {
    stopTimer();
}

auto ActiveInstrumentsWidget::resized() -> void {
    WorkspaceWidget::resized();
    auto area = getContentComponent().getLocalBounds();
    if (tangibleStatus.isVisible())
        tangibleStatus.setBounds(area.removeFromBottom(42).reduced(8, 2));
    grid.setBounds(area);
}

auto ActiveInstrumentsWidget::timerCallback() -> void {
    refreshFromSession();
}

auto ActiveInstrumentsWidget::refreshFromSession() -> void {
    grid.setShowKeys(session.keyboardLaunchpadEnabled.load());
    const auto mask = Sympathetic::state().tangibleMask.load(std::memory_order_relaxed);
    grid.setFromLaunchpad(session.midiInputProcess.getLaunchpadGrid(), mask);

    const bool tangibleActive = mask != Sympathetic::kTangibleInactive;
    if (tangibleActive != tangibleStatus.isVisible()) {
        tangibleStatus.setVisible(tangibleActive);
        resized();
    }
    if (tangibleActive) {
        juce::StringArray enabledParts;
        for (const auto &part: Sympathetic::kMarkerParts)
            if ((mask & part.localIds) != 0)
                enabledParts.add(juce::String(part.name.data()));
        tangibleStatus.setText(enabledParts.isEmpty() ? "Tangible: No parts enabled"
                                                     : "Tangible enabled: " + enabledParts.joinIntoString(", "),
                               juce::dontSendNotification);
    }
}
