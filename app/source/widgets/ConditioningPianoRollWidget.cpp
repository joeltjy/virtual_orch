#include "VirtualOrch/widgets/ConditioningPianoRollWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

ConditioningPianoRollWidget::ConditioningPianoRollWidget(AppSession &sessionIn)
    : NoteIoWidget(sessionIn,
                   UiConstants::workspaceConditioningPianoRollTitle,
                   UiConstants::workspaceWidgetTypeConditioningPianoRoll) {
    refreshFromSession();
}

auto ConditioningPianoRollWidget::refreshFromSession() -> void {
    if (session.modelConfig.inputFilterType == InputFilterType::PitchRangeSplit)
        setPitchRange(session.modelConfig.filterConditioningLow,
                      session.modelConfig.filterConditioningHigh - 1);
    else
        setPitchRange(session.modelConfig.inputLow, session.modelConfig.inputHigh);

    if (session.inputFilter != nullptr)
        presentTokens(session.inputFilter->getPastConditioning());
    else
        presentTokens({});
}
