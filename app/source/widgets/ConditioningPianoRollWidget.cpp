#include "VirtualOrch/widgets/ConditioningPianoRollWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

ConditioningPianoRollWidget::ConditioningPianoRollWidget(AppSession &sessionIn)
    : SessionPianoRollWidget(sessionIn,
                             UiConstants::workspaceConditioningPianoRollTitle,
                             UiConstants::workspaceWidgetTypeConditioningPianoRoll) {
    refreshFromSession();
}

auto ConditioningPianoRollWidget::refreshFromSession() -> void {
    // Prefer filter conditioning range when pitch-split is active; else full input range.
    if (session.modelConfig.inputFilterType == InputFilterType::PitchRangeSplit)
        pianoRoll.setPitchRange(session.modelConfig.filterConditioningLow,
                                session.modelConfig.filterConditioningHigh - 1);
    else
        pianoRoll.setPitchRange(session.modelConfig.inputLow, session.modelConfig.inputHigh);

    if (session.inputFilter != nullptr)
        pianoRoll.setNotes(session.inputFilter->getPastConditioning());
    else
        pianoRoll.setNotes({});
}
