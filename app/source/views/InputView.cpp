#include "VirtualOrch/views/InputView.h"

#include "VirtualOrch/ui/UiConstants.h"
#include "VirtualOrch/ui/WorkspaceCanvas.h"

namespace views {

auto InputView::applyTo(WorkspaceCanvas &canvas) -> void {
    canvas.beginViewLayout();
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeTokenPianoRoll,
                             {UiConstants::workspaceInputViewTokenRollX,
                              UiConstants::workspaceInputViewTokenRollY,
                              UiConstants::workspaceInputViewPianoRollWidth,
                              UiConstants::workspaceInputViewPianoRollHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeConditioningPianoRoll,
                             {UiConstants::workspaceInputViewConditioningRollX,
                              UiConstants::workspaceInputViewConditioningRollY,
                              UiConstants::workspaceInputViewPianoRollWidth,
                              UiConstants::workspaceInputViewPianoRollHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeGeneration,
                             {UiConstants::workspaceInputViewGenerationX,
                              UiConstants::workspaceInputViewGenerationY,
                              UiConstants::workspaceInputViewGenerationWidth,
                              UiConstants::workspaceInputViewGenerationHeight});
    canvas.endViewLayout();
}

} // namespace views
