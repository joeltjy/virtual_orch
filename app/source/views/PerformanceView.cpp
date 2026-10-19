#include "VirtualOrch/views/PerformanceView.h"

#include "VirtualOrch/ui/UiConstants.h"
#include "VirtualOrch/ui/WorkspaceCanvas.h"

namespace views {

auto PerformanceView::applyTo(WorkspaceCanvas &canvas) -> void {
    canvas.beginViewLayout();
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeActiveInstruments,
                             {UiConstants::workspacePerformanceViewLaunchpadX,
                              UiConstants::workspacePerformanceViewLaunchpadY,
                              UiConstants::workspacePerformanceViewLaunchpadWidth,
                              UiConstants::workspacePerformanceViewLaunchpadHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeInstrumentActiveLight,
                             {UiConstants::workspacePerformanceViewSeatingX,
                              UiConstants::workspacePerformanceViewSeatingY,
                              UiConstants::workspacePerformanceViewSeatingWidth,
                              UiConstants::workspacePerformanceViewSeatingHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeInstrumentGroupActivity,
                             {UiConstants::workspacePerformanceViewGroupsX,
                              UiConstants::workspacePerformanceViewGroupsY,
                              UiConstants::workspacePerformanceViewGroupsWidth,
                              UiConstants::workspacePerformanceViewGroupsHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeTdViewer,
                             {UiConstants::workspacePerformanceViewTdX,
                              UiConstants::workspacePerformanceViewTdY,
                              UiConstants::workspacePerformanceViewTdWidth,
                              UiConstants::workspacePerformanceViewTdHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeUserSelection,
                             {UiConstants::workspacePerformanceViewConsoleX,
                              UiConstants::workspacePerformanceViewConsoleY,
                              UiConstants::workspacePerformanceViewConsoleWidth,
                              UiConstants::workspacePerformanceViewConsoleHeight});
    canvas.endViewLayout();
}

} // namespace views
