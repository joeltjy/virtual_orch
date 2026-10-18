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
    canvas.endViewLayout();
}

} // namespace views
