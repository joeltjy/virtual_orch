#include "VirtualOrch/views/DefaultView.h"

#include "VirtualOrch/ui/UiConstants.h"
#include "VirtualOrch/ui/WorkspaceCanvas.h"

namespace views {

auto DefaultView::applyTo(WorkspaceCanvas &canvas) -> void {
    canvas.beginViewLayout();
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypePrompt,
                             {UiConstants::workspaceStubWidgetX,
                              UiConstants::workspaceStubWidgetY,
                              UiConstants::workspaceStubWidgetWidth,
                              UiConstants::workspaceStubWidgetHeight});
    canvas.placeWidgetOfType(
        UiConstants::workspaceWidgetTypeGeneration,
        {UiConstants::workspaceStubWidgetX + UiConstants::workspaceWidgetCascadeOffsetX,
         UiConstants::workspaceStubWidgetY + UiConstants::workspaceWidgetCascadeOffsetY,
         UiConstants::workspaceStubWidgetWidth,
         UiConstants::workspaceStubWidgetHeight});
    canvas.placeWidgetOfType(
        UiConstants::workspaceWidgetTypeTransport,
        {UiConstants::workspaceStubWidgetX + 2 * UiConstants::workspaceWidgetCascadeOffsetX,
         UiConstants::workspaceStubWidgetY + 2 * UiConstants::workspaceWidgetCascadeOffsetY,
         UiConstants::workspaceStubWidgetWidth,
         UiConstants::workspaceStubWidgetHeight});
    canvas.endViewLayout();
}

} // namespace views
