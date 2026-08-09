#include "VirtualOrch/views/OrchestrationDebugView.h"

#include "VirtualOrch/ui/UiConstants.h"
#include "VirtualOrch/ui/WorkspaceCanvas.h"

namespace views {

auto OrchestrationDebugView::applyTo(WorkspaceCanvas &canvas) -> void {
    canvas.beginViewLayout();
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeOrchestrationMidiPianoRoll,
                             {UiConstants::workspaceOrchestrationDebugMidiX,
                              UiConstants::workspaceOrchestrationDebugMidiY,
                              UiConstants::workspaceOrchestrationDebugRollWidth,
                              UiConstants::workspaceOrchestrationDebugRollHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeOrchestrationConditioningPianoRoll,
                             {UiConstants::workspaceOrchestrationDebugConditioningX,
                              UiConstants::workspaceOrchestrationDebugConditioningY,
                              UiConstants::workspaceOrchestrationDebugRollWidth,
                              UiConstants::workspaceOrchestrationDebugRollHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeOrchestrationReductionPianoRoll,
                             {UiConstants::workspaceOrchestrationDebugReductionX,
                              UiConstants::workspaceOrchestrationDebugReductionY,
                              UiConstants::workspaceOrchestrationDebugRollWidth,
                              UiConstants::workspaceOrchestrationDebugRollHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeGeneration,
                             {UiConstants::workspaceOrchestrationDebugGenerationX,
                              UiConstants::workspaceOrchestrationDebugGenerationY,
                              UiConstants::workspaceOrchestrationDebugGenerationWidth,
                              UiConstants::workspaceOrchestrationDebugGenerationHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeMode,
                             {UiConstants::workspaceOrchestrationDebugModeX,
                              UiConstants::workspaceOrchestrationDebugModeY,
                              UiConstants::workspaceOrchestrationDebugModeWidth,
                              UiConstants::workspaceOrchestrationDebugModeHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeActiveInstruments,
                             {UiConstants::workspaceOrchestrationDebugActiveInstrumentsX,
                              UiConstants::workspaceOrchestrationDebugActiveInstrumentsY,
                              UiConstants::workspaceOrchestrationDebugActiveInstrumentsWidth,
                              UiConstants::workspaceOrchestrationDebugActiveInstrumentsHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeOrchestrationOutputVisualizer,
                             {UiConstants::workspaceOrchestrationDebugOutputVisualizerX,
                              UiConstants::workspaceOrchestrationDebugOutputVisualizerY,
                              UiConstants::workspaceOrchestrationDebugOutputVisualizerWidth,
                              UiConstants::workspaceOrchestrationDebugOutputVisualizerHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypePlaybackOutput,
                             {UiConstants::workspaceOrchestrationDebugPlaybackOutputX,
                              UiConstants::workspaceOrchestrationDebugPlaybackOutputY,
                              UiConstants::workspaceOrchestrationDebugPlaybackOutputWidth,
                              UiConstants::workspaceOrchestrationDebugPlaybackOutputHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeReductionTransformerOutput,
                             {UiConstants::workspaceOrchestrationDebugReductionTransformerOutputX,
                              UiConstants::workspaceOrchestrationDebugReductionTransformerOutputY,
                              UiConstants::workspaceOrchestrationDebugReductionTransformerOutputWidth,
                              UiConstants::workspaceOrchestrationDebugReductionTransformerOutputHeight});
    canvas.endViewLayout();
}

} // namespace views
