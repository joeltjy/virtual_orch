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
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeReductionTemperature,
                             {UiConstants::workspaceOrchestrationDebugReductionTemperatureX,
                              UiConstants::workspaceOrchestrationDebugReductionTemperatureY,
                              UiConstants::workspaceOrchestrationDebugReductionTemperatureWidth,
                              UiConstants::workspaceOrchestrationDebugReductionTemperatureHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeMode,
                             {UiConstants::workspaceOrchestrationDebugModeX,
                              UiConstants::workspaceOrchestrationDebugModeY,
                              UiConstants::workspaceOrchestrationDebugModeWidth,
                              UiConstants::workspaceOrchestrationDebugModeHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeInstrumentLogits,
                             {UiConstants::workspaceOrchestrationDebugInstrumentLogitsX,
                              UiConstants::workspaceOrchestrationDebugInstrumentLogitsY,
                              UiConstants::workspaceOrchestrationDebugInstrumentLogitsWidth,
                              UiConstants::workspaceOrchestrationDebugInstrumentLogitsHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeReductionTau,
                             {UiConstants::workspaceOrchestrationDebugReductionTauX,
                              UiConstants::workspaceOrchestrationDebugReductionTauY,
                              UiConstants::workspaceOrchestrationDebugReductionTauWidth,
                              UiConstants::workspaceOrchestrationDebugReductionTauHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeActiveInstruments,
                             {UiConstants::workspaceOrchestrationDebugActiveInstrumentsX,
                              UiConstants::workspaceOrchestrationDebugActiveInstrumentsY,
                              UiConstants::workspaceOrchestrationDebugActiveInstrumentsWidth,
                              UiConstants::workspaceOrchestrationDebugActiveInstrumentsHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeOrchestrationOutputVisualizer,
                             {UiConstants::workspaceOrchestrationDebugOutputVisualizerX,
                              UiConstants::workspaceOrchestrationDebugOutputVisualizerY,
                              UiConstants::workspaceOrchestrationDebugRollWidth,
                              UiConstants::workspaceOrchestrationDebugOutputVisualizerHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypePlaybackOutput,
                             {UiConstants::workspaceOrchestrationDebugPlaybackOutputX,
                              UiConstants::workspaceOrchestrationDebugPlaybackOutputY,
                              UiConstants::workspaceOrchestrationDebugRollWidth,
                              UiConstants::workspaceOrchestrationDebugPlaybackOutputHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeReductionTransformerOutput,
                             {UiConstants::workspaceOrchestrationDebugReductionTransformerOutputX,
                              UiConstants::workspaceOrchestrationDebugReductionTransformerOutputY,
                              UiConstants::workspaceOrchestrationDebugRollWidth,
                              UiConstants::workspaceOrchestrationDebugReductionTransformerOutputHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeReductionModelInput,
                             {UiConstants::workspaceOrchestrationDebugReductionModelInputX,
                              UiConstants::workspaceOrchestrationDebugReductionModelInputY,
                              UiConstants::workspaceOrchestrationDebugReductionModelInputWidth,
                              UiConstants::workspaceOrchestrationDebugReductionModelInputHeight});
    canvas.placeWidgetOfType(UiConstants::workspaceWidgetTypeVocsepOutput,
                             {UiConstants::workspaceOrchestrationDebugVocsepOutputX,
                              UiConstants::workspaceOrchestrationDebugVocsepOutputY,
                              UiConstants::workspaceOrchestrationDebugRollWidth,
                              UiConstants::workspaceOrchestrationDebugVocsepOutputHeight});
    canvas.endViewLayout();
}

} // namespace views
