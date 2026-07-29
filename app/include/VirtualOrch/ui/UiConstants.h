#pragma once

#include <JuceHeader.h>

/**
 * Layout and color constants.
 */
namespace UiConstants {

// App root: tabs.
inline const juce::Colour appRootTabColour{juce::Colours::darkgrey};
inline constexpr int appRootSettingsContentWidth = 700;
inline constexpr int appRootSettingsContentHeight = 800;
inline constexpr const char *workspaceViewsComboPlaceholder = "Views";
inline constexpr const char *workspaceDefaultViewName = "Default";
inline constexpr const char *workspaceWidgetTypeStub = "stub";
inline constexpr const char *workspaceWidgetTypePrompt = "prompt";
inline constexpr const char *workspaceWidgetTypeGeneration = "generation";
inline constexpr const char *workspaceWidgetTypeTransport = "transport";
inline constexpr const char *workspacePromptWidgetTitle = "Prompt";
inline constexpr const char *workspaceGenerationWidgetTitle = "Generation";
inline constexpr const char *workspaceTransportWidgetTitle = "Transport";
inline constexpr const char *workspacePromptEmptyText = "(empty)";
inline constexpr const char *workspaceAddMenuPromptItem = "Prompt";
inline constexpr const char *workspaceAddMenuGenerationItem = "Generation";
inline constexpr const char *workspaceAddMenuTransportItem = "Transport";
inline constexpr int workspaceAddMenuPromptItemId = 1;
inline constexpr int workspaceAddMenuGenerationItemId = 2;
inline constexpr int workspaceAddMenuTransportItemId = 3;
inline constexpr int workspaceGenerationButtonWidth = 100;
inline constexpr int workspaceGenerationButtonHeight = 28;
inline constexpr int workspaceGenerationControlGap = 8;
inline constexpr int workspaceGenerationProgressHeight = 24;
inline constexpr int workspaceGenerationContentPadding = 8;
inline constexpr int workspaceTransportTimerIntervalMs = 10;
inline constexpr float workspaceTransportTimeFontHeight = 28.0f;
inline constexpr const char *workspaceSaveAsDialogTitle = "Save View As";
inline constexpr const char *workspaceSaveAsDialogMessage = "Enter a name for this view:";
inline constexpr const char *workspaceSaveAsNameFieldId = "name";
inline constexpr const char *workspaceSaveAsNameFieldLabel = "View name";
inline constexpr const char *workspaceSaveAsConfirmButton = "Save";
inline constexpr const char *workspaceSaveAsCancelButton = "Cancel";
inline constexpr int workspaceSaveAsConfirmResult = 1;
inline constexpr int workspaceSaveAsCancelResult = 0;
inline constexpr const char *workspaceGenerationNoModelTitle = "Generation";
inline constexpr const char *workspaceGenerationNoModelMessage =
    "Load a model in Settings before starting generation.";
inline constexpr const char *workspaceDirtyTitleSuffix = " *";
inline constexpr const char *workspaceOverwriteSaveTitle = "Overwrite View";
inline constexpr const char *workspaceOverwriteSaveMessagePrefix = "Overwrite view \"";
inline constexpr const char *workspaceOverwriteSaveMessageSuffix = "\"?";
inline constexpr const char *workspaceOverwriteConfirmButton = "Overwrite";
inline constexpr const char *workspaceOverwriteCancelButton = "Cancel";
inline constexpr int workspaceOverwriteConfirmResult = 1;
inline constexpr int workspaceOverwriteCancelResult = 0;

// Workspace page: toolbar.
inline constexpr int workspaceToolbarHeight = 36;
inline constexpr int workspaceToolbarPaddingX = 8;
inline constexpr int workspaceToolbarPaddingY = 4;
inline constexpr int workspaceToolbarTitleWidth = 90;
inline constexpr int workspaceToolbarTitleTrailingGap = 8;
inline constexpr int workspaceToolbarControlGap = 6;
inline constexpr int workspaceToolbarButtonWidth = 72;
inline constexpr int workspaceToolbarSaveAsButtonWidth = 88; // slightly wider for "Save As"
inline constexpr int workspaceToolbarViewComboWidth = 140;
// Half-pixel nudge so the toolbar divider sits on a crisp line.
inline constexpr float workspaceToolbarDividerPixelCenterOffset = 0.5f;

// Workspace page: chrome colors.
inline const juce::Colour workspacePageBackground{juce::Colours::black};
inline const juce::Colour workspaceToolbarBackground{juce::Colours::darkgrey.darker(0.4f)};
inline const juce::Colour workspaceToolbarDivider{juce::Colours::grey.withAlpha(0.5f)};
inline constexpr float workspaceToolbarDividerThickness = 1.0f;
inline const juce::Colour workspaceCanvasBackground{juce::Colours::black.brighter(0.08f)};
inline const juce::Colour workspaceToolbarTitleColour{juce::Colours::lightgrey};

// Workspace widget chrome.
inline constexpr int workspaceWidgetTitleBarHeight = 28;
inline constexpr int workspaceWidgetBorderThickness = 1;
inline constexpr float workspaceWidgetCornerRadius = 10.0f;
inline constexpr int workspaceWidgetTitlePaddingX = 8;
inline constexpr int workspaceStubWidgetX = 40;
inline constexpr int workspaceStubWidgetY = 40;
inline constexpr int workspaceStubWidgetWidth = 320;
inline constexpr int workspaceStubWidgetHeight = 200;
inline constexpr int workspaceWidgetMinWidth = 150;
inline constexpr int workspaceWidgetMinHeight = 80;
inline constexpr int workspaceWidgetMaxWidth = 0x3fffffff;
inline constexpr int workspaceWidgetMaxHeight = 0x3fffffff;
inline constexpr int workspaceWidgetResizeHandleThickness = 6;
// Keep at least this much of the widget on the canvas when dragging/resizing.
inline constexpr int workspaceWidgetMinVisibleEdge = 40;
inline constexpr int workspaceWidgetCloseButtonWidth = 28;
inline constexpr int workspaceWidgetCascadeOffsetX = 28;
inline constexpr int workspaceWidgetCascadeOffsetY = 28;
inline constexpr const char *workspaceStubWidgetTitle = "Stub Widget";
inline constexpr const char *workspaceWidgetCloseButtonText = "X";

inline const juce::Colour workspaceWidgetBorderColour{juce::Colours::grey};
inline const juce::Colour workspaceWidgetTitleBarBackground{juce::Colours::darkgrey.darker(0.2f)};
inline const juce::Colour workspaceWidgetContentBackground{juce::Colours::black.brighter(0.12f)};
inline const juce::Colour workspaceWidgetTitleTextColour{juce::Colours::lightgrey};
inline const juce::Colour workspaceWidgetCloseButtonColour{juce::Colours::lightgrey};
inline const juce::Colour workspacePromptEditorBackground{juce::Colours::black.brighter(0.06f)};
inline const juce::Colour workspacePromptEditorTextColour{juce::Colours::lightgrey};
inline const juce::Colour workspacePromptEditorEmptyTextColour{juce::Colours::grey};
inline const juce::Colour workspaceTransportTimeColour{juce::Colours::lightgrey};

} // namespace UiConstants
