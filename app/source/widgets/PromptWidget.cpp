#include "VirtualOrch/widgets/PromptWidget.h"

#include "VirtualOrch/MusicTransformer.h"
#include "VirtualOrch/ui/UiConstants.h"

PromptWidget::PromptWidget()
    : WorkspaceWidget(UiConstants::workspacePromptWidgetTitle,
                      UiConstants::workspaceWidgetTypePrompt) {
    display.setMultiLine(true, true);
    display.setReadOnly(true);
    display.setScrollbarsShown(true);
    display.setCaretVisible(false);
    display.setPopupMenuEnabled(true);
    display.setColour(juce::TextEditor::backgroundColourId, UiConstants::workspacePromptEditorBackground);
    display.setColour(juce::TextEditor::textColourId, UiConstants::workspacePromptEditorTextColour);
    display.setColour(juce::TextEditor::outlineColourId, juce::Colours::transparentBlack);
    display.setColour(juce::TextEditor::focusedOutlineColourId, juce::Colours::transparentBlack);
    display.setTextToShowWhenEmpty(UiConstants::workspacePromptEmptyText,
                                   UiConstants::workspacePromptEditorEmptyTextColour);
    getContentComponent().addAndMakeVisible(display);
}

auto PromptWidget::resized() -> void {
    WorkspaceWidget::resized();
    display.setBounds(getContentComponent().getLocalBounds().reduced(4));
}

auto PromptWidget::setInputData(const std::vector<int32_t> &data) -> void {
    juce::String text;
    for (size_t i = 0; i + 2 < data.size(); i += 3) {
        Token token{data[i], data[i + 1], data[i + 2]};
        text += juce::String(token.toUnderstandableString()) + "\n";
    }
    display.setText(text, juce::dontSendNotification);
    display.applyColourToAllText(UiConstants::workspacePromptEditorTextColour, true);
    display.moveCaretToEnd();
}
