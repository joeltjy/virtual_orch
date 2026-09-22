#include "VirtualOrch/widgets/PromptWidget.h"

#include "VirtualOrch/MusicTransformer.h"
#include "VirtualOrch/reduction/DenseTypes.h"
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

auto PromptWidget::setInputData(const std::vector<int32_t> &data, size_t tokenStride) -> void {
    // Only the tail is readable anyway, and rebuilding the whole session's text on every
    // refresh grows without bound until it stalls the message thread.
    const size_t stride = tokenStride == 0 ? size_t{3} : tokenStride;
    const size_t tokenCount = data.size() / stride;
    const size_t skipped =
        tokenCount > maxDisplayedTokens ? tokenCount - maxDisplayedTokens : size_t{0};

    juce::String text;
    text.preallocateBytes(maxDisplayedTokens * 48);
    if (skipped > 0)
        text << "... " << juce::String(static_cast<int>(skipped)) << " earlier tokens\n";
    for (size_t i = skipped * stride; i + stride - 1 < data.size(); i += stride) {
        Token token{data[i], data[i + 1], data[i + 2]};
        if (stride >= 4)
            token.velocity = data[i + 3] - static_cast<int32_t>(DenseVocab::VelocityOffset);
        text += juce::String(token.toUnderstandableString()) + "\n";
    }
    display.setText(text, juce::dontSendNotification);
    display.applyColourToAllText(UiConstants::workspacePromptEditorTextColour, true);
    display.moveCaretToEnd();
}
