#include "VirtualOrch/widgets/ReductionModelInputWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/NoteWindow.h"
#include "VirtualOrch/reduction/DenseTypes.h"
#include "VirtualOrch/reduction/ReductionTransformer.h"
#include "VirtualOrch/ui/UiConstants.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace {

/** UI / poll interval — keep off the NoteIo 10 ms path so Stop stays responsive. */
constexpr int kRefreshMs = 100;
/** At most one full lookback snapshot written this often. */
constexpr uint32_t kLogMinIntervalMs = 500;
/** Rotate (truncate) the log once it exceeds this size. */
constexpr int64_t kLogMaxBytes = 2 * 1024 * 1024;

/** Last N events from packed inputData, onset-sorted for readability. */
auto historyPacked(const std::vector<int32_t> &data, size_t stride, size_t maxNotes)
    -> std::vector<int32_t> {
    if (stride == 0 || data.size() < stride)
        return {};

    const size_t noteCount = data.size() / stride;
    const size_t skip = noteCount > maxNotes ? noteCount - maxNotes : size_t{0};
    std::vector<int32_t> packed(data.begin() + static_cast<std::ptrdiff_t>(skip * stride),
                                data.end());

    const size_t n = packed.size() / stride;
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return packed[a * stride] < packed[b * stride];
    });
    bool alreadySorted = true;
    for (size_t i = 0; i < n; ++i) {
        if (order[i] != i) {
            alreadySorted = false;
            break;
        }
    }
    if (alreadySorted)
        return packed;

    std::vector<int32_t> sorted;
    sorted.reserve(packed.size());
    for (const size_t idx: order) {
        const auto base = static_cast<std::ptrdiff_t>(idx * stride);
        sorted.insert(sorted.end(),
                      packed.begin() + base,
                      packed.begin() + base + static_cast<std::ptrdiff_t>(stride));
    }
    return sorted;
}

auto formatHistoryList(const std::vector<int32_t> &packed, size_t stride) -> juce::String {
    juce::String text;
    const size_t noteCount = stride == 0 ? 0 : packed.size() / stride;
    text.preallocateBytes(noteCount * 64);
    text << noteCount << " notes (RT input history)\n";
    for (size_t i = 0; i + stride - 1 < packed.size(); i += stride) {
        const int32_t onset = packed[i];
        const int32_t durToken = packed[i + 1];
        const int32_t noteToken = packed[i + 2];
        int32_t velocity = DenseConfig::DefaultVelocity;
        if (stride >= 4) {
            velocity = juce::jlimit(
                0, DenseConfig::MaxVelocity - 1,
                packed[i + 3] - static_cast<int32_t>(DenseVocab::VelocityOffset));
        }
        if (stride >= 4) {
            // Dense: decode instrument from DenseVocab (not AMT Token helpers).
            const int32_t instr = DenseQuantize::instrumentOfNoteToken(noteToken);
            const int32_t pitch =
                instr >= 0
                    ? (noteToken - static_cast<int32_t>(DenseVocab::NoteOffset))
                          % DenseConfig::MaxPitch
                    : noteToken;
            const int32_t durCs =
                durToken - static_cast<int32_t>(DenseVocab::DurOffset);
            text << "(" << onset << ", " << durCs << ", " << instr << " - " << pitch
                 << ", vel " << velocity << ")\n";
        } else {
            Token token{onset, durToken, noteToken, velocity};
            text << juce::String(token.toUnderstandableString()) << "\n";
        }
    }
    return text;
}

auto rtModelInputLogFile() -> juce::File {
    const auto dir = juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
                         .getChildFile("virtual-orch")
                         .getChildFile("Logs");
    dir.createDirectory();
    return dir.getChildFile("rt_model_input.log");
}

/** Append on a worker thread so the message thread never blocks on disk. */
auto appendLookbackLogAsync(juce::String body) -> void {
    juce::Thread::launch([body = std::move(body)] {
        auto file = rtModelInputLogFile();
        if (file.getSize() > kLogMaxBytes) {
            const auto rotated = file.getSiblingFile("rt_model_input.prev.log");
            rotated.deleteFile();
            file.moveFileTo(rotated);
        }
        const auto stamp = juce::Time::getCurrentTime().toString(true, true, true, true);
        file.appendText("===== " + stamp + " =====\n" + body + "\n");
    });
}

} // namespace

ReductionModelInputWidget::ReductionModelInputWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceReductionModelInputTitle,
                      UiConstants::workspaceWidgetTypeReductionModelInput),
      session(sessionIn) {
    display.setMultiLine(true, true);
    display.setReadOnly(true);
    display.setScrollbarsShown(true);
    display.setCaretVisible(false);
    display.setPopupMenuEnabled(true);
    display.setColour(juce::TextEditor::backgroundColourId,
                      UiConstants::workspacePromptEditorBackground);
    display.setColour(juce::TextEditor::textColourId, UiConstants::workspacePromptEditorTextColour);
    display.setColour(juce::TextEditor::outlineColourId, juce::Colours::transparentBlack);
    display.setColour(juce::TextEditor::focusedOutlineColourId, juce::Colours::transparentBlack);
    display.setTextToShowWhenEmpty("(empty)", UiConstants::workspacePromptEditorEmptyTextColour);
    getContentComponent().addAndMakeVisible(display);

    startTimer(kRefreshMs);
    refreshFromSession();
}

ReductionModelInputWidget::~ReductionModelInputWidget() {
    stopTimer();
}

auto ReductionModelInputWidget::resized() -> void {
    WorkspaceWidget::resized();
    display.setBounds(getContentComponent().getLocalBounds().reduced(4));
}

auto ReductionModelInputWidget::timerCallback() -> void {
    refreshFromSession();
}

auto ReductionModelInputWidget::refreshFromSession() -> void {
    const bool dense = isDenseMusicArch(session.musicModelArch);
    const size_t stride = dense ? size_t{4} : size_t{3};
    const auto packed =
        historyPacked(session.getActiveInputData(), stride, NoteWindow::maxReductionInputNotes);

    if (packed != lastDisplayedLookback) {
        lastDisplayedLookback = packed;
        const auto text = formatHistoryList(packed, stride);
        display.setText(text, juce::dontSendNotification);
        display.applyColourToAllText(UiConstants::workspacePromptEditorTextColour, true);
        display.moveCaretToEnd();
    }

    if (packed != lastLoggedLookback) {
        pendingLogLookback = packed;
        logPending = true;
    }

    const auto nowMs = juce::Time::getMillisecondCounter();
    if (logPending && (lastLogFlushMs == 0 || nowMs - lastLogFlushMs >= kLogMinIntervalMs)) {
        lastLoggedLookback = pendingLogLookback;
        logPending = false;
        lastLogFlushMs = nowMs;
        appendLookbackLogAsync(formatHistoryList(lastLoggedLookback, stride));
    }
}
