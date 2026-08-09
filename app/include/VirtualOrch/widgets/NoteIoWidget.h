#pragma once

#include <JuceHeader.h>

#include <vector>

#include "VirtualOrch/OutputPlayback.h"
#include "VirtualOrch/orchestration-models/OrchestrationModel.h"
#include "VirtualOrch/widgets/NoteIoTableView.h"
#include "VirtualOrch/widgets/PianoRollView.h"
#include "VirtualOrch/widgets/WorkspaceWidget.h"

class AppSession;

/**
 * Shared I/O chrome: note table (5 or 6 columns) above a piano roll.
 * Subclasses implement refreshFromSession() and call present* helpers.
 */
class NoteIoWidget : public WorkspaceWidget, private juce::Timer {
public:
    NoteIoWidget(AppSession &session,
                 juce::String titleText,
                 juce::String typeId,
                 bool showChannelColumn = false);

    ~NoteIoWidget() override;

    auto resized() -> void override;

protected:
    virtual auto refreshFromSession() -> void = 0;

    /** History (+ optional pending) Tokens → table rows + piano roll. */
    auto presentTokens(std::vector<Token> history, std::vector<Token> pending = {}) -> void;

    auto presentOrchestrationNotes(std::vector<OrchestrationNote> notes) -> void;

    auto presentPlaybackRecords(std::vector<PlaybackNoteOnRecord> records) -> void;

    auto setPitchRange(int32_t low, int32_t high) -> void;

    auto setNoteColours(juce::Colour history, juce::Colour pending) -> void;

    AppSession &session;
    NoteIoTableView table;
    PianoRollView pianoRoll;

private:
    auto timerCallback() -> void override;

    static auto rowFromToken(const Token &token) -> NoteIoRow;

    static auto tokenFromPlaybackRecord(const PlaybackNoteOnRecord &record) -> Token;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NoteIoWidget)
};
