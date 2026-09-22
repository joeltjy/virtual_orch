#include "VirtualOrch/widgets/NoteIoWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/InstrumentUiColours.h"
#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/NoteWindow.h"
#include "VirtualOrch/ui/UiConstants.h"

namespace {

constexpr float noteIoTableWidthFraction = 1.0f / 3.0f;

auto coloursForLocalIds(const std::vector<int32_t> &localIds) -> std::vector<juce::Colour> {
    std::vector<juce::Colour> colours;
    colours.reserve(localIds.size());
    for (const auto id: localIds)
        colours.push_back(InstrumentUiColours::forLocalInstrument(id));
    return colours;
}

} // namespace

NoteIoWidget::NoteIoWidget(AppSession &sessionIn,
                           juce::String titleText,
                           juce::String typeId,
                           bool showChannelColumn)
    : WorkspaceWidget(std::move(titleText), std::move(typeId)),
      session(sessionIn),
      table(showChannelColumn),
      pianoRoll(sessionIn.clock) {
    getContentComponent().addAndMakeVisible(table);
    getContentComponent().addAndMakeVisible(pianoRoll);
    startTimer(UiConstants::noteIoWidgetTimerIntervalMs);
}

NoteIoWidget::~NoteIoWidget() {
    stopTimer();
}

auto NoteIoWidget::resized() -> void {
    WorkspaceWidget::resized();
    auto area = getContentComponent().getLocalBounds();
    // Strict 1/3–2/3 split so the roll never collapses to zero width.
    const int tableWidth = juce::jmax(1, static_cast<int>(area.getWidth() * noteIoTableWidthFraction));
    table.setBounds(area.removeFromLeft(tableWidth));
    pianoRoll.setBounds(area);
}

auto NoteIoWidget::timerCallback() -> void {
    refreshFromSession();
}

auto NoteIoWidget::windowCutoffCs() const -> int32_t {
    return NoteWindow::cutoffCs(static_cast<int32_t>(session.clock.getTime()));
}

auto NoteIoWidget::setPitchRange(int32_t low, int32_t high) -> void {
    pianoRoll.setPitchRange(low, high);
}

auto NoteIoWidget::setNoteColours(juce::Colour history, juce::Colour pending) -> void {
    pianoRoll.setNoteColours(history, pending);
}

auto NoteIoWidget::rowFromToken(const Token &token) -> NoteIoRow {
    NoteIoRow row;
    row.onset = token.time;
    row.durationCs = token.getRealDuration();
    row.pitch = token.getPitch();
    row.velocity = token.velocity > 0 ? token.velocity : 100;
    if (token.note != static_cast<int32_t>(Vocab::ClearQueue)
        && token.note != static_cast<int32_t>(Vocab::BarSeparator)
        && token.note != static_cast<int32_t>(Vocab::Rest)
        && token.note >= static_cast<int32_t>(Vocab::NoteOffset))
        row.localInstrumentId = token.getInstrument();
    return row;
}

auto NoteIoWidget::tokenFromPlaybackRecord(const PlaybackNoteOnRecord &record) -> Token {
    return Token{
        .time = record.onsetCs,
        .duration = static_cast<int32_t>(Vocab::DurOffset + juce::jmax(0, record.durationCs)),
        .note = static_cast<int32_t>(Vocab::NoteOffset
                                    + Config::MaxPitch
                                          * juce::jmax(0, record.localInstrumentId)
                                    + record.pitch),
        .velocity = record.velocity,
    };
}

auto NoteIoWidget::presentTokens(std::vector<Token> history, std::vector<Token> pending) -> void {
    std::vector<NoteIoRow> rows;
    rows.reserve(history.size() + pending.size());
    for (const auto &token: history)
        rows.push_back(rowFromToken(token));
    for (const auto &token: pending)
        rows.push_back(rowFromToken(token));
    table.setRows(std::move(rows));
    pianoRoll.setNotes(std::move(history));
    pianoRoll.setPendingNotes(std::move(pending));
}

auto NoteIoWidget::presentOrchestrationNotes(std::vector<OrchestrationNote> notes) -> void {
    std::vector<NoteIoRow> rows;
    std::vector<Token> tokens;
    std::vector<int32_t> localIds;
    rows.reserve(notes.size());
    tokens.reserve(notes.size());
    localIds.reserve(notes.size());
    for (const auto &note: notes) {
        NoteIoRow row;
        row.onset = note.token.time;
        row.durationCs = note.token.getRealDuration();
        row.pitch = note.token.getPitch();
        row.velocity = note.velocity > 0 ? note.velocity : note.token.velocity;
        row.localInstrumentId = note.localInstrumentId;
        rows.push_back(row);
        tokens.push_back(note.token);
        localIds.push_back(note.localInstrumentId);
    }
    auto colours = coloursForLocalIds(localIds);
    table.setRows(std::move(rows));
    pianoRoll.setNotes(std::move(tokens), std::move(colours));
    pianoRoll.setPendingNotes({});
}

auto NoteIoWidget::presentPlaybackRecords(std::vector<PlaybackNoteOnRecord> records) -> void {
    std::vector<NoteIoRow> rows;
    std::vector<Token> tokens;
    std::vector<int32_t> localIds;
    rows.reserve(records.size());
    tokens.reserve(records.size());
    localIds.reserve(records.size());
    for (const auto &record: records) {
        NoteIoRow row;
        row.onset = record.onsetCs;
        row.durationCs = record.durationCs;
        row.pitch = record.pitch;
        row.velocity = record.velocity;
        row.localInstrumentId = record.localInstrumentId;
        row.channel = record.channel;
        rows.push_back(row);
        tokens.push_back(tokenFromPlaybackRecord(record));
        localIds.push_back(record.localInstrumentId);
    }
    auto colours = coloursForLocalIds(localIds);
    table.setRows(std::move(rows));
    pianoRoll.setNotes(std::move(tokens), std::move(colours));
    pianoRoll.setPendingNotes({});
}
