#include "VirtualOrch/widgets/VocsepOutputWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/ui/UiConstants.h"
#include "VirtualOrch/widgets/NoteIoRow.h"
#include "VirtualOrch/widgets/VoiceIdUiColours.h"

VocsepOutputWidget::VocsepOutputWidget(AppSession &sessionIn)
    : NoteIoWidget(sessionIn,
                   UiConstants::workspaceVocsepOutputTitle,
                   UiConstants::workspaceWidgetTypeVocsepOutput) {
    table.setProgramColumnAsVoiceId(true);
    refreshFromSession();
}

auto VocsepOutputWidget::refreshFromSession() -> void {
    const bool loaded = session.voiceSeparation.isLoaded();
    titleLabel.setText(loaded ? juce::String(UiConstants::workspaceVocsepOutputTitle)
                              : juce::String(UiConstants::workspaceVocsepOutputTitle) + " (not loaded)",
                       juce::dontSendNotification);

    setPitchRange(session.modelConfig.inputLow, session.modelConfig.inputHigh);
    auto tokens = session.activeReduction().getOutputHistorySince(windowCutoffCs());
    std::vector<NoteIoRow> rows;
    std::vector<juce::Colour> colours;
    rows.reserve(tokens.size());
    colours.reserve(tokens.size());
    for (const auto &token: tokens) {
        NoteIoRow row;
        row.onset = token.time;
        row.durationCs = token.getRealDuration();
        row.pitch = token.getPitch();
        row.velocity = token.velocity > 0 ? token.velocity : 100;
        if (token.note != static_cast<int32_t>(Vocab::ClearQueue)
            && token.note != static_cast<int32_t>(Vocab::BarSeparator)
            && token.note != static_cast<int32_t>(Vocab::Rest)
            && token.note >= static_cast<int32_t>(Vocab::NoteOffset))
            row.localInstrumentId = token.voiceId;
        rows.push_back(row);
        colours.push_back(VoiceIdUiColours::forVoiceId(token.voiceId));
    }
    table.setRows(std::move(rows));
    pianoRoll.setNotes(std::move(tokens), std::move(colours));
    pianoRoll.setPendingNotes({});
}
