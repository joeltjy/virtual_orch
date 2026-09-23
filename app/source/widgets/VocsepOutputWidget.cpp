#include "VirtualOrch/widgets/VocsepOutputWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/MusicToken.h"
#include "VirtualOrch/ui/UiConstants.h"
#include "VirtualOrch/widgets/NoteIoRow.h"

namespace {

/** High-contrast categorical palette; voiceId % size cycles. */
constexpr juce::uint32 kVoicePalette[] = {
    0xffe6194b, 0xff3cb44b, 0xffffe119, 0xff4363d8, 0xfff58231, 0xff911eb4, 0xff42d4f4,
    0xfff032e6, 0xffbfef45, 0xfffabed4, 0xff469990, 0xffdcbeff, 0xff9a6324, 0xfffffac8,
    0xff800000, 0xffaaffc3,
};
constexpr int kVoicePaletteSize = static_cast<int>(sizeof(kVoicePalette) / sizeof(kVoicePalette[0]));

auto colourForVoiceId(int32_t voiceId) -> juce::Colour {
    if (voiceId < 0)
        return juce::Colours::grey;
    return juce::Colour(kVoicePalette[voiceId % kVoicePaletteSize]);
}

} // namespace

VocsepOutputWidget::VocsepOutputWidget(AppSession &sessionIn)
    : NoteIoWidget(sessionIn,
                   UiConstants::workspaceVocsepOutputTitle,
                   UiConstants::workspaceWidgetTypeVocsepOutput) {
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
            row.localInstrumentId = token.voiceId >= 0 ? token.voiceId : -1;
        rows.push_back(row);
        colours.push_back(colourForVoiceId(token.voiceId));
    }
    table.setRows(std::move(rows));
    pianoRoll.setNotes(std::move(tokens), std::move(colours));
    pianoRoll.setPendingNotes({});
}
