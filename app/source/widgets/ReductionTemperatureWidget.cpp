#include "VirtualOrch/widgets/ReductionTemperatureWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"

namespace {

constexpr double kTempMin = 0.05;
constexpr double kTempMax = 2.0;
constexpr double kTempStep = 0.05;

auto configureTempSlider(juce::Slider &slider, juce::Label &label, const juce::String &name)
    -> void {
    label.setText(name, juce::dontSendNotification);
    label.setJustificationType(juce::Justification::centredLeft);
    label.setColour(juce::Label::textColourId, UiConstants::workspaceWidgetTitleTextColour);

    slider.setSliderStyle(juce::Slider::LinearHorizontal);
    slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 48, 20);
    slider.setRange(kTempMin, kTempMax, kTempStep);
    slider.setNumDecimalPlacesToDisplay(2);
}

template <typename Dense>
auto applyTemps(Dense &dense, float onset, float duration, float note, float velocity) -> void {
    dense.setOnsetTemperature(onset);
    dense.setDurationTemperature(duration);
    dense.setNoteTemperature(note);
    dense.setVelocityTemperature(velocity);
}

} // namespace

ReductionTemperatureWidget::ReductionTemperatureWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceReductionTemperatureTitle,
                      UiConstants::workspaceWidgetTypeReductionTemperature),
      session(sessionIn) {
    configureTempSlider(onsetSlider, onsetLabel, "Onset");
    configureTempSlider(durationSlider, durationLabel, "Duration");
    configureTempSlider(noteSlider, noteLabel, "Note");
    configureTempSlider(velocitySlider, velocityLabel, "Velocity");

    auto &host = getContentComponent();
    for (juce::Component *c: {static_cast<juce::Component *>(&onsetLabel),
                              static_cast<juce::Component *>(&durationLabel),
                              static_cast<juce::Component *>(&noteLabel),
                              static_cast<juce::Component *>(&velocityLabel),
                              static_cast<juce::Component *>(&onsetSlider),
                              static_cast<juce::Component *>(&durationSlider),
                              static_cast<juce::Component *>(&noteSlider),
                              static_cast<juce::Component *>(&velocitySlider)}) {
        host.addAndMakeVisible(c);
    }

    syncSlidersFromSession();

    auto pushTemps = [this] {
        const auto onset = static_cast<float>(onsetSlider.getValue());
        const auto duration = static_cast<float>(durationSlider.getValue());
        const auto note = static_cast<float>(noteSlider.getValue());
        const auto velocity = static_cast<float>(velocitySlider.getValue());
        applyTemps(session.reductionTransformerV1, onset, duration, note, velocity);
        applyTemps(session.reductionTransformerV2, onset, duration, note, velocity);
    };
    onsetSlider.onValueChange = pushTemps;
    durationSlider.onValueChange = pushTemps;
    noteSlider.onValueChange = pushTemps;
    velocitySlider.onValueChange = pushTemps;
}

auto ReductionTemperatureWidget::syncSlidersFromSession() -> void {
    // Prefer the active dense backend; fall back to V1 defaults.
    if (session.musicModelArch == MusicModelArch::DenseV2) {
        auto &dense = session.reductionTransformerV2;
        onsetSlider.setValue(dense.getOnsetTemperature(), juce::dontSendNotification);
        durationSlider.setValue(dense.getDurationTemperature(), juce::dontSendNotification);
        noteSlider.setValue(dense.getNoteTemperature(), juce::dontSendNotification);
        velocitySlider.setValue(dense.getVelocityTemperature(), juce::dontSendNotification);
        return;
    }
    auto &dense = session.reductionTransformerV1;
    onsetSlider.setValue(dense.getOnsetTemperature(), juce::dontSendNotification);
    durationSlider.setValue(dense.getDurationTemperature(), juce::dontSendNotification);
    noteSlider.setValue(dense.getNoteTemperature(), juce::dontSendNotification);
    velocitySlider.setValue(dense.getVelocityTemperature(), juce::dontSendNotification);
}

auto ReductionTemperatureWidget::resized() -> void {
    WorkspaceWidget::resized();
    auto area = getContentComponent().getLocalBounds().reduced(
        UiConstants::workspaceModeContentPadding);

    const int rowH = UiConstants::workspaceModeRowHeight;
    const int gap = UiConstants::workspaceModeRowGap;
    const int labelW = 72;

    auto placeRow = [&](juce::Label &label, juce::Slider &slider) {
        auto row = area.removeFromTop(rowH);
        label.setBounds(row.removeFromLeft(labelW));
        slider.setBounds(row);
        area.removeFromTop(gap);
    };

    placeRow(onsetLabel, onsetSlider);
    placeRow(durationLabel, durationSlider);
    placeRow(noteLabel, noteSlider);
    placeRow(velocityLabel, velocitySlider);
}
