#include "VirtualOrch/widgets/InstrumentActiveLightWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"
#include "VirtualOrchBinaryData.h"

namespace {

/** How much of a dimmed section's brightness is left (the overlay is black at 1 - this). */
constexpr float kDimmedBrightness = 0.35f;

} // namespace

SeatingPlanView::SeatingPlanView() {
    setOpaque(true);
    plan = juce::ImageFileFormat::loadFrom(VirtualOrchBinaryData::orch_seating_plan_no_perc_png,
                                           static_cast<size_t>(
                                               VirtualOrchBinaryData::orch_seating_plan_no_perc_pngSize));
    const auto mask = juce::ImageFileFormat::loadFrom(
        VirtualOrchBinaryData::orch_seating_plan_sectors_png,
        static_cast<size_t>(VirtualOrchBinaryData::orch_seating_plan_sectors_pngSize));
    if (! mask.isValid())
        return;
    maskWidth = mask.getWidth();
    maskHeight = mask.getHeight();
    sectorOfPixel.assign(static_cast<size_t>(maskWidth) * static_cast<size_t>(maskHeight), 0);
    const juce::Image::BitmapData data(mask, juce::Image::BitmapData::readOnly);
    for (int y = 0; y < maskHeight; ++y)
        for (int x = 0; x < maskWidth; ++x)
            sectorOfPixel[static_cast<size_t>(y) * static_cast<size_t>(maskWidth)
                          + static_cast<size_t>(x)] = data.getPixelColour(x, y).getRed();
    rebuildOverlay();
}

auto SeatingPlanView::setActive(const std::array<bool, PerformanceActivity::kSectorCount> &next)
    -> void {
    if (next == active)
        return;
    active = next;
    rebuildOverlay();
    repaint();
}

auto SeatingPlanView::rebuildOverlay() -> void {
    if (sectorOfPixel.empty())
        return;
    overlay = juce::Image(juce::Image::ARGB, maskWidth, maskHeight, true);
    const juce::Image::BitmapData data(overlay, juce::Image::BitmapData::writeOnly);
    const auto dim = juce::Colours::black.withAlpha(1.0f - kDimmedBrightness);
    for (int y = 0; y < maskHeight; ++y) {
        for (int x = 0; x < maskWidth; ++x) {
            const auto id = sectorOfPixel[static_cast<size_t>(y) * static_cast<size_t>(maskWidth)
                                          + static_cast<size_t>(x)];
            if (id != 0 && id <= PerformanceActivity::kSectorCount
                && ! active[static_cast<size_t>(id - 1)])
                data.setPixelColour(x, y, dim);
        }
    }
}

auto SeatingPlanView::paint(juce::Graphics &g) -> void {
    g.fillAll(juce::Colours::white);
    if (! plan.isValid())
        return;
    const auto area = getLocalBounds().toFloat();
    const auto dest = juce::RectanglePlacement(juce::RectanglePlacement::centred)
                          .appliedTo(juce::Rectangle<float>(static_cast<float>(plan.getWidth()),
                                                            static_cast<float>(plan.getHeight())),
                                     area);
    g.setImageResamplingQuality(juce::Graphics::highResamplingQuality);
    g.drawImage(plan, dest);
    if (overlay.isValid())
        g.drawImage(overlay, dest);
}

InstrumentActiveLightWidget::InstrumentActiveLightWidget(AppSession &sessionIn)
    : WorkspaceWidget(UiConstants::workspaceInstrumentActiveLightTitle,
                      UiConstants::workspaceWidgetTypeInstrumentActiveLight),
      session(sessionIn) {
    getContentComponent().addAndMakeVisible(view);
    startTimerHz(20);
    timerCallback();
}

InstrumentActiveLightWidget::~InstrumentActiveLightWidget() {
    stopTimer();
}

auto InstrumentActiveLightWidget::resized() -> void {
    WorkspaceWidget::resized();
    view.setBounds(getContentComponent().getLocalBounds());
}

auto InstrumentActiveLightWidget::timerCallback() -> void {
    view.setActive(session.activityLog.activeSectors(juce::Time::currentTimeMillis()));
}
