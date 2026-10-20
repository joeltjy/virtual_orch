#include "VirtualOrch/widgets/InstrumentActiveLightWidget.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ui/UiConstants.h"
#include "VirtualOrchBinaryData.h"

#include <algorithm>
#include <memory>
#include <vector>

namespace {

/** How much of a dimmed section's brightness is left (the overlay is black at 1 - this). */
constexpr float kDimmedBrightness = 0.35f;

} // namespace

SeatingPlanView::SeatingPlanView() {
    setOpaque(true);
    const auto plan = juce::ImageFileFormat::loadFrom(
                          VirtualOrchBinaryData::orch_seating_plan_no_perc_png,
                          static_cast<size_t>(VirtualOrchBinaryData::orch_seating_plan_no_perc_pngSize))
                          .convertedToFormat(juce::Image::ARGB);
    const auto mask = juce::ImageFileFormat::loadFrom(
        VirtualOrchBinaryData::orch_seating_plan_sectors_png,
        static_cast<size_t>(VirtualOrchBinaryData::orch_seating_plan_sectors_pngSize));
    if (! plan.isValid())
        return;
    sourceWidth = plan.getWidth();
    sourceHeight = plan.getHeight();
    dimBase = plan.createCopy();
    if (! mask.isValid() || mask.getWidth() != sourceWidth || mask.getHeight() != sourceHeight)
        return;

    // Section of every pixel (mask value = sector + 1) and each section's bounding box.
    std::vector<uint8_t> sectorOf(static_cast<size_t>(sourceWidth) * static_cast<size_t>(sourceHeight), 0);
    std::array<int, kSectors> minX{}, minY{}, maxX{}, maxY{};
    minX.fill(sourceWidth);
    minY.fill(sourceHeight);
    maxX.fill(-1);
    maxY.fill(-1);
    {
        const juce::Image::BitmapData m(mask, juce::Image::BitmapData::readOnly);
        for (int y = 0; y < sourceHeight; ++y)
            for (int x = 0; x < sourceWidth; ++x) {
                const auto id = m.getPixelColour(x, y).getRed();
                if (id == 0 || id > kSectors)
                    continue;
                sectorOf[static_cast<size_t>(y) * static_cast<size_t>(sourceWidth) + static_cast<size_t>(x)] = id;
                const auto i = static_cast<size_t>(id - 1);
                minX[i] = std::min(minX[i], x);
                minY[i] = std::min(minY[i], y);
                maxX[i] = std::max(maxX[i], x);
                maxY[i] = std::max(maxY[i], y);
            }
    }

    for (size_t i = 0; i < static_cast<size_t>(kSectors); ++i) {
        if (maxX[i] < 0)
            continue;
        patchBounds[i] = {minX[i], minY[i], maxX[i] - minX[i] + 1, maxY[i] - minY[i] + 1};
        brightPatch[i] = juce::Image(juce::Image::ARGB, patchBounds[i].getWidth(),
                                     patchBounds[i].getHeight(), true);
    }

    const juce::Image::BitmapData src(plan, juce::Image::BitmapData::readOnly);
    const juce::Image::BitmapData dim(dimBase, juce::Image::BitmapData::readWrite);
    std::array<std::unique_ptr<juce::Image::BitmapData>, kSectors> patchData;
    for (size_t i = 0; i < static_cast<size_t>(kSectors); ++i)
        if (brightPatch[i].isValid())
            patchData[i] = std::make_unique<juce::Image::BitmapData>(
                brightPatch[i], juce::Image::BitmapData::writeOnly);
    for (int y = 0; y < sourceHeight; ++y)
        for (int x = 0; x < sourceWidth; ++x) {
            const auto id = sectorOf[static_cast<size_t>(y) * static_cast<size_t>(sourceWidth) + static_cast<size_t>(x)];
            if (id == 0)
                continue;
            const auto i = static_cast<size_t>(id - 1);
            const auto colour = src.getPixelColour(x, y);
            dim.setPixelColour(x, y, colour.interpolatedWith(juce::Colours::black, 1.0f - kDimmedBrightness)
                                         .withAlpha(colour.getFloatAlpha()));
            patchData[i]->setPixelColour(x - patchBounds[i].getX(), y - patchBounds[i].getY(), colour);
        }
}

auto SeatingPlanView::resized() -> void {
    if (! dimBase.isValid() || getWidth() <= 0 || getHeight() <= 0)
        return;
    dest = juce::RectanglePlacement(juce::RectanglePlacement::centred)
               .appliedTo(juce::Rectangle<float>(static_cast<float>(sourceWidth),
                                                 static_cast<float>(sourceHeight)),
                          getLocalBounds().toFloat())
               .toNearestInt();
    if (dest.isEmpty())
        return;
    const double sx = static_cast<double>(dest.getWidth()) / sourceWidth;
    const double sy = static_cast<double>(dest.getHeight()) / sourceHeight;
    scaledBase = dimBase.rescaled(dest.getWidth(), dest.getHeight(),
                                  juce::Graphics::highResamplingQuality);
    for (size_t i = 0; i < static_cast<size_t>(kSectors); ++i) {
        scaledPatch[i] = {};
        scaledBounds[i] = {};
        if (patchBounds[i].isEmpty())
            continue;
        const int x0 = dest.getX() + juce::roundToInt(patchBounds[i].getX() * sx);
        const int y0 = dest.getY() + juce::roundToInt(patchBounds[i].getY() * sy);
        const int x1 = dest.getX() + juce::roundToInt(patchBounds[i].getRight() * sx);
        const int y1 = dest.getY() + juce::roundToInt(patchBounds[i].getBottom() * sy);
        scaledBounds[i] = {x0, y0, std::max(1, x1 - x0), std::max(1, y1 - y0)};
        scaledPatch[i] = brightPatch[i].rescaled(scaledBounds[i].getWidth(),
                                                 scaledBounds[i].getHeight(),
                                                 juce::Graphics::highResamplingQuality);
    }
}

auto SeatingPlanView::setActive(const std::array<bool, PerformanceActivity::kSectorCount> &next)
    -> void {
    if (next == active)
        return;
    for (size_t i = 0; i < static_cast<size_t>(kSectors); ++i)
        if (next[i] != active[i]) {
            if (scaledBounds[i].isEmpty())
                repaint();
            else
                repaint(scaledBounds[i]);
        }
    active = next;
}

auto SeatingPlanView::paint(juce::Graphics &g) -> void {
    g.fillAll(juce::Colours::white);
    if (! scaledBase.isValid())
        return;
    g.drawImageAt(scaledBase, dest.getX(), dest.getY());
    for (size_t i = 0; i < static_cast<size_t>(kSectors); ++i)
        if (active[i] && scaledPatch[i].isValid())
            g.drawImageAt(scaledPatch[i], scaledBounds[i].getX(), scaledBounds[i].getY());
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
