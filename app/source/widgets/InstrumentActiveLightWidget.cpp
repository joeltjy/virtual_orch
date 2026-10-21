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
    precomputeBlur(plan);
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

SeatingPlanView::~SeatingPlanView() {
    stopTimer();
}

auto SeatingPlanView::setFrozen(bool frozen, bool immediate) -> void {
    freezeTarget = frozen;
    if (immediate) {
        freezeAlpha = frozen ? 1.0f : 0.0f;
        repaint();
        return;
    }
    if (freezeAlpha != (frozen ? 1.0f : 0.0f))
        startTimerHz(30);
}

auto SeatingPlanView::timerCallback() -> void {
    // ~0.3 s fade in, ~0.2 s fade out
    const float step = freezeTarget ? 0.11f : 0.17f;
    const float goal = freezeTarget ? 1.0f : 0.0f;
    freezeAlpha = freezeAlpha < goal ? std::min(goal, freezeAlpha + step)
                                     : std::max(goal, freezeAlpha - step);
    if (freezeAlpha == goal)
        stopTimer();
    repaint();
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

namespace {

/** Separable gaussian blur of an ARGB image (edges clamped). */
auto gaussianBlur(juce::Image &image, float sigma) -> void {
    const int radius = std::max(1, static_cast<int>(std::ceil(sigma * 2.5f)));
    std::vector<float> kernel(static_cast<size_t>(2 * radius + 1));
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        const float v = std::exp(-static_cast<float>(i * i) / (2.0f * sigma * sigma));
        kernel[static_cast<size_t>(i + radius)] = v;
        sum += v;
    }
    for (auto &v: kernel)
        v /= sum;

    const int w = image.getWidth();
    const int h = image.getHeight();
    std::vector<float> channels(static_cast<size_t>(w) * static_cast<size_t>(h) * 4);
    std::vector<float> tmp(channels.size());
    {
        const juce::Image::BitmapData data(image, juce::Image::BitmapData::readOnly);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const auto c = data.getPixelColour(x, y);
                const size_t o = (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) * 4;
                channels[o] = c.getFloatRed();
                channels[o + 1] = c.getFloatGreen();
                channels[o + 2] = c.getFloatBlue();
                channels[o + 3] = c.getFloatAlpha();
            }
    }
    const auto pass = [&](const std::vector<float> &src, std::vector<float> &dst, bool horizontal) {
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                float acc[4] = {0, 0, 0, 0};
                for (int k = -radius; k <= radius; ++k) {
                    const int sx = horizontal ? std::clamp(x + k, 0, w - 1) : x;
                    const int sy = horizontal ? y : std::clamp(y + k, 0, h - 1);
                    const size_t o = (static_cast<size_t>(sy) * static_cast<size_t>(w) + static_cast<size_t>(sx)) * 4;
                    const float wgt = kernel[static_cast<size_t>(k + radius)];
                    for (int c = 0; c < 4; ++c)
                        acc[c] += src[o + static_cast<size_t>(c)] * wgt;
                }
                const size_t o = (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) * 4;
                for (int c = 0; c < 4; ++c)
                    dst[o + static_cast<size_t>(c)] = acc[c];
            }
    };
    pass(channels, tmp, true);
    pass(tmp, channels, false);
    const juce::Image::BitmapData out(image, juce::Image::BitmapData::writeOnly);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const size_t o = (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) * 4;
            out.setPixelColour(x, y, juce::Colour::fromFloatRGBA(channels[o], channels[o + 1],
                                                                  channels[o + 2], channels[o + 3]));
        }
}

} // namespace

auto SeatingPlanView::precomputeBlur(const juce::Image &plan) -> void {
    if (! plan.isValid())
        return;
    // Quarter size is plenty for a blurred picture; it is stretched when drawn.
    blurred = plan.rescaled(std::max(8, plan.getWidth() / 4), std::max(8, plan.getHeight() / 4),
                            juce::Graphics::highResamplingQuality)
                  .convertedToFormat(juce::Image::ARGB);
    gaussianBlur(blurred, static_cast<float>(blurred.getWidth()) / 55.0f);
}

auto SeatingPlanView::paint(juce::Graphics &g) -> void {
    g.fillAll(juce::Colours::white);
    if (! scaledBase.isValid())
        return;
    g.drawImageAt(scaledBase, dest.getX(), dest.getY());
    for (size_t i = 0; i < static_cast<size_t>(kSectors); ++i)
        if (active[i] && scaledPatch[i].isValid())
            g.drawImageAt(scaledPatch[i], scaledBounds[i].getX(), scaledBounds[i].getY());

    if (freezeAlpha <= 0.001f || ! blurred.isValid())
        return;

    // Frozen: the pre-blurred plan fades in over the orchestra, dimmed a little, with the reason.
    const auto area = dest.toFloat();
    g.setImageResamplingQuality(juce::Graphics::highResamplingQuality);
    g.setOpacity(freezeAlpha);
    g.drawImage(blurred, area);
    g.setColour(juce::Colours::black.withAlpha(0.38f * freezeAlpha));
    g.fillRect(area);

    const float titleHeight = std::clamp(area.getHeight() * 0.13f, 14.0f, 56.0f);
    g.setColour(juce::Colours::white.withAlpha(freezeAlpha));
    g.setFont(juce::Font(titleHeight, juce::Font::bold));
    g.drawText("Pianist soloing", area.withTrimmedBottom(titleHeight * 0.6f),
               juce::Justification::centred);
    g.setColour(juce::Colours::white.withAlpha(0.85f * freezeAlpha));
    g.setFont(juce::Font(std::max(11.0f, titleHeight * 0.38f)));
    g.drawText("orchestra frozen while the pedal is down",
               area.withTrimmedTop(area.getHeight() * 0.5f + titleHeight * 0.4f),
               juce::Justification::centredTop);
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
    const auto nowMs = juce::Time::currentTimeMillis();
    view.setActive(session.activityLog.activeSectors(nowMs));
    view.setFrozen(PerformanceActivity::pedalShowsFrozen(
        session.midiInputProcess.getPedalDownForMs(nowMs)));
}
