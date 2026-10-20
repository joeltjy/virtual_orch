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
    precomputeBlur();
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

auto SeatingPlanView::precomputeBlur() -> void {
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

    if (freezeAlpha <= 0.001f || ! blurred.isValid())
        return;

    // Frozen: the pre-blurred plan fades in over the orchestra, dimmed a little, with the reason.
    g.setOpacity(freezeAlpha);
    g.drawImage(blurred, dest);
    g.setColour(juce::Colours::black.withAlpha(0.38f * freezeAlpha));
    g.fillRect(dest);

    const float titleHeight = std::clamp(dest.getHeight() * 0.13f, 14.0f, 56.0f);
    g.setColour(juce::Colours::white.withAlpha(freezeAlpha));
    g.setFont(juce::Font(titleHeight, juce::Font::bold));
    g.drawText("Pianist soloing", dest.withTrimmedBottom(titleHeight * 0.6f),
               juce::Justification::centred);
    g.setColour(juce::Colours::white.withAlpha(0.85f * freezeAlpha));
    g.setFont(juce::Font(std::max(11.0f, titleHeight * 0.38f)));
    g.drawText("orchestra frozen while the pedal is down",
               dest.withTrimmedTop(dest.getHeight() * 0.5f + titleHeight * 0.4f),
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
