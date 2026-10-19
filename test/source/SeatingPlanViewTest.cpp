#include <gtest/gtest.h>

#include <VirtualOrch/widgets/InstrumentActiveLightWidget.h>

namespace {

using PerformanceActivity::Sector;

auto brightness(const juce::Image &image, float fx, float fy) -> float {
    return image.getPixelAt(static_cast<int>(fx * static_cast<float>(image.getWidth())),
                            static_cast<int>(fy * static_cast<float>(image.getHeight())))
        .getBrightness();
}

} // namespace

namespace virtual_orch_test {

// Points inside sectors, as fractions of the 1832 x 859 image (empty parts of each fill).
TEST(SeatingPlanView, dimsOnlyTheInactiveSectors) {
    const juce::ScopedJuceInitialiser_GUI gui;

    SeatingPlanView view;
    view.setBounds(0, 0, 916, 430); // half size, same aspect as the image

    std::array<bool, PerformanceActivity::kSectorCount> active{};
    active[static_cast<size_t>(Sector::Cellos)] = true;
    view.setActive(active);

    const auto image = view.createComponentSnapshot(view.getLocalBounds(), false);
    ASSERT_EQ(image.getWidth(), 916);

    const float cellos = brightness(image, 1250.0f / 1832.0f, 560.0f / 859.0f);
    const float violas = brightness(image, 900.0f / 1832.0f, 520.0f / 859.0f);
    const float flutes = brightness(image, 420.0f / 1832.0f, 250.0f / 859.0f);
    const float piano = brightness(image, 700.0f / 1832.0f, 770.0f / 859.0f);
    EXPECT_GT(cellos, 0.6f) << "active sector keeps its colour";
    EXPECT_LT(violas, cellos * 0.5f) << "inactive sector is dimmed";
    EXPECT_LT(flutes, 0.45f);
    EXPECT_LT(piano, 0.45f);

    // Everything on: nothing dimmed.
    active.fill(true);
    view.setActive(active);
    const auto lit = view.createComponentSnapshot(view.getLocalBounds(), false);
    EXPECT_GT(brightness(lit, 900.0f / 1832.0f, 520.0f / 859.0f), 0.6f);
    EXPECT_GT(brightness(lit, 700.0f / 1832.0f, 770.0f / 859.0f), 0.5f);
}

} // namespace virtual_orch_test
