#include <gtest/gtest.h>

#include <JuceHeader.h>

#include <VirtualOrch/widgets/TdControlWidget.h>

#include <cmath>
#include <cstdlib>

namespace virtual_orch_test {

// Draws the widget's view with two models and 30 s of history; with VIRTUAL_ORCH_TDCONTROL_PNG=<path>
// it is also saved for a visual check.
TEST(TdControlView, rendersTwoModelsAndTheirHistory) {
    const juce::ScopedJuceInitialiser_GUI gui;
    TdControlView view;
    view.setBounds(0, 0, 640, 360);
    for (int i = 0; i <= 240; ++i) {
        const double t = i * 0.125;
        TdController::State w;
        w.target = t < 12.0 ? 4 : 16;
        w.measured = t < 12.0 ? 4.0 + std::sin(t) : 16.0 * (1.0 - std::exp(-(t - 12.0) / 3.0)) + 2.0;
        w.gamma = t < 12.0 ? 0.5 : 6.0 * (1.0 - std::exp(-(t - 12.0) / 3.0));
        w.k = 0.17;
        w.sensitivity = 0.5;
        w.frozen = false;
        w.why = "";
        TdController::State v1 = w;
        v1.frozen = true;
        v1.why = "not drawing";
        view.setRows({{"W", w}, {"v1", v1}}, t);
    }
    const auto image = view.createComponentSnapshot(view.getLocalBounds(), false);
    ASSERT_EQ(image.getWidth(), 640);
    if (const char *path = std::getenv("VIRTUAL_ORCH_TDCONTROL_PNG")) {
        juce::File file(path);
        file.deleteFile();
        juce::FileOutputStream out(file);
        juce::PNGImageFormat().writeImageToStream(image, out);
    }
}

} // namespace virtual_orch_test
