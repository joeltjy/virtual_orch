#include <gtest/gtest.h>

#include <cstdlib>

#include <VirtualOrch/LaunchpadGrid.h>
#include <VirtualOrch/ScoreLayout.h>
#include <VirtualOrch/ui/UiConstants.h>
#include <VirtualOrch/widgets/ActiveInstrumentsView.h>
#include <VirtualOrch/widgets/InstrumentGroupActivityWidget.h>
#include <VirtualOrch/widgets/ScoreWidget.h>
#include <VirtualOrch/widgets/TdViewerWidget.h>
#include <VirtualOrch/widgets/UserSelectionWidget.h>

namespace virtual_orch_test {

TEST(PerformanceText, theTitleBarGrowsWithTheTextScale) {
    const juce::ScopedJuceInitialiser_GUI gui;
    TdViewerWidget widget;
    widget.setBounds(0, 0, 470, 300);
    const int before = widget.getContentComponent().getY();
    widget.setTextScale(UiConstants::performanceTextScale);
    EXPECT_GT(widget.getContentComponent().getY(), before + 20) << "a taller title bar";
    widget.setTextScale(1.0f);
    EXPECT_EQ(widget.getContentComponent().getY(), before);
}

// Renders the audience views at 1x and the performance scale; with
// VIRTUAL_ORCH_PERFORMANCE_PNG=<path> the sheet is saved for a visual check.
TEST(PerformanceText, rendersTheAudienceViewsLarge) {
    const juce::ScopedJuceInitialiser_GUI gui;
    juce::Image sheet(juce::Image::RGB, 2 * 460, 4 * 300, true);
    juce::Graphics g(sheet);
    for (int col = 0; col < 2; ++col) {
        const float ts = col == 0 ? 1.0f : UiConstants::performanceTextScale;
        const auto put = [&](juce::Component &c, int row) {
            c.setBounds(0, 0, 450, 290);
            const auto img = c.createComponentSnapshot(c.getLocalBounds(), false);
            g.drawImageAt(img, col * 460, row * 300);
        };
        TdBarView td;
        td.textScale = ts;
        td.setValue(4);
        put(td, 0);
        GroupBarsView bars;
        bars.textScale = ts;
        bars.setProportions({0.5f, 0.3f, 0.2f});
        put(bars, 1);
        ConsoleView console;
        console.textScale = ts;
        console.setLines({"TD 4", "Register MEDIUM (G3 to D5)", "Model: Str Ens on", "Model: Pizz on",
                          "Input to the reduction open"});
        put(console, 2);
        ScoreView score;
        score.setTextScale(ts);
        score.setFrame(1000, {{900, 72, false, false}, {900, 48, false, true}, {1060, 74, false, false}},
                       {{900, "Cmaj7"}, {1060, "G7/B"}});
        put(score, 3);
    }
    if (const char *path = std::getenv("VIRTUAL_ORCH_PERFORMANCE_PNG")) {
        juce::File file(path);
        file.deleteFile();
        juce::FileOutputStream out(file);
        juce::PNGImageFormat().writeImageToStream(sheet, out);
    }
}

} // namespace virtual_orch_test
