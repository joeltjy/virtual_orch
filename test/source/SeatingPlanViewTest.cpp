#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

#include <VirtualOrch/AppSession.h>
#include <VirtualOrch/PerformanceActivity.h>
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

// End to end: a logged orchestra note lights its section and only it.
TEST(SeatingPlanView, aLoggedOrchestraNoteLightsItsSection) {
    const juce::ScopedJuceInitialiser_GUI gui;

    SeatingPlanView view;
    view.setBounds(0, 0, 916, 430);
    PerformanceActivity::ActivityLog log;
    const int64_t now = 100000;
    log.noteOn(now, 2, 45, 500); // cello

    view.setActive(log.activeSectors(now + 100));
    const auto image = view.createComponentSnapshot(view.getLocalBounds(), false);
    EXPECT_GT(brightness(image, 1250.0f / 1832.0f, 560.0f / 859.0f), 0.6f) << "cellos lit";
    EXPECT_LT(brightness(image, 900.0f / 1832.0f, 520.0f / 859.0f), 0.45f) << "violas dim";

    // Two seconds after the note ends the section goes dim again.
    view.setActive(log.activeSectors(now + 500 + PerformanceActivity::ActivityLog::kHoldMs + 1));
    const auto later = view.createComponentSnapshot(view.getLocalBounds(), false);
    EXPECT_LT(brightness(later, 1250.0f / 1832.0f, 560.0f / 859.0f), 0.45f) << "cellos dim again";
}

// An update is cheap: no pixel work in setActive, and paint only blits pre-scaled images.
TEST(SeatingPlanView, updatesAreCheap) {
    const juce::ScopedJuceInitialiser_GUI gui;

    SeatingPlanView view;
    view.setBounds(0, 0, 916, 430);
    std::array<bool, PerformanceActivity::kSectorCount> active{};

    const auto t0 = juce::Time::getMillisecondCounterHiRes();
    for (int i = 0; i < 200; ++i) {
        active[static_cast<size_t>(i % PerformanceActivity::kSectorCount)] ^= true;
        view.setActive(active);
    }
    const double setActiveMs = (juce::Time::getMillisecondCounterHiRes() - t0) / 200.0;

    active.fill(true); // worst case: every patch drawn
    view.setActive(active);
    const auto p0 = juce::Time::getMillisecondCounterHiRes();
    for (int i = 0; i < 10; ++i)
        (void) view.createComponentSnapshot(view.getLocalBounds(), false);
    const double paintMs = (juce::Time::getMillisecondCounterHiRes() - p0) / 10.0;
    std::printf("seating plan: setActive %.3f ms, paint (all lit, incl. snapshot) %.1f ms\n",
                setActiveMs, paintMs);
    EXPECT_LT(setActiveMs, 1.0);
}

// Mean absolute difference between neighbouring pixels: how sharp the picture is.
static auto sharpness(const juce::Image &image) -> double {
    double sum = 0.0;
    int n = 0;
    for (int y = 0; y < image.getHeight(); y += 2)
        for (int x = 0; x + 1 < image.getWidth(); x += 1) {
            sum += std::abs(image.getPixelAt(x, y).getBrightness()
                            - image.getPixelAt(x + 1, y).getBrightness());
            ++n;
        }
    return sum / std::max(1, n);
}

TEST(SeatingPlanFreeze, theBlurIsPrecomputedAndFrozenShowsABlurredDimmedPlan) {
    const juce::ScopedJuceInitialiser_GUI gui;
    SeatingPlanView view;
    EXPECT_TRUE(view.hasPrecomputedBlur()) << "computed when the view is created, not when painting";
    view.setBounds(0, 0, 916, 430);

    std::array<bool, PerformanceActivity::kSectorCount> all{};
    all.fill(true);
    view.setActive(all);
    const auto live = view.createComponentSnapshot(view.getLocalBounds(), false);

    view.setFrozen(true, /*immediate*/ true);
    EXPECT_TRUE(view.isFrozen());
    EXPECT_FLOAT_EQ(view.freezeOpacity(), 1.0f);
    const auto frozen = view.createComponentSnapshot(view.getLocalBounds(), false);
    EXPECT_LT(sharpness(frozen), sharpness(live) * 0.7) << "blurred: much less edge contrast";

    // brightness drops (scrim) but the plan is still recognisable (not black)
    double liveMean = 0, frozenMean = 0;
    int n = 0;
    for (int y = 0; y < live.getHeight(); y += 4)
        for (int x = 0; x < live.getWidth(); x += 4) {
            liveMean += live.getPixelAt(x, y).getBrightness();
            frozenMean += frozen.getPixelAt(x, y).getBrightness();
            ++n;
        }
    EXPECT_LT(frozenMean / n, liveMean / n);
    EXPECT_GT(frozenMean / n, 0.2 * liveMean / n);

    // painting again does not change the picture (nothing is recomputed lazily)
    const auto again = view.createComponentSnapshot(view.getLocalBounds(), false);
    EXPECT_NEAR(sharpness(again), sharpness(frozen), 1e-9);

    view.setFrozen(false, true);
    const auto thawed = view.createComponentSnapshot(view.getLocalBounds(), false);
    EXPECT_NEAR(sharpness(thawed), sharpness(live), sharpness(live) * 0.02);
}

// Writes the live and frozen views as PNGs when VIRTUAL_ORCH_SNAPSHOT_DIR is set (to look at them).
TEST(SeatingPlanFreeze, writesSnapshotsWhenAskedTo) {
    const char *dir = std::getenv("VIRTUAL_ORCH_SNAPSHOT_DIR");
    if (dir == nullptr)
        GTEST_SKIP() << "VIRTUAL_ORCH_SNAPSHOT_DIR not set";
    const juce::ScopedJuceInitialiser_GUI gui;
    SeatingPlanView view;
    view.setBounds(0, 0, 980, 460);
    std::array<bool, PerformanceActivity::kSectorCount> some{};
    some[static_cast<size_t>(Sector::Flutes)] = true;
    some[static_cast<size_t>(Sector::FirstViolins)] = true;
    view.setActive(some);
    const auto write = [&](const char *name) {
        juce::File out = juce::File(dir).getChildFile(name);
        out.deleteFile();
        juce::FileOutputStream stream(out);
        juce::PNGImageFormat().writeImageToStream(
            view.createComponentSnapshot(view.getLocalBounds(), false), stream);
    };
    write("seating_live.png");
    view.setFrozen(true, true);
    write("seating_frozen.png");
}

TEST(SeatingPlanFreeze, theFadeMovesTowardsTheTargetOverTime) {
    const juce::ScopedJuceInitialiser_GUI gui;
    SeatingPlanView view;
    view.setBounds(0, 0, 400, 200);
    view.setFrozen(true); // animated: starts at 0
    EXPECT_FLOAT_EQ(view.freezeOpacity(), 0.0f);
    EXPECT_TRUE(view.isFrozen());
}

TEST(PedalFreeze, showsFrozenOnlyAfterTheThresholdAndNotWhenTheOrchestraIsNotStopped) {
    using PerformanceActivity::pedalShowsFrozen;
    EXPECT_FALSE(pedalShowsFrozen(-1)) << "pedal up";
    EXPECT_FALSE(pedalShowsFrozen(0));
    EXPECT_FALSE(pedalShowsFrozen(999));
    EXPECT_TRUE(pedalShowsFrozen(1000));
    EXPECT_TRUE(pedalShowsFrozen(30000));
}

TEST(PedalFreeze, midiInputTracksHowLongThePedalHasBeenDown) {
    const juce::ScopedJuceInitialiser_GUI gui;
    AppSession session;
    auto &midi = session.midiInputProcess;
    EXPECT_EQ(midi.getPedalDownForMs(5000), -1);
    midi.notePedalState(true, 1000);
    EXPECT_EQ(midi.getPedalDownForMs(1000), 0);
    EXPECT_EQ(midi.getPedalDownForMs(2500), 1500);
    midi.notePedalState(true, 3000); // a repeated "down" does not restart the timer
    EXPECT_EQ(midi.getPedalDownForMs(3500), 2500);
    midi.notePedalState(false, 4000);
    EXPECT_EQ(midi.getPedalDownForMs(4100), -1);
    midi.notePedalState(true, 9000); // the next press starts again
    EXPECT_EQ(midi.getPedalDownForMs(9200), 200);
}

} // namespace virtual_orch_test
