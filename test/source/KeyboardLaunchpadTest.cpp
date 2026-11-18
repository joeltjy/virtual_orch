#include <gtest/gtest.h>

#include <set>

#include <VirtualOrch/AppSession.h>
#include <VirtualOrch/KeyboardLaunchpad.h>
#include <VirtualOrch/TemporalDensity.h>

namespace virtual_orch_test {

using KeyboardLaunchpad::Target;
using KeyboardLaunchpad::targetFor;

TEST(KeyboardLaunchpad, mapsEveryButtonTheAppUsesToItsOwnKey) {
    // Every grid button with an action in LaunchpadGrid / AppSession (main @ 9563268).
    std::set<std::pair<int, int>> wanted;
    for (int c = 0; c < 3; ++c) wanted.insert({LaunchpadGrid::kModeRow, c});
    for (int c = 0; c < 7; ++c) wanted.insert({LaunchpadGrid::kTdRow, c});
    wanted.insert({LaunchpadGrid::kTdRow, LaunchpadGrid::kSideCol});
    for (int c: {0, 1, 2, 4, 5, 6, 7}) {
        wanted.insert({LaunchpadGrid::kUserRowFirst, c});
        wanted.insert({LaunchpadGrid::kModelRowFirst, c});
    }
    for (int c = 0; c < 5; ++c) {
        wanted.insert({LaunchpadGrid::kUserRowFirst + 1, c});
        wanted.insert({LaunchpadGrid::kModelRowFirst + 1, c});
    }
    wanted.insert({LaunchpadGrid::kFlushRow, LaunchpadGrid::kSideCol});
    for (int c = 0; c < 3; ++c) wanted.insert({LaunchpadGrid::kRegisterRow, c});
    wanted.insert({LaunchpadGrid::kRegisterRow, LaunchpadGrid::kSideCol});
    for (int c = 0; c < 4; ++c) wanted.insert({LaunchpadGrid::kAccompRow, c});

    std::set<std::pair<int, int>> mapped;
    std::set<int> scenes;
    std::set<std::pair<int, bool>> keys;
    for (const auto &b: KeyboardLaunchpad::bindings()) {
        EXPECT_TRUE(keys.insert({b.key, b.shift}).second) << "key used twice: " << b.key;
        if (b.target.isScene())
            scenes.insert(b.target.scene);
        else
            EXPECT_TRUE(mapped.insert({b.target.row, b.target.col}).second);
    }
    EXPECT_EQ(mapped, wanted);
    EXPECT_EQ(scenes, (std::set<int>{0, 1, 2, 3}));
    EXPECT_EQ(KeyboardLaunchpad::bindings().size(), wanted.size() + 4);
}

TEST(KeyboardLaunchpad, groupedByFunctionLayout) {
    EXPECT_EQ(targetFor('1', false), (Target{LaunchpadGrid::kTdRow, 0, -1}));
    EXPECT_EQ(targetFor('7', false), (Target{LaunchpadGrid::kTdRow, 6, -1}));
    EXPECT_EQ(targetFor('8', false), (Target{LaunchpadGrid::kTdRow, LaunchpadGrid::kSideCol, -1}));
    EXPECT_EQ(targetFor('9', false), (Target{LaunchpadGrid::kRegisterRow, 0, -1}));
    EXPECT_EQ(targetFor('=', false), (Target{LaunchpadGrid::kRegisterRow, LaunchpadGrid::kSideCol, -1}));
    EXPECT_EQ(targetFor('q', false), (Target{LaunchpadGrid::kModelRowFirst, 0, -1})); // lower case too
    EXPECT_EQ(targetFor('Q', true), (Target{LaunchpadGrid::kUserRowFirst, 0, -1}));
    EXPECT_EQ(targetFor('G', false), (Target{LaunchpadGrid::kModelRowFirst + 1, 4, -1}));
    EXPECT_EQ(targetFor('G', true), (Target{LaunchpadGrid::kUserRowFirst + 1, 4, -1}));
    EXPECT_EQ(targetFor(juce::KeyPress::returnKey, false),
              (Target{LaunchpadGrid::kFlushRow, LaunchpadGrid::kSideCol, -1}));
    EXPECT_EQ(targetFor('X', false), (Target{LaunchpadGrid::kModeRow, LaunchpadGrid::kJamCol, -1}));
    EXPECT_EQ(targetFor('M', false), (Target{LaunchpadGrid::kAccompRow, 3, -1}));
    EXPECT_EQ(targetFor(juce::KeyPress::F1Key, false), (Target{-1, -1, 0}));
    EXPECT_EQ(targetFor(juce::KeyPress::F4Key, true), (Target{-1, -1, 3})); // Shift ignored off letters
    EXPECT_FALSE(targetFor('R', false).has_value()); // the unmapped pad of the instrument row
    EXPECT_FALSE(targetFor('P', false).has_value());
    EXPECT_EQ(KeyboardLaunchpad::labelForTarget({LaunchpadGrid::kUserRowFirst, 0, -1}),
              juce::String(juce::CharPointer_UTF8("\xe2\x87\xa7Q")));
    EXPECT_EQ(KeyboardLaunchpad::labelForTarget({-1, -1, 1}), "F2");
}

TEST(KeyboardLaunchpad, aKeyDoesWhatItsButtonDoes) {
    auto gui = juce::ScopedJuceInitialiser_GUI{};
    AppSession session;
    session.keyboardLaunchpadEnabled = true;
    auto &grid = session.midiInputProcess.getLaunchpadGrid();
    juce::Component dummy;
    auto &listener = session.keyboardLaunchpad;

    // Q toggles the model String Ensemble pad, exactly like pressing it.
    ASSERT_TRUE(grid.isInstrumentPad(LaunchpadGrid::kModelRowFirst, 0));
    EXPECT_FALSE(grid.getGridState(LaunchpadGrid::kModelRowFirst, 0));
    EXPECT_TRUE(listener.keyPressed(juce::KeyPress('q'), &dummy));
    EXPECT_TRUE(grid.getGridState(LaunchpadGrid::kModelRowFirst, 0));
    EXPECT_EQ(session.orchestrationTransformer.modelInstruments.count(6), 1u);

    // A held key (auto-repeat) does not press again; once released it does.
    EXPECT_TRUE(listener.keyPressed(juce::KeyPress('q'), &dummy));
    EXPECT_TRUE(grid.getGridState(LaunchpadGrid::kModelRowFirst, 0));
    listener.keyStateChanged(false, &dummy); // nothing is physically down in the test
    juce::Thread::sleep(static_cast<int>(LaunchpadGrid::kBurstMs) + 20);
    EXPECT_TRUE(listener.keyPressed(juce::KeyPress('Q'), &dummy));
    EXPECT_FALSE(grid.getGridState(LaunchpadGrid::kModelRowFirst, 0));
    listener.keyStateChanged(false, &dummy);

    // Shift+T: the input-row Flute pad.
    EXPECT_TRUE(listener.keyPressed(juce::KeyPress('T', juce::ModifierKeys::shiftModifier, 0), &dummy));
    EXPECT_TRUE(grid.getGridState(LaunchpadGrid::kUserRowFirst, 4));
    listener.keyStateChanged(false, &dummy);

    // 4 = TD 8 (manual), 8 = AUTO back on.
    EXPECT_TRUE(listener.keyPressed(juce::KeyPress('4'), &dummy));
    EXPECT_EQ(TemporalDensity::get(), 8);
    listener.keyStateChanged(false, &dummy);
    EXPECT_TRUE(listener.keyPressed(juce::KeyPress('8'), &dummy));
    EXPECT_TRUE(TemporalDensity::isAuto());
    listener.keyStateChanged(false, &dummy);

    // C = Accompaniment mode.
    EXPECT_TRUE(listener.keyPressed(juce::KeyPress('c'), &dummy));
    EXPECT_EQ(session.orchestrationTransformer.getMode(), OrchestrationMode::Accompaniment);
    listener.keyStateChanged(false, &dummy);

    // Unmapped keys and Cmd shortcuts are left to the app; nothing at all while disabled.
    EXPECT_FALSE(listener.keyPressed(juce::KeyPress('p'), &dummy));
    EXPECT_FALSE(listener.keyPressed(juce::KeyPress('q', juce::ModifierKeys::commandModifier, 0), &dummy));
    session.keyboardLaunchpadEnabled = false;
    EXPECT_FALSE(listener.keyPressed(juce::KeyPress('w'), &dummy));
    EXPECT_FALSE(grid.getGridState(LaunchpadGrid::kModelRowFirst, 1));
}

} // namespace virtual_orch_test
