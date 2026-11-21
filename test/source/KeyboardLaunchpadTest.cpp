#include <gtest/gtest.h>

#include <set>

#include <VirtualOrch/AppSession.h>
#include <VirtualOrch/KeyboardLaunchpad.h>
#include <VirtualOrch/TemporalDensity.h>

namespace virtual_orch_test {

using KeyboardLaunchpad::Target;
using KeyboardLaunchpad::targetFor;

TEST(KeyboardLaunchpad, mapsEveryButtonTheAppUsesToItsOwnKey) {
    // Every grid button with an action; instrument keys name your side (cols 0-3) only.
    std::set<std::pair<int, int>> wanted;
    for (int c = 0; c < 3; ++c) wanted.insert({LaunchpadGrid::kModeRow, c});
    for (int c = 0; c < 7; ++c) wanted.insert({LaunchpadGrid::kTdRow, c});
    wanted.insert({LaunchpadGrid::kTdRow, LaunchpadGrid::kSideCol});
    const int perRow[LaunchpadGrid::kInstrumentRowCount] = {3, 4, 4, 1};
    for (int r = 0; r < LaunchpadGrid::kInstrumentRowCount; ++r)
        for (int c = 0; c < perRow[r]; ++c)
            wanted.insert({LaunchpadGrid::kInstrumentRowFirst + r, LaunchpadGrid::kUserColFirst + c});
    wanted.insert({LaunchpadGrid::kFlushRow, LaunchpadGrid::kSideCol});
    wanted.insert({LaunchpadGrid::kResetRow, LaunchpadGrid::kSideCol});
    for (int c = 0; c < 3; ++c) wanted.insert({LaunchpadGrid::kRegisterRow, c});
    wanted.insert({LaunchpadGrid::kRegisterRow, LaunchpadGrid::kSideCol});
    for (int c = 0; c < 4; ++c) wanted.insert({LaunchpadGrid::kAccompRow, LaunchpadGrid::kPolyphonyColFirst + c});
    for (int c = 0; c < LaunchpadGrid::kPresetCount; ++c) wanted.insert({LaunchpadGrid::kPresetRow, c});

    std::set<std::pair<int, int>> mapped;
    std::set<int> scenes;
    std::set<int> keys;
    for (const auto &b: KeyboardLaunchpad::bindings()) {
        EXPECT_TRUE(keys.insert(b.key).second) << "key used twice: " << b.key;
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
    constexpr int kStrings = LaunchpadGrid::kInstrumentRowFirst;
    EXPECT_EQ(targetFor('1'), (Target{LaunchpadGrid::kTdRow, 0, -1}));
    EXPECT_EQ(targetFor('7'), (Target{LaunchpadGrid::kTdRow, 6, -1}));
    EXPECT_EQ(targetFor('8'), (Target{LaunchpadGrid::kTdRow, LaunchpadGrid::kSideCol, -1}));
    EXPECT_EQ(targetFor('9'), (Target{LaunchpadGrid::kRegisterRow, 0, -1}));
    EXPECT_EQ(targetFor('='), (Target{LaunchpadGrid::kRegisterRow, LaunchpadGrid::kSideCol, -1}));
    EXPECT_EQ(targetFor('q'), (Target{kStrings, 0, -1})); // lower case too
    EXPECT_EQ(targetFor('F'), (Target{kStrings + 1, 3, -1}));   // bassoon
    EXPECT_EQ(targetFor('V'), (Target{kStrings + 2, 3, -1}));   // tuba
    EXPECT_EQ(targetFor('T'), (Target{kStrings + 3, 0, -1}));   // timpani
    EXPECT_EQ(targetFor(juce::KeyPress::returnKey),
              (Target{LaunchpadGrid::kFlushRow, LaunchpadGrid::kSideCol, -1}));
    EXPECT_EQ(targetFor(juce::KeyPress::backspaceKey),
              (Target{LaunchpadGrid::kResetRow, LaunchpadGrid::kSideCol, -1}));
    EXPECT_EQ(targetFor('U'), (Target{LaunchpadGrid::kModeRow, LaunchpadGrid::kJamCol, -1}));
    EXPECT_EQ(targetFor('.'), (Target{LaunchpadGrid::kAccompRow, LaunchpadGrid::kPolyphonyColFirst + 3, -1}));
    EXPECT_EQ(targetFor(juce::KeyPress::F5Key), (Target{LaunchpadGrid::kPresetRow, 0, -1}));
    EXPECT_EQ(targetFor(juce::KeyPress::F12Key), (Target{LaunchpadGrid::kPresetRow, 7, -1}));
    EXPECT_EQ(targetFor(juce::KeyPress::F1Key), (Target{-1, -1, 0}));
    EXPECT_FALSE(targetFor('R').has_value()); // the unmapped pad after tremolo
    EXPECT_FALSE(targetFor('P').has_value());
    EXPECT_EQ(KeyboardLaunchpad::labelForTarget({kStrings, 0, -1}), "Q");
    EXPECT_EQ(KeyboardLaunchpad::labelForTarget({kStrings, LaunchpadGrid::kModelColFirst, -1}), "Q")
        << "the model's pad shows the same key (it presses it in Jam)";
    EXPECT_EQ(KeyboardLaunchpad::labelForTarget({-1, -1, 1}), "F2");
    EXPECT_EQ(KeyboardLaunchpad::labelForTarget({LaunchpadGrid::kPresetRow, 2, -1}), "F7");
}

TEST(KeyboardLaunchpad, aKeyDoesWhatItsButtonDoes) {
    auto gui = juce::ScopedJuceInitialiser_GUI{};
    AppSession session;
    session.keyboardLaunchpadEnabled = true;
    auto &grid = session.midiInputProcess.getLaunchpadGrid();
    juce::Component dummy;
    auto &listener = session.keyboardLaunchpad;
    constexpr int kStrings = LaunchpadGrid::kInstrumentRowFirst;
    constexpr int kModelEns = LaunchpadGrid::kModelColFirst;

    // Jam (the start mode): Q toggles the model's String Ensemble pad, exactly like pressing it.
    ASSERT_EQ(grid.controlledTarget(), InstrumentUpdateTarget::Model);
    ASSERT_TRUE(grid.isInstrumentPad(kStrings, kModelEns));
    EXPECT_FALSE(grid.getGridState(kStrings, kModelEns));
    EXPECT_TRUE(listener.keyPressed(juce::KeyPress('q'), &dummy));
    EXPECT_TRUE(grid.getGridState(kStrings, kModelEns));
    EXPECT_FALSE(grid.getGridState(kStrings, 0));
    EXPECT_EQ(session.orchestrationTransformer.modelInstruments.count(6), 1u);

    // A held key (auto-repeat) does not press again; once released it does.
    EXPECT_TRUE(listener.keyPressed(juce::KeyPress('q'), &dummy));
    EXPECT_TRUE(grid.getGridState(kStrings, kModelEns));
    listener.keyStateChanged(false, &dummy); // nothing is physically down in the test
    juce::Thread::sleep(static_cast<int>(LaunchpadGrid::kBurstMs) + 20);
    EXPECT_TRUE(listener.keyPressed(juce::KeyPress('Q'), &dummy));
    EXPECT_FALSE(grid.getGridState(kStrings, kModelEns));
    listener.keyStateChanged(false, &dummy);

    // Y = Edit: now A presses your Flute pad.
    EXPECT_TRUE(listener.keyPressed(juce::KeyPress('y'), &dummy));
    EXPECT_EQ(session.orchestrationTransformer.getMode(), OrchestrationMode::Edit);
    listener.keyStateChanged(false, &dummy);
    EXPECT_TRUE(listener.keyPressed(juce::KeyPress('a'), &dummy));
    EXPECT_TRUE(grid.getGridState(kStrings + 1, 0));
    EXPECT_FALSE(grid.getGridState(kStrings + 1, LaunchpadGrid::kModelColFirst));
    listener.keyStateChanged(false, &dummy);

    // 4 = TD 8 (manual), 8 = AUTO back on.
    EXPECT_TRUE(listener.keyPressed(juce::KeyPress('4'), &dummy));
    EXPECT_EQ(TemporalDensity::get(), 8);
    listener.keyStateChanged(false, &dummy);
    EXPECT_TRUE(listener.keyPressed(juce::KeyPress('8'), &dummy));
    EXPECT_TRUE(TemporalDensity::isAuto());
    listener.keyStateChanged(false, &dummy);

    // I = Accompaniment mode.
    EXPECT_TRUE(listener.keyPressed(juce::KeyPress('i'), &dummy));
    EXPECT_EQ(session.orchestrationTransformer.getMode(), OrchestrationMode::Accompaniment);
    listener.keyStateChanged(false, &dummy);

    // Unmapped keys and Cmd shortcuts are left to the app; nothing at all while disabled.
    EXPECT_FALSE(listener.keyPressed(juce::KeyPress('p'), &dummy));
    EXPECT_FALSE(listener.keyPressed(juce::KeyPress('q', juce::ModifierKeys::commandModifier, 0), &dummy));
    session.keyboardLaunchpadEnabled = false;
    EXPECT_FALSE(listener.keyPressed(juce::KeyPress('w'), &dummy));
    EXPECT_FALSE(grid.getGridState(kStrings, LaunchpadGrid::kModelColFirst + 1));
}

} // namespace virtual_orch_test
