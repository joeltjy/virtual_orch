#include <gtest/gtest.h>

#include <VirtualOrch/MidiReconnect.h>

namespace virtual_orch_test {

namespace {
const juce::StringArray kAll{"14-0", "16-1", "20-0", "32-0"};
const juce::StringArray kKeystationGone{"14-0", "16-1", "32-0"};
// Piano and pedal both on the Keystation, Launchpad on 16-1, no additional piano.
const juce::StringArray kInUse{"20-0", "16-1", "20-0", ""};
} // namespace

TEST(MidiReconnect, aDroppedDeviceIsReportedOnceAsGone) {
    const auto gone = MidiReconnect::disappeared(kAll, kKeystationGone, kInUse);
    ASSERT_EQ(gone.size(), 1);
    EXPECT_EQ(gone[0], "20-0");
    EXPECT_TRUE(MidiReconnect::reappeared(kAll, kKeystationGone, kInUse).isEmpty());
}

TEST(MidiReconnect, aReturningDeviceIsReopenedOnceEvenWhenTwoRolesUseIt) {
    const auto back = MidiReconnect::reappeared(kKeystationGone, kAll, kInUse);
    ASSERT_EQ(back.size(), 1);
    EXPECT_EQ(back[0], "20-0");
}

TEST(MidiReconnect, devicesTheAppDoesNotUseAreIgnored) {
    // Midi Through (14-0) and the mio (32-0, an output here) come and go: nothing to reopen.
    const juce::StringArray before{"16-1", "20-0"};
    EXPECT_TRUE(MidiReconnect::reappeared(before, kAll, kInUse).isEmpty());
    EXPECT_TRUE(MidiReconnect::disappeared(kAll, before, kInUse).isEmpty());
}

TEST(MidiReconnect, aListChangeThatKeepsEveryUsedDeviceReopensNothing) {
    // Plugging in an unrelated device must not interrupt the inputs in use.
    const juce::StringArray more{"14-0", "16-1", "20-0", "32-0", "40-0"};
    EXPECT_TRUE(MidiReconnect::reappeared(kAll, more, kInUse).isEmpty());
}

} // namespace virtual_orch_test
