#include <gtest/gtest.h>

#include <VirtualOrch/SessionLog.h>

namespace virtual_orch_test {

TEST(SessionLog, isOffUntilStartedSoTestsWriteNothing) {
    EXPECT_FALSE(SessionLog::isRunning());
    SessionLog::add("RT", "ignored"); // no file, no crash
    EXPECT_EQ(SessionLog::currentFile(), juce::File());
}

TEST(SessionLog, describesMidiWithRawBytes) {
    const auto text = SessionLog::describeMidi(juce::MidiMessage::noteOn(1, 60, static_cast<juce::uint8>(100)));
    EXPECT_TRUE(text.startsWith("90 3C 64")) << text;
    EXPECT_TRUE(text.containsIgnoreCase("note on")) << text;
}

TEST(SessionLog, skipsOnlyClockAndActiveSensing) {
    EXPECT_TRUE(SessionLog::isSkippedMidi(juce::MidiMessage::midiClock()));
    EXPECT_TRUE(SessionLog::isSkippedMidi(juce::MidiMessage(0xfe)));
    EXPECT_FALSE(SessionLog::isSkippedMidi(juce::MidiMessage::noteOn(1, 60, static_cast<juce::uint8>(1))));
    EXPECT_FALSE(SessionLog::isSkippedMidi(juce::MidiMessage::controllerEvent(1, 64, 127)));
    EXPECT_FALSE(SessionLog::isSkippedMidi(juce::MidiMessage::quarterFrame(0, 1)));
}

} // namespace virtual_orch_test
