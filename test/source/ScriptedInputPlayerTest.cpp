#include <gtest/gtest.h>

#include <mutex>
#include <vector>

#include <VirtualOrch/AppSession.h>
#include <VirtualOrch/ScriptedInputPlayer.h>

namespace virtual_orch_test {

namespace {

/** A two-track file: track 1 = C E G at 0 / 0.1 / 0.2 s (0.1 s each) with a pedal; track 2 = A. */
auto writeTestFile() -> juce::File {
    juce::MidiFile midi;
    midi.setTicksPerQuarterNote(480); // default tempo 120 bpm: 960 ticks per second
    juce::MidiMessageSequence piano, other;
    piano.addEvent(juce::MidiMessage::textMetaEvent(3, "Piano"), 0);
    piano.addEvent(juce::MidiMessage::controllerEvent(1, 64, 127), 0);
    for (int k = 0; k < 3; ++k) {
        piano.addEvent(juce::MidiMessage::noteOn(1, 60 + 2 * k + (k > 0 ? k - 1 : 0), static_cast<juce::uint8>(90)), 96 * k);
        piano.addEvent(juce::MidiMessage::noteOff(1, 60 + 2 * k + (k > 0 ? k - 1 : 0)), 96 * k + 96);
    }
    piano.addEvent(juce::MidiMessage::controllerEvent(1, 64, 0), 300);
    other.addEvent(juce::MidiMessage::noteOn(2, 69, static_cast<juce::uint8>(80)), 48);
    other.addEvent(juce::MidiMessage::noteOff(2, 69), 144);
    other.addEvent(juce::MidiMessage::noteOn(10, 36, static_cast<juce::uint8>(80)), 0); // drums: skipped
    other.addEvent(juce::MidiMessage::noteOff(10, 36), 10);
    piano.updateMatchedPairs();
    other.updateMatchedPairs();
    midi.addTrack(piano);
    midi.addTrack(other);
    auto file = juce::File::createTempFile(".mid");
    juce::FileOutputStream out(file);
    midi.writeTo(out);
    out.flush();
    return file;
}

struct Recorder : OutputProcessor {
    std::mutex m;
    std::vector<juce::MidiMessage> relayed;
    void send(NoteOnEvent) override {}
    void send(NoteOffEvent) override {}
    void relayMidi(const juce::MidiMessage &msg) override {
        std::scoped_lock l(m);
        relayed.push_back(msg);
    }
};

} // namespace

TEST(ScriptedInputPlayer, readsNotesAndControllersInTimeOrder) {
    const auto file = writeTestFile();
    std::vector<juce::MidiMessage> got;
    ScriptedInputPlayer player([&](const juce::String &, const juce::String &, const juce::MidiMessage &m) { got.push_back(m); });
    ASSERT_TRUE(player.load(file).isEmpty());
    ASSERT_EQ(player.getTracks().size(), 2u);
    EXPECT_EQ(player.getTracks()[0].name, "Piano");
    EXPECT_EQ(player.getTracks()[0].noteCount, 3);
    EXPECT_EQ(player.getTracks()[1].noteCount, 1); // the drum hit is not counted

    // All tracks: 3 + 1 notes (on + off) + 2 pedal CCs, no drums, sorted by time.
    const auto &evs = player.getEvents();
    EXPECT_EQ(evs.size(), 10u);
    for (size_t i = 1; i < evs.size(); ++i)
        EXPECT_LE(evs[i - 1].timeSeconds, evs[i].timeSeconds);
    EXPECT_NEAR(player.getLengthSeconds(), 300.0 / 960.0, 1e-6);
    // At 0.1 s the first note-off comes before the second note-on.
    EXPECT_TRUE(evs[3].message.isNoteOff() || evs[3].message.isController() || evs[3].timeSeconds < 0.1);

    player.setTrack(1); // only the A
    EXPECT_EQ(player.getEvents().size(), 2u);
    EXPECT_EQ(player.getEvents()[0].message.getNoteNumber(), 69);
    file.deleteFile();
}

TEST(ScriptedInputPlayer, playsInRealTimeAndReleasesEverythingOnStop) {
    const auto file = writeTestFile();
    std::mutex m;
    std::vector<std::pair<double, juce::MidiMessage>> got;
    ScriptedInputPlayer player([&](const juce::String &id, const juce::String &, const juce::MidiMessage &msg) {
        EXPECT_EQ(id, ScriptedInputPlayer::kScriptedInputIdentifier);
        std::scoped_lock l(m);
        got.push_back({juce::Time::getMillisecondCounterHiRes(), msg});
    });
    ASSERT_TRUE(player.load(file).isEmpty());
    player.setTrack(0);
    const double t0 = juce::Time::getMillisecondCounterHiRes();
    player.play();
    juce::Thread::sleep(500);
    EXPECT_FALSE(player.isPlaying()); // 0.31 s file, played once
    {
        std::scoped_lock l(m);
        ASSERT_EQ(got.size(), 8u);
        // The third note-on is due at 0.2 s.
        int ons = 0;
        for (const auto &[t, msg]: got)
            if (msg.isNoteOn() && ++ons == 3)
                EXPECT_NEAR(t - t0, 200.0, 15.0);
    }

    // Stop mid-note: the held note and the pedal are released.
    got.clear();
    player.play();
    juce::Thread::sleep(50);
    player.stop();
    std::scoped_lock l(m);
    ASSERT_GE(got.size(), 4u);
    EXPECT_TRUE(got[got.size() - 2].second.isNoteOff());
    EXPECT_TRUE(got.back().second.isController());
    EXPECT_EQ(got.back().second.getControllerValue(), 0);
    file.deleteFile();
}

TEST(ScriptedInputPlayer, actsAsThePlayKeyboardOfTheSession) {
    auto gui = juce::ScopedJuceInitialiser_GUI{};
    const auto file = writeTestFile();
    AppSession session;
    auto recorder = std::make_unique<Recorder>();
    auto *rec = recorder.get();
    session.outputProcessor = std::move(recorder);
    session.selectedMidiInputIdentifier = ScriptedInputPlayer::kScriptedInputIdentifier;
    ASSERT_TRUE(session.isScriptedInputSelected());
    ASSERT_TRUE(session.scriptedInput.load(file).isEmpty());
    session.scriptedInput.setTrack(0);
    session.scriptedInput.play();
    juce::Thread::sleep(500);
    session.scriptedInput.stop();
    // Jam mode relays the play keyboard to the monitor bus: all 6 note messages + 2 pedal CCs.
    {
        std::scoped_lock l(rec->m);
        int notes = 0, ccs = 0;
        for (const auto &msg: rec->relayed) {
            notes += msg.isNoteOnOrOff() ? 1 : 0;
            ccs += msg.isController() ? 1 : 0;
        }
        EXPECT_EQ(notes, 6);
        EXPECT_EQ(ccs, 2);
        rec->relayed.clear();
    }

    // Not selected as the MIDI Input: the same file reaches nothing.
    session.selectedMidiInputIdentifier = "some-device";
    session.scriptedInput.play();
    juce::Thread::sleep(400);
    session.scriptedInput.stop();
    std::scoped_lock l(rec->m);
    EXPECT_TRUE(rec->relayed.empty());
    file.deleteFile();
}

} // namespace virtual_orch_test
