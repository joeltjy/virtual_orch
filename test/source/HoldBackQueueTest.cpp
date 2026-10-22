#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include <VirtualOrch/Clock.h>
#include <VirtualOrch/HoldBackQueue.h>
#include <VirtualOrch/Metrics.h>
#include <VirtualOrch/OrchestrationTransformer.h>
#include <VirtualOrch/PedalClear.h>
#include <VirtualOrch/orchestration-models/IodPretrained.h>

namespace {

auto note(int32_t onset, int32_t pitch = 60, int32_t durCs = 50) -> Token {
    return Token{onset, static_cast<int32_t>(Vocab::DurOffset) + durCs,
                 static_cast<int32_t>(Vocab::NoteOffset) + pitch};
}

auto clearAt(int32_t t) -> Token {
    return Token{t, static_cast<int32_t>(Vocab::DurOffset), static_cast<int32_t>(Vocab::ClearQueue)};
}

auto times(const std::vector<Token> &tokens) -> std::vector<int32_t> {
    std::vector<int32_t> out;
    for (const auto &t: tokens)
        out.push_back(t.time);
    return out;
}

} // namespace

namespace virtual_orch_test {

TEST(HoldBackQueue, releasesOnlyNotesWithinTheWindowInOnsetOrder) {
    HoldBackQueue q;
    q.accept({note(900), note(300), note(1500), note(1000), note(2500)});
    EXPECT_EQ(times(q.release(0, 100)), (std::vector<int32_t>{}));
    EXPECT_EQ(times(q.release(250, 100)), (std::vector<int32_t>{300}));
    EXPECT_EQ(times(q.release(900, 100)), (std::vector<int32_t>{900, 1000}));
    EXPECT_EQ(q.size(), 2u);
    EXPECT_EQ(times(q.release(2400, 100)), (std::vector<int32_t>{1500, 2500}));
    EXPECT_EQ(q.size(), 0u);
}

TEST(HoldBackQueue, aNoteAlreadyInsideTheWindowIsReleasedAtOnce) {
    HoldBackQueue q;
    q.accept({note(1050)});
    EXPECT_EQ(times(q.release(1000, 100)), (std::vector<int32_t>{1050}));
}

TEST(HoldBackQueue, noClockReleasesEverything) {
    HoldBackQueue q;
    q.accept({note(50000), note(10)});
    EXPECT_EQ(times(q.release(-1)), (std::vector<int32_t>{10, 50000}));
}

TEST(HoldBackQueue, aClearErasesTheHeldNotesItCoversInArrivalOrder) {
    HoldBackQueue q;
    // pedal at 1000: earlier arrivals at or after it never come out, later arrivals do
    EXPECT_EQ(q.accept({note(500), note(1200), note(3000), clearAt(1000), note(1500)}), 2u);
    EXPECT_EQ(times(q.release(100000)), (std::vector<int32_t>{500, 1500}));
}

TEST(HoldBackQueue, aClearAtZeroEmptiesTheQueueBeforeTheRegeneratedNotes) {
    HoldBackQueue q;
    q.accept({note(2000), note(2500)});
    EXPECT_EQ(q.accept({clearAt(0), note(2100)}), 2u);
    EXPECT_EQ(times(q.release(100000)), (std::vector<int32_t>{2100}));
}

TEST(HoldBackQueue, aPedalClearTokenUsesTheSameRule) {
    HoldBackQueue q;
    q.accept({note(2000)});
    q.accept({PedalClear::makeClearToken(1500)});
    EXPECT_EQ(q.size(), 0u);
}

TEST(HoldBackQueue, restampsReachHeldNotes) {
    HoldBackQueue q;
    Token t = note(2000);
    t.noteId = 7;
    t.voiceId = 1;
    q.accept({t});
    EXPECT_EQ(q.applyRestamp({7, 3}), 1);
    EXPECT_EQ(q.release(100000)[0].voiceId, 3);
}

TEST(HoldBackQueue, pausedOrchestrationDropsPastNotesAndKeepsFutureOnes) {
    HoldBackQueue q;
    q.accept({note(100), note(400), note(900)});
    EXPECT_EQ(q.dropPast(500), 2u);
    EXPECT_EQ(times(q.release(100000)), (std::vector<int32_t>{900}));
}

TEST(HoldBackQueue, restsAndSeparatorsAreNotHeld) {
    HoldBackQueue q;
    q.accept({Token{10, static_cast<int32_t>(Vocab::DurOffset) + 5, static_cast<int32_t>(Vocab::Rest)},
              Token{20, static_cast<int32_t>(Vocab::DurOffset), static_cast<int32_t>(Vocab::BarSeparator)}});
    EXPECT_EQ(q.size(), 0u);
}

// Whole transformer: a reduction note 3 s ahead is not orchestrated until 1 s before its onset,
// so an instrument toggled in between decides its instrument.
TEST(HoldBackOrchestration, aToggleBeforeTheWindowDecidesTheInstrument) {
    Metrics metrics;
    Clock clock(metrics);
    OrchestrationTransformer ot;
    IodPretrained model;
    const auto dummy = std::string(VIRTUAL_ORCH_TEST_RESOURCES) + "/iod_dummy.onnx";
    model.init(dummy.c_str());
    ASSERT_TRUE(model.isModelLoaded());
    ot.orchestrationModel = &model;
    ot.clock = &clock;
    ot.setMode(OrchestrationMode::Jam);
    ot.replaceInstrumentSets({}, {7}); // flute only
    clock.startAtTime(0);

    ot.startThread(); // starting clears the incoming queues, so push afterwards
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    // reduction notes 2 s ahead of the clock (onset in cs)
    const auto start = static_cast<int32_t>(clock.getTime());
    for (int i = 0; i < 6; ++i)
        ot.reductionIncoming.push(note(start + 200 + i * 5, 58 + i * 2));
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    EXPECT_EQ(ot.heldCount(), 6u) << "all six notes are still waiting, un-orchestrated";
    OrchestrationNote out;
    EXPECT_FALSE(ot.outputTokenQueue.pull(out)) << "nothing was orchestrated yet";

    // toggle: flute off, oboe on, while the notes are still held
    ot.applyOneInstrumentUpdate({7, InstrumentUpdateTarget::Model, 0});
    ot.applyOneInstrumentUpdate({8, InstrumentUpdateTarget::Model, 1});

    // 1 s before the onsets (about 1.0 s from now) they are released
    std::vector<int32_t> instruments;
    for (int i = 0; i < 80 && instruments.empty(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        while (ot.outputTokenQueue.pull(out))
            instruments.push_back(out.localInstrumentId);
    }
    ot.signalThreadShouldExit();
    ot.stopThread(2000);
    ASSERT_FALSE(instruments.empty());
    for (const auto id: instruments)
        EXPECT_EQ(id, 8) << "orchestrated with the instrument list from after the toggle";
}

} // namespace virtual_orch_test// Edit: the user's key goes in with the provisional 4 s length; its note-off must give every
// orchestra note made from it (the OT output history, which the visualizer draws) the real length.
TEST(EditNoteOff, theOutputHistoryGetsTheKeysRealLength) {
    Metrics metrics;
    Clock clock(metrics);
    OrchestrationTransformer ot;
    IodPretrained model;
    const auto dummy = std::string(VIRTUAL_ORCH_TEST_RESOURCES) + "/iod_dummy.onnx";
    model.init(dummy.c_str());
    ASSERT_TRUE(model.isModelLoaded());
    ot.orchestrationModel = &model;
    ot.clock = &clock;
    ot.setMode(OrchestrationMode::Edit);
    ot.replaceInstrumentSets({7, 9, 6}, {}); // user: flute, clarinet, string ensemble
    clock.startAtTime(0);
    ot.startThread();
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    const auto now = static_cast<int32_t>(clock.getTime());
    Token key{now, static_cast<int32_t>(Vocab::DurOffset) + 400, static_cast<int32_t>(Vocab::NoteOffset) + 72, 80};
    key.keyId = 5;
    // A short key: its note-off reaches the OT before (or with) the note itself.
    Token released = key;
    released.duration = static_cast<int32_t>(Vocab::DurOffset) + 35;
    ot.updatesIncoming.push({key.withInstrument(InstrumentConstants::kReductionInputLocalInstrumentId),
                             released.withInstrument(InstrumentConstants::kReductionInputLocalInstrumentId)});
    ot.midiInputIncoming.push(key.withInstrument(InstrumentConstants::kKeyboardLocalInstrumentId));
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    const auto snap = ot.getDebugSnapshot();
    ot.signalThreadShouldExit();
    ot.stopThread(2000);
    ASSERT_FALSE(snap.outputHistory.empty()) << "the key was orchestrated";
    for (const auto &n: snap.outputHistory) {
        EXPECT_EQ(n.token.keyId, 5);
        EXPECT_EQ(n.token.getRealDuration(), 35) << "instrument " << n.localInstrumentId;
    }
}


