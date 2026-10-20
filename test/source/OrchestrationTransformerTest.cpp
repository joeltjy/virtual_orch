#include <gtest/gtest.h>

#include <VirtualOrch/OrchestrationTransformer.h>

namespace virtual_orch_test {

TEST(OrchestrationTransformerHelpers, DrainIncomingEmpty) {
    CircularFifo<Token> incoming;
    const auto batch = OrchestrationTransformer::drainIncoming(incoming);
    EXPECT_TRUE(batch.empty());
}

TEST(OrchestrationTransformerHelpers, DrainIncomingPullsAllInOrder) {
    CircularFifo<Token> incoming;
    const Token a{10, Vocab::DurOffset + 1, static_cast<int32_t>(Vocab::NoteOffset + 60)};
    const Token b{20, Vocab::DurOffset + 2, static_cast<int32_t>(Vocab::NoteOffset + 62)};
    const Token c{30, Vocab::DurOffset + 3, static_cast<int32_t>(Vocab::NoteOffset + 64)};
    incoming.push(a);
    incoming.push(b);
    incoming.push(c);

    const auto batch = OrchestrationTransformer::drainIncoming(incoming);
    ASSERT_EQ(batch.size(), 3u);
    EXPECT_TRUE(tokenEquals(batch[0], a));
    EXPECT_TRUE(tokenEquals(batch[1], b));
    EXPECT_TRUE(tokenEquals(batch[2], c));

    const auto secondDrain = OrchestrationTransformer::drainIncoming(incoming);
    EXPECT_TRUE(secondDrain.empty());
}

TEST(OrchestrationTransformerHelpers, AppendToHistory) {
    std::vector<Token> history;
    const Token existing{1, Vocab::DurOffset + 1, static_cast<int32_t>(Vocab::NoteOffset + 40)};
    history.push_back(existing);

    const std::vector<Token> batch{
        {2, Vocab::DurOffset + 2, static_cast<int32_t>(Vocab::NoteOffset + 41)},
        {3, Vocab::DurOffset + 3, static_cast<int32_t>(Vocab::NoteOffset + 42)},
    };
    OrchestrationTransformer::appendToHistory(history, batch);

    ASSERT_EQ(history.size(), 3u);
    EXPECT_TRUE(tokenEquals(history[0], existing));
    EXPECT_TRUE(tokenEquals(history[1], batch[0]));
    EXPECT_TRUE(tokenEquals(history[2], batch[1]));
}

TEST(OrchestrationTransformerHelpers, AppendToHistoryEmptyBatch) {
    std::vector<Token> history{
        {1, Vocab::DurOffset + 1, static_cast<int32_t>(Vocab::NoteOffset + 40)},
    };
    OrchestrationTransformer::appendToHistory(history, {});
    ASSERT_EQ(history.size(), 1u);
}

TEST(OrchestrationTransformerHelpers, TokenWithInstrumentPacksNote) {
    const int32_t pitch = 60;
    const int32_t instrument = 5;
    Token token{100, Vocab::DurOffset + 10, static_cast<int32_t>(Vocab::NoteOffset + pitch)};

    const auto packed = OrchestrationTransformer::tokenWithInstrument(token, instrument);
    EXPECT_EQ(packed.time, 100);
    EXPECT_EQ(packed.duration, Vocab::DurOffset + 10);
    EXPECT_EQ(packed.getPitch(), pitch);
    EXPECT_EQ(packed.getInstrument(), instrument);
    EXPECT_EQ(packed.note,
              static_cast<int32_t>(Vocab::NoteOffset + Config::MaxPitch * instrument + pitch));
}

TEST(OrchestrationTransformerHelpers, SortTokensByTimeThenDuration) {
    std::vector<Token> tokens{
        {20, Vocab::DurOffset + 5, static_cast<int32_t>(Vocab::NoteOffset + 1)},
        {10, Vocab::DurOffset + 9, static_cast<int32_t>(Vocab::NoteOffset + 2)},
        {10, Vocab::DurOffset + 3, static_cast<int32_t>(Vocab::NoteOffset + 3)},
        {30, Vocab::DurOffset + 1, static_cast<int32_t>(Vocab::NoteOffset + 4)},
    };

    OrchestrationTransformer::sortTokensByTimeThenDuration(tokens);

    ASSERT_EQ(tokens.size(), 4u);
    EXPECT_EQ(tokens[0].time, 10);
    EXPECT_EQ(tokens[0].duration, Vocab::DurOffset + 3);
    EXPECT_EQ(tokens[1].time, 10);
    EXPECT_EQ(tokens[1].duration, Vocab::DurOffset + 9);
    EXPECT_EQ(tokens[2].time, 20);
    EXPECT_EQ(tokens[3].time, 30);
}

TEST(OrchestrationTransformerHelpers, ApplyInstrumentUpdatesSetsAndClearsUser) {
    OrchestrationTransformer ot;
    ot.instrumentUpdates.push({0, InstrumentUpdateTarget::User, 1});
    ot.applyInstrumentUpdates();
    EXPECT_TRUE(ot.userInstruments.contains(0));
    EXPECT_TRUE(ot.modelInstruments.empty());

    ot.instrumentUpdates.push({0, InstrumentUpdateTarget::User, 1});
    ot.applyInstrumentUpdates();
    EXPECT_TRUE(ot.userInstruments.contains(0));
    EXPECT_EQ(ot.userInstruments.size(), 1u);

    ot.instrumentUpdates.push({0, InstrumentUpdateTarget::User, 0});
    ot.applyInstrumentUpdates();
    EXPECT_FALSE(ot.userInstruments.contains(0));

    ot.instrumentUpdates.push({0, InstrumentUpdateTarget::User, 0});
    ot.applyInstrumentUpdates();
    EXPECT_TRUE(ot.userInstruments.empty());
}

TEST(OrchestrationTransformerHelpers, ApplyInstrumentUpdatesSetsAndClearsModel) {
    OrchestrationTransformer ot;
    ot.instrumentUpdates.push({3, InstrumentUpdateTarget::Model, 1});
    ot.applyInstrumentUpdates();
    EXPECT_TRUE(ot.modelInstruments.contains(3));
    EXPECT_TRUE(ot.userInstruments.empty());

    ot.instrumentUpdates.push({3, InstrumentUpdateTarget::Model, 0});
    ot.applyInstrumentUpdates();
    EXPECT_FALSE(ot.modelInstruments.contains(3));
}

TEST(OrchestrationTransformerHelpers, ApplyInstrumentUpdatesIndependentTargets) {
    OrchestrationTransformer ot;
    ot.instrumentUpdates.push({5, InstrumentUpdateTarget::User, 1});
    ot.applyInstrumentUpdates();
    ASSERT_TRUE(ot.userInstruments.contains(5));

    ot.instrumentUpdates.push({5, InstrumentUpdateTarget::Model, 1});
    ot.applyInstrumentUpdates();
    EXPECT_TRUE(ot.modelInstruments.contains(5));
    EXPECT_TRUE(ot.userInstruments.contains(5));
    EXPECT_EQ(ot.userInstruments.size(), 1u);

    ot.instrumentUpdates.push({5, InstrumentUpdateTarget::User, 0});
    ot.applyInstrumentUpdates();
    EXPECT_FALSE(ot.userInstruments.contains(5));
    EXPECT_TRUE(ot.modelInstruments.contains(5));
}

TEST(OrchestrationTransformerHelpers, ApplyInstrumentUpdatesIgnoresOutOfRange) {
    OrchestrationTransformer ot;
    ot.instrumentUpdates.push({-1, InstrumentUpdateTarget::User, 0});
    ot.instrumentUpdates.push({128, InstrumentUpdateTarget::Model, 0});
    ot.applyInstrumentUpdates();
    EXPECT_TRUE(ot.userInstruments.empty());
    EXPECT_TRUE(ot.modelInstruments.empty());
}

TEST(OrchestrationTransformerHelpers, DebugSnapshotRoundTrip) {
    OrchestrationTransformer ot;
    const Token pending{10, Vocab::DurOffset + 1, static_cast<int32_t>(Vocab::NoteOffset + 60)};
    const Token historyToken{1, Vocab::DurOffset + 2, static_cast<int32_t>(Vocab::NoteOffset + 40)};

    OrchestrationDebugSnapshot snap;
    snap.midiHistory = {historyToken};
    snap.midiPending = {pending};
    snap.userInstruments = {0, 12};
    snap.modelInstruments = {40};

    ot.publishDebugSnapshot(std::move(snap));
    const auto got = ot.getDebugSnapshot();

    ASSERT_EQ(got.midiHistory.size(), 1u);
    ASSERT_EQ(got.midiPending.size(), 1u);
    EXPECT_TRUE(tokenEquals(got.midiHistory[0], historyToken));
    EXPECT_TRUE(tokenEquals(got.midiPending[0], pending));
    EXPECT_TRUE(got.userInstruments.contains(0));
    EXPECT_TRUE(got.userInstruments.contains(12));
    EXPECT_TRUE(got.modelInstruments.contains(40));
    EXPECT_TRUE(got.conditioningHistory.empty());
    EXPECT_TRUE(got.reductionPending.empty());
}

TEST(OrchestrationTransformerHelpers, DebugSnapshotPreAppendContract) {
    // Mirrors threadRun: publish with history before append; pending = drained batch.
    std::vector<Token> history;
    const Token batchToken{20, Vocab::DurOffset + 3, static_cast<int32_t>(Vocab::NoteOffset + 62)};
    const std::vector<Token> batch{batchToken};

    OrchestrationTransformer ot;
    OrchestrationDebugSnapshot beforeAppend;
    beforeAppend.midiHistory = history;
    beforeAppend.midiPending = batch;
    ot.publishDebugSnapshot(std::move(beforeAppend));

    auto snap = ot.getDebugSnapshot();
    EXPECT_TRUE(snap.midiHistory.empty());
    ASSERT_EQ(snap.midiPending.size(), 1u);
    EXPECT_TRUE(tokenEquals(snap.midiPending[0], batchToken));

    OrchestrationTransformer::appendToHistory(history, batch);

    OrchestrationDebugSnapshot afterAppend;
    afterAppend.midiHistory = history;
    afterAppend.midiPending = {};
    ot.publishDebugSnapshot(std::move(afterAppend));

    snap = ot.getDebugSnapshot();
    ASSERT_EQ(snap.midiHistory.size(), 1u);
    EXPECT_TRUE(tokenEquals(snap.midiHistory[0], batchToken));
    EXPECT_TRUE(snap.midiPending.empty());
}

TEST(OrchestrationTransformerHelpers, SnapNearbyOnsetsAveragesWithin30ms) {
    // 10 ms grid (cs). 3+4+6 → mean 4.333 → rounds to 4; 20 alone.
    Token a{3, Vocab::DurOffset + 10, static_cast<int32_t>(Vocab::NoteOffset + 60)};
    Token b{4, Vocab::DurOffset + 10, static_cast<int32_t>(Vocab::NoteOffset + 64)};
    Token c{6, Vocab::DurOffset + 10, static_cast<int32_t>(Vocab::NoteOffset + 67)};
    Token d{20, Vocab::DurOffset + 10, static_cast<int32_t>(Vocab::NoteOffset + 72)};
    std::vector<Token> tokens{a, b, c, d};
    OrchestrationTransformer::snapNearbyOnsets(tokens);
    EXPECT_EQ(tokens[0].time, 4);
    EXPECT_EQ(tokens[1].time, 4);
    EXPECT_EQ(tokens[2].time, 4);
    EXPECT_EQ(tokens[3].time, 20);
}

TEST(OrchestrationTransformerHelpers, SnapNearbyOnsetsSkipsGapAbove30ms) {
    Token a{0, Vocab::DurOffset + 10, static_cast<int32_t>(Vocab::NoteOffset + 60)};
    Token b{4, Vocab::DurOffset + 10, static_cast<int32_t>(Vocab::NoteOffset + 64)}; // 40 ms gap
    std::vector<Token> tokens{a, b};
    OrchestrationTransformer::snapNearbyOnsets(tokens);
    EXPECT_EQ(tokens[0].time, 0);
    EXPECT_EQ(tokens[1].time, 4);
}

TEST(OrchestrationTransformerHelpers, ApplyTokenUpdatesDrainsQueue) {
    OrchestrationTransformer ot;
    const Token oldNote{10, Vocab::DurOffset + 5, static_cast<int32_t>(Vocab::NoteOffset + 60)};
    const Token newNote{10, Vocab::DurOffset + 20, static_cast<int32_t>(Vocab::NoteOffset + 60)};
    ot.updatesIncoming.push({oldNote, newNote});
    ot.applyTokenUpdates();

    TokenUpdate leftover{};
    EXPECT_FALSE(ot.updatesIncoming.pull(leftover));
    EXPECT_TRUE(ot.getMidiInputHistory().empty());
    EXPECT_TRUE(ot.getConditioningHistory().empty());
}

} // namespace virtual_orch_test
