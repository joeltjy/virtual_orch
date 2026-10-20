#include <gtest/gtest.h>

#include <vector>

#include <VirtualOrch/MusicToken.h>
#include <VirtualOrch/OrchestrationTransformer.h>
#include <VirtualOrch/orchestration-models/InstrumentCombinations.h>

namespace virtual_orch_test {

namespace {

auto keyNote(int32_t time, int32_t realDur, int32_t pitch, int32_t velocity, int32_t keyId) -> Token {
    return Token{.time = time,
                 .duration = static_cast<int32_t>(Vocab::DurOffset) + realDur,
                 .note = static_cast<int32_t>(Vocab::NoteOffset) + pitch,
                 .velocity = velocity,
                 .keyId = keyId};
}

} // namespace

TEST(KeyboardNoteOff, keyIdMatchesDespiteChangedOnsetPitchAndVelocity) {
    const Token source = keyNote(100, 400, 60, 90, 7);
    // Orchestra note from that key: chord-snapped onset, octave up, brass-scaled velocity.
    const Token orchestra = keyNote(102, 400, 72, 70, 7);
    EXPECT_TRUE(tokenMatchesUpdateSource(orchestra, source));
    EXPECT_FALSE(tokenEqualsByOnsetPitch(orchestra, source)); // the old rule missed it
}

TEST(KeyboardNoteOff, differentKeyIdDoesNotMatchEvenWhenFieldsAreEqual) {
    const Token source = keyNote(100, 400, 60, 90, 7);
    EXPECT_FALSE(tokenMatchesUpdateSource(keyNote(100, 400, 60, 90, 8), source));
    EXPECT_FALSE(tokenMatchesUpdateSource(keyNote(100, 400, 60, 90, -1), source));
}

TEST(KeyboardNoteOff, withoutKeyIdTheFieldMatchIsKept) {
    const Token source = keyNote(100, 400, 60, 90, -1);
    EXPECT_TRUE(tokenMatchesUpdateSource(keyNote(100, 400, 60, 90, -1), source));
    EXPECT_FALSE(tokenMatchesUpdateSource(keyNote(101, 400, 60, 90, -1), source));
    EXPECT_FALSE(tokenMatchesUpdateSource(keyNote(100, 400, 60, 80, -1), source));
}

TEST(KeyboardNoteOff, instrumentCombinationsCopiesTheSourceKeyIdToEveryInstrument) {
    // Two input notes; the first expands to violin + flute + tuba, the second to violin.
    const std::vector<int32_t> sampledTokens = {
        0, 10050, 60, 100,
        InstrumentCombinations::groupsTokenForCombo(7),
        InstrumentCombinations::stringsOffset + 1,
        InstrumentCombinations::woodwindOffset + 1,
        InstrumentCombinations::brassOffset + 8,
        InstrumentCombinations::endNote,
        0, 10050, 64, 100,
        InstrumentCombinations::groupsTokenForCombo(1),
        InstrumentCombinations::stringsOffset + 1,
        InstrumentCombinations::endNote,
        InstrumentCombinations::endNote,
        InstrumentCombinations::endNote,
    };

    InstrumentCombinations model;
    const auto notes = model.encodedCombinationToInstruments(sampledTokens, {11, 12});
    ASSERT_EQ(notes.size(), 4u);
    for (size_t i = 0; i < 3; ++i)
        EXPECT_EQ(notes[i].token.keyId, 11) << "note " << i;
    EXPECT_EQ(notes[3].token.keyId, 12);

    // Without source ids (generated reduction notes) nothing is tagged.
    for (const auto &n: model.encodedCombinationToInstruments(sampledTokens))
        EXPECT_EQ(n.token.keyId, -1);
}

TEST(KeyboardNoteOff, orchestrationTransformerForwardsTheKeyIdToPlayback) {
    OrchestrationTransformer ot;
    const Token oldNote =
        keyNote(10, 400, 60, 90, 3).withInstrument(InstrumentConstants::kReductionInputLocalInstrumentId);
    const Token newNote =
        keyNote(10, 35, 60, 90, 3).withInstrument(InstrumentConstants::kReductionInputLocalInstrumentId);
    ot.updatesIncoming.push({oldNote, newNote});
    ot.applyTokenUpdates();

    TokenUpdate forwarded{};
    ASSERT_TRUE(ot.outputDurationUpdates.pull(forwarded));
    EXPECT_EQ(forwarded.oldNote.keyId, 3);
    EXPECT_EQ(forwarded.newNote.getRealDuration(), 35);
}

} // namespace virtual_orch_test
