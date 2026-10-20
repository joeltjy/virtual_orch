#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <VirtualOrch/InstrumentConstants.h>
#include <VirtualOrch/orchestration-models/InstrumentCombinations.h>

namespace virtual_orch_test {

TEST(InstrumentCombinations, TokenLayoutMatchesCheckpointVocab) {
    // ice20_causal_lora_s2 logits width; a mismatch means the offsets are stale and
    // ids like `mask` would fall outside the model's embedding.
    EXPECT_EQ(InstrumentCombinations::expectedVocabSize, 11594);
    EXPECT_EQ(InstrumentCombinations::numStrings + InstrumentCombinations::numWoodwinds
                  + InstrumentCombinations::numBrass + InstrumentCombinations::numOther,
              static_cast<int32_t>(InstrumentConstants::kInstrumentMappings.size()));
}

TEST(InstrumentCombinations, EncodedCombinationToInstruments_ViolinFluteTuba) {
    // onset=0, dur=50, pitch=60, vel=100,
    // groups = strings|woodwinds|brass (7),
    // strings combo bit0 → violin(0),
    // woodwind combo bit0 → flute(7),
    // brass combo bit3 → tuba(14),
    // other = endNote (skip)
    const std::vector<int32_t> sampledTokens = {
        0,
        10050,
        60,
        100,
        InstrumentCombinations::groupsTokenForCombo(7),
        InstrumentCombinations::stringsOffset + 1,
        InstrumentCombinations::woodwindOffset + 1,
        InstrumentCombinations::brassOffset + 8,
        InstrumentCombinations::endNote,
    };

    InstrumentCombinations model;
    const auto notes = model.encodedCombinationToInstruments(sampledTokens);

    ASSERT_EQ(notes.size(), 3u);

    const std::array<int32_t, 3> expectedInstruments = {0, 7, 14}; // violin, flute, tuba
    for (size_t i = 0; i < notes.size(); ++i) {
        EXPECT_EQ(notes[i].token.time, 0) << "note " << i;
        EXPECT_EQ(notes[i].token.getRealDuration(), 50) << "note " << i;
        EXPECT_EQ(notes[i].token.getPitch(), 60) << "note " << i;
        // Woodwinds are scaled up and brass down per instrument (scaleVelocityForInstrument).
        EXPECT_EQ(notes[i].velocity,
                  InstrumentConstants::scaleVelocityForInstrument(expectedInstruments[i], 100))
            << "note " << i;
        EXPECT_EQ(notes[i].localInstrumentId, expectedInstruments[i]) << "note " << i;
    }
}

TEST(InstrumentCombinations, AllowedBrassTokens_OnlySelectedBits) {
    // local brass ids 11,13,14 → relative bits 0,2,3
    // allowed nonempty combos: 1,4,5,8,9,12,13 (+ brassOffset); 0 forbidden
    const std::vector<int32_t> instruments = {11, 13, 14};
    InstrumentCombinations model;

    auto allowed = model.allowedTokensForFamily(2, instruments); // brass
    std::sort(allowed.begin(), allowed.end());

    const std::vector<int32_t> expected = {
        InstrumentCombinations::brassOffset + 1,
        InstrumentCombinations::brassOffset + 4,
        InstrumentCombinations::brassOffset + 5,
        InstrumentCombinations::brassOffset + 8,
        InstrumentCombinations::brassOffset + 9,
        InstrumentCombinations::brassOffset + 12,
        InstrumentCombinations::brassOffset + 13,
    };
    EXPECT_EQ(InstrumentCombinations::brassOffset, 11543);
    EXPECT_EQ(allowed, expected);
}

TEST(InstrumentCombinations, AllowedGroups_NoBrassMasksBrassFamilyBit) {
    // violin(0) + flute(7) only → families strings|woodwinds = bits 0|1 = 3
    const std::vector<int32_t> instruments = {0, 7};
    InstrumentCombinations model;

    auto allowed = model.allowedTokensForGroups(instruments);
    std::sort(allowed.begin(), allowed.end());

    // nonempty subsets of {strings, woodwinds}: 1,2,3 (0 forbidden)
    const std::vector<int32_t> expected = {
        InstrumentCombinations::groupsTokenForCombo(1),
        InstrumentCombinations::groupsTokenForCombo(2),
        InstrumentCombinations::groupsTokenForCombo(3),
    };
    EXPECT_EQ(allowed, expected);

    for (const int32_t token: allowed) {
        const int32_t combo = InstrumentCombinations::comboFromGroupsToken(token);
        EXPECT_EQ(combo & (1 << 2), 0) << "brass family bit must be clear";
    }
}

TEST(InstrumentCombinations, EncodedCombinationRespectsGroupsSubset) {
    // groups = strings only (1); woodwind/brass family tokens present but must be ignored
    const std::vector<int32_t> sampledTokens = {
        0,
        10050,
        60,
        100,
        InstrumentCombinations::groupsTokenForCombo(1),
        InstrumentCombinations::stringsOffset + 1,
        InstrumentCombinations::woodwindOffset + 1,
        InstrumentCombinations::brassOffset + 8,
        InstrumentCombinations::endNote,
    };

    InstrumentCombinations model;
    const auto notes = model.encodedCombinationToInstruments(sampledTokens);

    ASSERT_EQ(notes.size(), 1u);
    EXPECT_EQ(notes[0].localInstrumentId, 0);
}

TEST(InstrumentCombinations, EmptyFamilyAllowsOnlyEndNote) {
    InstrumentCombinations model;
    const auto allowed = model.allowedTokensForFamily(2, {0, 7}); // no brass
    ASSERT_EQ(allowed.size(), 1u);
    EXPECT_EQ(allowed.front(), InstrumentCombinations::endNote);
}

TEST(InstrumentCombinations, PitchFilter_FluteOnlyRejectsG3) {
    // Flute GM range starts at C4 (60); G3 (55) is below.
    const auto filtered = InstrumentConstants::filterLocalInstrumentsForPitch({7}, 55);
    EXPECT_TRUE(filtered.empty());
}

TEST(InstrumentCombinations, PitchFilter_ViolinFluteG3KeepsViolin) {
    const auto filtered = InstrumentConstants::filterLocalInstrumentsForPitch({0, 7}, 55);
    ASSERT_EQ(filtered.size(), 1u);
    EXPECT_EQ(filtered.front(), 0);
}

TEST(InstrumentCombinations, AllEndNoteFamiliesEmitNoNotes) {
    // What the pitch mask produces when no active instrument can play the pitch: there is
    // no "empty" groups id, so every family slot is endNote instead.
    const std::vector<int32_t> sampledTokens = {
        0,
        10050,
        55,
        100,
        InstrumentCombinations::groupsTokenForCombo(1),
        InstrumentCombinations::endNote,
        InstrumentCombinations::endNote,
        InstrumentCombinations::endNote,
        InstrumentCombinations::endNote,
    };

    InstrumentCombinations model;
    const auto notes = model.encodedCombinationToInstruments(sampledTokens);
    EXPECT_TRUE(notes.empty());
}

TEST(InstrumentCombinations, GroupsTokensFillFifteenSlotsBelowStrings) {
    // Combos 1..15 pack into the 15 ids before stringsOffset — no id is left for the
    // empty set, and the all-families combo must not spill into the strings block.
    EXPECT_EQ(InstrumentCombinations::numGroupTokens, 15);
    EXPECT_EQ(InstrumentCombinations::groupsTokenForCombo(1),
              InstrumentCombinations::groupsOffset);
    EXPECT_EQ(InstrumentCombinations::groupsTokenForCombo(15),
              InstrumentCombinations::stringsOffset - 1);

    for (int32_t combo = 1; combo <= InstrumentCombinations::numGroupTokens; ++combo) {
        EXPECT_EQ(InstrumentCombinations::comboFromGroupsToken(
                      InstrumentCombinations::groupsTokenForCombo(combo)),
                  combo);
    }

    // Out-of-block ids decode to "no families" rather than a bogus mask.
    EXPECT_EQ(InstrumentCombinations::comboFromGroupsToken(InstrumentCombinations::groupsOffset - 1),
              0);
    EXPECT_EQ(InstrumentCombinations::comboFromGroupsToken(InstrumentCombinations::stringsOffset),
              0);
}

} // namespace virtual_orch_test
