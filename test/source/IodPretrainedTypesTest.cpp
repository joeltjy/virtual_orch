#include <gtest/gtest.h>

#include <set>

#include <VirtualOrch/orchestration-models/IodPretrainedTypes.h>

namespace virtual_orch_test {
namespace {

TEST(IodPretrainedTypes, ComboVocabLayout) {
    EXPECT_EQ(IodPretrainedTypes::bos, 0);
    EXPECT_EQ(IodPretrainedTypes::groupsOffset, 1);
    EXPECT_EQ(IodPretrainedTypes::groupsSize, 15);
    EXPECT_EQ(IodPretrainedTypes::stringsOffset, 16);
    EXPECT_EQ(IodPretrainedTypes::stringsSize, 512);
    EXPECT_EQ(IodPretrainedTypes::woodwindOffset, 528);
    EXPECT_EQ(IodPretrainedTypes::woodwindSize, 4096);
    EXPECT_EQ(IodPretrainedTypes::brassOffset, 4624);
    EXPECT_EQ(IodPretrainedTypes::brassSize, 4096);
    EXPECT_EQ(IodPretrainedTypes::otherOffset, 8720);
    EXPECT_EQ(IodPretrainedTypes::otherSize, 64);
    EXPECT_EQ(IodPretrainedTypes::endNote, 8784);
    EXPECT_EQ(IodPretrainedTypes::comboVocabSize, 8785);
    EXPECT_EQ(IodPretrainedTypes::noteSeqLen, 384);
    EXPECT_EQ(IodPretrainedTypes::nNotesMax, 128);
}

TEST(IodPretrainedTypes, IodBitRoundTrip) {
    for (int32_t instr = 0; instr < 4; ++instr) {
        for (const int32_t delta: {-1, 0, 1}) {
            const int32_t bit = IodPretrainedTypes::iodBit(instr, delta);
            EXPECT_EQ(bit / IodPretrainedTypes::deltasPerInstrument, instr);
            EXPECT_EQ(bit % IodPretrainedTypes::deltasPerInstrument, delta + 1);
        }
    }

    const auto &strings = IodPretrainedTypes::familySpecs[0];
    const auto pd = IodPretrainedTypes::programDeltaFromBit(strings, IodPretrainedTypes::iodBit(1, -1));
    ASSERT_TRUE(pd.has_value());
    EXPECT_EQ(pd->first, 45);
    EXPECT_EQ(pd->second, -1);

    EXPECT_FALSE(IodPretrainedTypes::programDeltaFromBit(strings, 9).has_value());
}

TEST(IodPretrainedTypes, ThirteenIodProgramsMapToLocals) {
    const std::array<std::pair<int32_t, int32_t>, 13> expected{{
        {44, 4},  {45, 5},  {48, 6},
        {73, 7},  {68, 8},  {71, 9},  {70, 10},
        {56, 11}, {60, 12}, {57, 13}, {58, 14},
        {47, 16}, {52, 19},
    }};

    std::set<int32_t> seenLocals;
    for (const auto &[gm, local]: expected) {
        EXPECT_TRUE(IodPretrainedTypes::isIodGmProgram(gm)) << "gm=" << gm;
        const auto got = IodPretrainedTypes::localIdForIodGm(gm);
        ASSERT_TRUE(got.has_value()) << "gm=" << gm;
        EXPECT_EQ(*got, local) << "gm=" << gm;
        seenLocals.insert(*got);

        const auto fi = IodPretrainedTypes::familyInstrIndexForGm(gm);
        ASSERT_TRUE(fi.has_value()) << "gm=" << gm;
    }
    EXPECT_EQ(seenLocals.size(), 13u);
}

TEST(IodPretrainedTypes, DisabledTax20LocalsHaveNoIodSlot) {
    for (const int32_t local: IodPretrainedTypes::disabledTax20Locals) {
        EXPECT_TRUE(IodPretrainedTypes::isDisabledTax20Local(local));
        const auto gm = InstrumentConstants::gmIdForLocalInstrumentId(local);
        ASSERT_TRUE(gm.has_value());
        EXPECT_FALSE(IodPretrainedTypes::isIodGmProgram(*gm))
            << "local=" << local << " gm=" << *gm;
        EXPECT_FALSE(IodPretrainedTypes::familyInstrIndexForGm(*gm).has_value());
    }
}

TEST(IodPretrainedTypes, GroupsAndFamilyTokenCodec) {
    EXPECT_EQ(IodPretrainedTypes::groupsTokenForMask(5), 5);
    EXPECT_EQ(IodPretrainedTypes::maskFromGroupsToken(5), 5);
    EXPECT_EQ(IodPretrainedTypes::maskFromGroupsToken(0), 0);
    EXPECT_EQ(IodPretrainedTypes::maskFromGroupsToken(16), 0);

    const auto &ww = IodPretrainedTypes::familySpecs[1];
    const int32_t fluteDelta0 = 1 << IodPretrainedTypes::iodBit(0, 0);
    const int32_t token = IodPretrainedTypes::familyTokenForMask(ww, fluteDelta0);
    EXPECT_EQ(token, IodPretrainedTypes::woodwindOffset + fluteDelta0);
    EXPECT_EQ(IodPretrainedTypes::maskFromFamilyToken(ww, token), fluteDelta0);
    EXPECT_EQ(IodPretrainedTypes::maskFromFamilyToken(ww, IodPretrainedTypes::endNote), 0);
}

} // namespace
} // namespace virtual_orch_test
