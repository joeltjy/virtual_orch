#include <gtest/gtest.h>

#include <VirtualOrch/GeneralMidiPitchRanges.h>
#include <VirtualOrch/orchestration-models/IodPretrainedTypes.h>

namespace virtual_orch_test {
namespace {

TEST(IodPretrainedForbid, EmptyPadFullyBansAllFamilies) {
    const auto fb = IodPretrainedTypes::forbidBitsForPitch({}, 60);
    EXPECT_EQ(fb.fullyBannedFamilyBits, 0b1111);
    for (int32_t f = 0; f < IodPretrainedTypes::numFamilies; ++f) {
        const auto &spec = IodPretrainedTypes::familySpecs[static_cast<size_t>(f)];
        EXPECT_EQ(fb.familyForbid[static_cast<size_t>(f)], (1 << spec.numBits) - 1) << "f=" << f;
    }
    EXPECT_FALSE(IodPretrainedTypes::isLegalGroupsMask(1, fb.fullyBannedFamilyBits));
    EXPECT_FALSE(IodPretrainedTypes::isLegalGroupsMask(0b1111, fb.fullyBannedFamilyBits));
}

TEST(IodPretrainedForbid, ViolinOnlySameAsEmpty) {
    const auto empty = IodPretrainedTypes::forbidBitsForPitch({}, 60);
    const auto violin = IodPretrainedTypes::forbidBitsForPitch({0}, 60);
    EXPECT_EQ(violin.fullyBannedFamilyBits, empty.fullyBannedFamilyBits);
    EXPECT_EQ(violin.familyForbid, empty.familyForbid);
}

TEST(IodPretrainedForbid, FluteOnlyMidC) {
    const auto fb = IodPretrainedTypes::forbidBitsForPitch({7}, 60);

    EXPECT_EQ(fb.fullyBannedFamilyBits & 0b0001, 0b0001);
    EXPECT_EQ(fb.fullyBannedFamilyBits & 0b0100, 0b0100);
    EXPECT_EQ(fb.fullyBannedFamilyBits & 0b1000, 0b1000);
    EXPECT_EQ(fb.fullyBannedFamilyBits & 0b0010, 0);

    const int32_t wwForbid = fb.familyForbid[1];
    EXPECT_NE(wwForbid & (1 << IodPretrainedTypes::iodBit(0, -1)), 0);
    EXPECT_EQ(wwForbid & (1 << IodPretrainedTypes::iodBit(0, 0)), 0);
    EXPECT_EQ(wwForbid & (1 << IodPretrainedTypes::iodBit(0, 1)), 0);
    for (int32_t i = 1; i < 4; ++i) {
        for (const int32_t d: {-1, 0, 1}) {
            EXPECT_NE(wwForbid & (1 << IodPretrainedTypes::iodBit(i, d)), 0)
                << "instr=" << i << " delta=" << d;
        }
    }

    EXPECT_TRUE(IodPretrainedTypes::isLegalGroupsMask(0b0010, fb.fullyBannedFamilyBits));
    EXPECT_FALSE(IodPretrainedTypes::isLegalGroupsMask(0b0011, fb.fullyBannedFamilyBits));

    const int32_t fluteDelta0 = 1 << IodPretrainedTypes::iodBit(0, 0);
    EXPECT_TRUE(IodPretrainedTypes::isLegalFamilyMask(fluteDelta0, wwForbid));
    const int32_t fluteDeltaNeg = 1 << IodPretrainedTypes::iodBit(0, -1);
    EXPECT_FALSE(IodPretrainedTypes::isLegalFamilyMask(fluteDeltaNeg, wwForbid));
}

TEST(IodPretrainedForbid, TremoloAndHornEdgePitch) {
    const auto hornRange = GeneralMidiPitchRanges::pitchRangeForId(60);
    const int32_t pitchNearTop = hornRange.high - 6;
    const int32_t pitchAtTop = hornRange.high;

    const auto fbTop = IodPretrainedTypes::forbidBitsForPitch({4, 12}, pitchAtTop);
    EXPECT_EQ(fbTop.fullyBannedFamilyBits & 0b0001, 0);
    EXPECT_EQ(fbTop.fullyBannedFamilyBits & 0b0100, 0);

    const int32_t brassForbid = fbTop.familyForbid[2];
    EXPECT_NE(brassForbid & (1 << IodPretrainedTypes::iodBit(1, 1)), 0) << "Δ=+1 at top";
    EXPECT_EQ(brassForbid & (1 << IodPretrainedTypes::iodBit(1, 0)), 0) << "Δ=0 at top";

    const auto fbMid = IodPretrainedTypes::forbidBitsForPitch({4, 12}, pitchNearTop);
    if (pitchNearTop + 12 <= hornRange.high) {
        EXPECT_EQ(fbMid.familyForbid[2] & (1 << IodPretrainedTypes::iodBit(1, 1)), 0);
    }

    const auto tremRange = GeneralMidiPitchRanges::pitchRangeForId(44);
    if (pitchAtTop + 12 > tremRange.high) {
        EXPECT_NE(fbTop.familyForbid[0] & (1 << IodPretrainedTypes::iodBit(0, 1)), 0);
    }
}

TEST(IodPretrainedForbid, OnlyActiveIodLocalsUnlockPrograms) {
    const auto fb = IodPretrainedTypes::forbidBitsForPitch({5, 9}, 72);
    EXPECT_EQ(fb.fullyBannedFamilyBits & 0b0100, 0b0100);
    EXPECT_EQ(fb.fullyBannedFamilyBits & 0b1000, 0b1000);
    EXPECT_EQ(fb.fullyBannedFamilyBits & 0b0001, 0);
    EXPECT_EQ(fb.fullyBannedFamilyBits & 0b0010, 0);

    EXPECT_NE(fb.familyForbid[0] & (1 << IodPretrainedTypes::iodBit(0, 0)), 0);
    EXPECT_EQ(fb.familyForbid[0] & (1 << IodPretrainedTypes::iodBit(1, 0)), 0);
    EXPECT_NE(fb.familyForbid[1] & (1 << IodPretrainedTypes::iodBit(0, 0)), 0);
    EXPECT_EQ(fb.familyForbid[1] & (1 << IodPretrainedTypes::iodBit(2, 0)), 0);
}

TEST(IodPretrainedForbid, SharedOctaveAcrossInstrumentsIllegal) {
    const auto &ww = IodPretrainedTypes::familySpecs[1];
    const int32_t fluteD0 = 1 << IodPretrainedTypes::iodBit(0, 0);
    const int32_t fluteD0AndD1 =
        (1 << IodPretrainedTypes::iodBit(0, 0)) | (1 << IodPretrainedTypes::iodBit(0, 1));
    const int32_t fluteAndOboeD0 =
        (1 << IodPretrainedTypes::iodBit(0, 0)) | (1 << IodPretrainedTypes::iodBit(1, 0));

    EXPECT_FALSE(IodPretrainedTypes::familyMaskHasSharedOctave(ww, fluteD0));
    EXPECT_FALSE(IodPretrainedTypes::familyMaskHasSharedOctave(ww, fluteD0AndD1));
    EXPECT_TRUE(IodPretrainedTypes::familyMaskHasSharedOctave(ww, fluteAndOboeD0));

    EXPECT_TRUE(IodPretrainedTypes::isLegalFamilyMask(fluteD0AndD1, 0, &ww, 0));
    EXPECT_FALSE(IodPretrainedTypes::isLegalFamilyMask(fluteAndOboeD0, 0, &ww, 0));

    // Cross-family: Δ=0 already claimed → flute Δ=0 illegal, flute Δ=+1 still ok.
    const int32_t claimedDelta0 = 1 << (0 + 1);
    EXPECT_FALSE(IodPretrainedTypes::isLegalFamilyMask(fluteD0, 0, &ww, claimedDelta0));
    const int32_t fluteD1 = 1 << IodPretrainedTypes::iodBit(0, 1);
    EXPECT_TRUE(IodPretrainedTypes::isLegalFamilyMask(fluteD1, 0, &ww, claimedDelta0));
}

TEST(IodPretrainedForbid, FilterPairsKeepsSameInstrMultiDelta) {
    using namespace IodPretrainedTypes;
    const int32_t flute0 = packProgramDelta(73, 0);
    const int32_t flute1 = packProgramDelta(73, 1);
    const int32_t oboe0 = packProgramDelta(68, 0);
    const auto kept = filterPairsUniqueOctave({flute0, flute1, oboe0});
    ASSERT_EQ(kept.size(), 2u);
    EXPECT_EQ(kept[0], flute0);
    EXPECT_EQ(kept[1], flute1);
}

TEST(IodPretrainedForbid, OuterVoiceDeltaRolesByConcurrentPitch) {
    using namespace IodPretrainedTypes;
    // Same-onset chord: 48 (bottom), 60 (inner), 72 (top)
    const int32_t onset[] = {100, 100, 100};
    const int32_t durTok[] = {encoderDurOffset + 50, encoderDurOffset + 50, encoderDurOffset + 50};
    const int32_t pitch[] = {48, 60, 72};

    const auto bottom = outerVoiceDeltaAllows(onset, durTok, pitch, 3, 0);
    EXPECT_TRUE(bottom.allowMinus1);
    EXPECT_FALSE(bottom.allowPlus1);

    const auto inner = outerVoiceDeltaAllows(onset, durTok, pitch, 3, 1);
    EXPECT_FALSE(inner.allowMinus1);
    EXPECT_FALSE(inner.allowPlus1);

    const auto top = outerVoiceDeltaAllows(onset, durTok, pitch, 3, 2);
    EXPECT_FALSE(top.allowMinus1);
    EXPECT_TRUE(top.allowPlus1);

    // Overlapping different onsets: held low note under a later high note.
    const int32_t o2[] = {0, 40};
    const int32_t d2[] = {encoderDurOffset + 100, encoderDurOffset + 40};
    const int32_t p2[] = {48, 72};
    const auto heldBass = outerVoiceDeltaAllows(o2, d2, p2, 2, 0);
    EXPECT_TRUE(heldBass.allowMinus1);
    EXPECT_FALSE(heldBass.allowPlus1);
    const auto lateTop = outerVoiceDeltaAllows(o2, d2, p2, 2, 1);
    EXPECT_FALSE(lateTop.allowMinus1);
    EXPECT_TRUE(lateTop.allowPlus1);

    // Non-overlapping: each is alone → both ±1 ok.
    const int32_t o3[] = {0, 200};
    const int32_t d3[] = {encoderDurOffset + 50, encoderDurOffset + 50};
    const int32_t p3[] = {48, 72};
    const auto first = outerVoiceDeltaAllows(o3, d3, p3, 2, 0);
    EXPECT_TRUE(first.allowMinus1);
    EXPECT_TRUE(first.allowPlus1);
    const auto second = outerVoiceDeltaAllows(o3, d3, p3, 2, 1);
    EXPECT_TRUE(second.allowMinus1);
    EXPECT_TRUE(second.allowPlus1);

    // Lone note: both allowed.
    const int32_t soloOnset[] = {50};
    const int32_t soloDur[] = {encoderDurOffset + 10};
    const int32_t soloPitch[] = {60};
    const auto solo = outerVoiceDeltaAllows(soloOnset, soloDur, soloPitch, 1, 0);
    EXPECT_TRUE(solo.allowMinus1);
    EXPECT_TRUE(solo.allowPlus1);
}

TEST(IodPretrainedForbid, ApplyOuterVoiceForbidMasksPlusMinusBits) {
    using namespace IodPretrainedTypes;
    auto fb = forbidBitsForPitch({7}, 60); // flute only
    applyOuterVoiceDeltaForbid(fb, /*allowMinus1*/ false, /*allowPlus1*/ false);

    const int32_t ww = fb.familyForbid[1];
    EXPECT_NE(ww & (1 << iodBit(0, -1)), 0);
    EXPECT_EQ(ww & (1 << iodBit(0, 0)), 0); // Δ=0 still legal
    EXPECT_NE(ww & (1 << iodBit(0, +1)), 0);

    EXPECT_TRUE(isLegalFamilyMask(1 << iodBit(0, 0), ww, &familySpecs[1], 0));
    EXPECT_FALSE(isLegalFamilyMask(1 << iodBit(0, 1), ww, &familySpecs[1], 0));
    EXPECT_FALSE(isLegalFamilyMask(1 << iodBit(0, -1), ww, &familySpecs[1], 0));
}

} // namespace
} // namespace virtual_orch_test
