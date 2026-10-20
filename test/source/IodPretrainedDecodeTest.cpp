#include <gtest/gtest.h>

#include <VirtualOrch/MusicToken.h>
#include <VirtualOrch/orchestration-models/IodPretrainedTypes.h>

namespace virtual_orch_test {
namespace {

auto makeCombo(int32_t groupsTok, std::initializer_list<int32_t> familyToks)
    -> std::array<int32_t, IodPretrainedTypes::comboLen> {
    std::array<int32_t, IodPretrainedTypes::comboLen> combo{};
    combo.fill(IodPretrainedTypes::endNote);
    combo[0] = groupsTok;
    size_t i = 1;
    for (const int32_t t: familyToks) {
        if (i >= combo.size())
            break;
        combo[i++] = t;
    }
    return combo;
}

TEST(IodPretrainedDecode, EmptyGroupsYieldsNoNotes) {
    const auto combo = makeCombo(0, {});
    const auto notes =
        IodPretrainedTypes::decodeComboToNotes(100, Vocab::DurOffset + 40, 60, 80, combo);
    EXPECT_TRUE(notes.empty());
}

TEST(IodPretrainedDecode, AllEndAfterGroupsYieldsNoNotes) {
    const auto combo = makeCombo(0b0010, {});
    const auto notes =
        IodPretrainedTypes::decodeComboToNotes(0, Vocab::DurOffset + 10, 60, 100, combo);
    EXPECT_TRUE(notes.empty());
}

TEST(IodPretrainedDecode, PairsRoundTripPreservesFluteDeltaZero) {
    const auto &ww = IodPretrainedTypes::familySpecs[1];
    const int32_t fluteD0 = 1 << IodPretrainedTypes::iodBit(0, 0);
    const auto combo =
        makeCombo(0b0010, {IodPretrainedTypes::familyTokenForMask(ww, fluteD0)});
    const auto pairs = IodPretrainedTypes::pairsFromCombo(combo);
    ASSERT_EQ(pairs.size(), 1u);
    EXPECT_EQ(IodPretrainedTypes::unpackProgram(pairs[0]), 73);
    EXPECT_EQ(IodPretrainedTypes::unpackDelta(pairs[0]), 0);
    const auto rebuilt = IodPretrainedTypes::comboFromPairs(pairs);
    EXPECT_EQ(rebuilt[0], combo[0]);
    EXPECT_EQ(rebuilt[1], combo[1]);
}

TEST(IodPretrainedDecode, OctaveDeltaShiftsPitch) {
    const auto &ww = IodPretrainedTypes::familySpecs[1];
    const int32_t fluteD1 = 1 << IodPretrainedTypes::iodBit(0, 1);
    const auto combo =
        makeCombo(0b0010, {IodPretrainedTypes::familyTokenForMask(ww, fluteD1)});

    const auto notes =
        IodPretrainedTypes::decodeComboToNotes(0, Vocab::DurOffset + 10, 60, 100, combo);
    ASSERT_EQ(notes.size(), 1u);
    EXPECT_EQ(notes[0].localInstrumentId, 7);
    EXPECT_EQ(notes[0].token.getPitch(), 72);
}

TEST(IodPretrainedDecode, SkipsOutOfMidiPitch) {
    const auto &ww = IodPretrainedTypes::familySpecs[1];
    const int32_t fluteD1 = 1 << IodPretrainedTypes::iodBit(0, 1);
    const auto combo =
        makeCombo(0b0010, {IodPretrainedTypes::familyTokenForMask(ww, fluteD1)});

    const auto notes =
        IodPretrainedTypes::decodeComboToNotes(0, Vocab::DurOffset + 10, 120, 100, combo);
    EXPECT_TRUE(notes.empty());
}

TEST(IodPretrainedDecode, MultipleInstrumentsAndFamilies) {
    const auto &strings = IodPretrainedTypes::familySpecs[0];
    const auto &brass = IodPretrainedTypes::familySpecs[2];
    const int32_t pizz = 1 << IodPretrainedTypes::iodBit(1, 0);
    const int32_t hornNeg = 1 << IodPretrainedTypes::iodBit(1, -1);
    const auto combo = makeCombo(
        0b0101,
        {IodPretrainedTypes::familyTokenForMask(strings, pizz),
         IodPretrainedTypes::familyTokenForMask(brass, hornNeg)});

    const auto notes =
        IodPretrainedTypes::decodeComboToNotes(10, Vocab::DurOffset + 50, 72, 70, combo);
    ASSERT_EQ(notes.size(), 2u);

    EXPECT_EQ(notes[0].localInstrumentId, 5);
    EXPECT_EQ(notes[0].token.getPitch(), 72);
    EXPECT_EQ(notes[0].velocity, 70); // strings ×1
    EXPECT_EQ(notes[1].localInstrumentId, 12);
    EXPECT_EQ(notes[1].token.getPitch(), 60);
    EXPECT_EQ(notes[1].velocity, 25); // brass ×0.35
}

TEST(IodPretrainedDecode, DropsSecondInstrumentOnSharedOctave) {
    const auto &ww = IodPretrainedTypes::familySpecs[1];
    const int32_t fluteAndOboeD0 = (1 << IodPretrainedTypes::iodBit(0, 0))
                                   | (1 << IodPretrainedTypes::iodBit(1, 0));
    const auto combo =
        makeCombo(0b0010, {IodPretrainedTypes::familyTokenForMask(ww, fluteAndOboeD0)});

    const auto notes =
        IodPretrainedTypes::decodeComboToNotes(0, Vocab::DurOffset + 10, 60, 100, combo);
    ASSERT_EQ(notes.size(), 1u);
    EXPECT_EQ(notes[0].localInstrumentId, 7); // flute kept; oboe Δ=0 dropped
    EXPECT_EQ(notes[0].token.getPitch(), 60);
}

TEST(IodPretrainedDecode, TwoDeltasSameInstrument) {
    const auto &ww = IodPretrainedTypes::familySpecs[1];
    const int32_t fluteBoth =
        (1 << IodPretrainedTypes::iodBit(0, 0)) | (1 << IodPretrainedTypes::iodBit(0, 1));
    const auto combo =
        makeCombo(0b0010, {IodPretrainedTypes::familyTokenForMask(ww, fluteBoth)});

    const auto notes =
        IodPretrainedTypes::decodeComboToNotes(0, Vocab::DurOffset + 10, 60, 100, combo);
    ASSERT_EQ(notes.size(), 2u);
    EXPECT_EQ(notes[0].localInstrumentId, 7);
    EXPECT_EQ(notes[0].token.getPitch(), 60);
    EXPECT_EQ(notes[1].localInstrumentId, 7);
    EXPECT_EQ(notes[1].token.getPitch(), 72);
}

} // namespace
} // namespace virtual_orch_test
