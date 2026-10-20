#include <gtest/gtest.h>

#include <cmath>

#include <VirtualOrch/MusicToken.h>
#include <VirtualOrch/ProjectPaths.h>
#include <VirtualOrch/orchestration-models/IodPretrained.h>
#include <VirtualOrch/orchestration-models/IodPretrainedTypes.h>

namespace virtual_orch_test {
namespace {

auto onnxPath() -> juce::File {
    return onnxExportDir().getChildFile("instrument_octave_predictions.onnx");
}

auto makeToken(int32_t onset, int32_t durCs, int32_t pitch) -> Token {
    Token t;
    t.time = onset;
    t.duration = static_cast<int32_t>(Vocab::DurOffset) + durCs;
    t.note = static_cast<int32_t>(Vocab::NoteOffset) + pitch;
    t.velocity = 80;
    return t;
}

TEST(IodPretrainedOrt, LoadAndForwardPadWindow) {
    const auto path = onnxPath();
    if (! path.existsAsFile()) {
        GTEST_SKIP() << "missing " << path.getFullPathName();
    }

    IodPretrained model;
    model.init(path.getFullPathName().toRawUTF8());
    if (! model.isModelLoaded()) {
        GTEST_SKIP() << "ORT session failed (need CUDA for Trilu in this export)";
    }

    auto window = IodPretrainedTypes::packEncoderWindow({makeToken(0, 40, 60)});
    const auto logits = model.runLogits(window);
    ASSERT_FALSE(logits.empty());
    const size_t expected = static_cast<size_t>(IodPretrainedTypes::nNotesMax)
                            * static_cast<size_t>(IodPretrainedTypes::comboLen)
                            * static_cast<size_t>(IodPretrainedTypes::comboVocabSize);
    EXPECT_EQ(logits.size(), expected);
    bool anyFinite = false;
    for (size_t i = 0; i < static_cast<size_t>(IodPretrainedTypes::comboVocabSize); ++i) {
        if (std::isfinite(logits[i])) {
            anyFinite = true;
            break;
        }
    }
    EXPECT_TRUE(anyFinite);
}

TEST(IodPretrainedOrt, EmptyInstrumentsYieldsEndCombos) {
    const auto path = onnxPath();
    if (! path.existsAsFile()) {
        GTEST_SKIP() << "missing " << path.getFullPathName();
    }

    IodPretrained model;
    model.init(path.getFullPathName().toRawUTF8());
    if (! model.isModelLoaded()) {
        GTEST_SKIP() << "ORT session failed (need CUDA for Trilu in this export)";
    }

    std::vector<Token> tokens = {makeToken(0, 40, 60), makeToken(50, 40, 64)};
    auto window = IodPretrainedTypes::packEncoderWindow(tokens);
    ASSERT_EQ(window.nNotes, 2);

    const auto combos = model.sampleWindowCombos(window, /*instruments*/ {}, /*applyGroupsBias*/ false);
    ASSERT_EQ(combos.size(), 2u);
    for (const auto &combo: combos) {
        for (int32_t s = 0; s < IodPretrainedTypes::comboLen; ++s)
            EXPECT_EQ(combo[static_cast<size_t>(s)], IodPretrainedTypes::endNote) << s;

        const auto notes = IodPretrainedTypes::decodeComboToNotes(
            0, Vocab::DurOffset + 40, 60, 80, combo);
        EXPECT_TRUE(notes.empty());
    }
}

TEST(IodPretrainedOrt, FlutePadSamplesLegalCombo) {
    const auto path = onnxPath();
    if (! path.existsAsFile()) {
        GTEST_SKIP() << "missing " << path.getFullPathName();
    }

    IodPretrained model;
    model.init(path.getFullPathName().toRawUTF8());
    if (! model.isModelLoaded()) {
        GTEST_SKIP() << "ORT session failed (need CUDA for Trilu in this export)";
    }

    auto window = IodPretrainedTypes::packEncoderWindow({makeToken(0, 40, 60)});
    const auto combos = model.sampleWindowCombos(window, {7}, /*applyGroupsBias*/ false);
    ASSERT_EQ(combos.size(), 1u);

    const auto forbid = IodPretrainedTypes::forbidBitsForPitch({7}, 60);
    const auto &combo = combos[0];
    EXPECT_NE(combo[0], IodPretrainedTypes::endNote);

    const int32_t groupsMask = IodPretrainedTypes::maskFromGroupsToken(combo[0]);
    EXPECT_TRUE(IodPretrainedTypes::isLegalGroupsMask(groupsMask, forbid.fullyBannedFamilyBits));

    const auto notes = IodPretrainedTypes::decodeComboToNotes(
        window.onsetCs[0], window.durationToken[0], window.basePitch[0], window.velocity[0],
        combo);
    for (const auto &n: notes) {
        EXPECT_EQ(n.localInstrumentId, 7);
        const int32_t dp = n.token.getPitch() - 60;
        EXPECT_TRUE(dp == -12 || dp == 0 || dp == 12) << dp;
    }
}

TEST(IodPretrainedOrt, GetOutputEndToEnd) {
    const auto path = onnxPath();
    if (! path.existsAsFile()) {
        GTEST_SKIP() << "missing " << path.getFullPathName();
    }

    IodPretrained model;
    model.init(path.getFullPathName().toRawUTF8());
    if (! model.isModelLoaded()) {
        GTEST_SKIP() << "ORT session failed (need CUDA for Trilu in this export)";
    }

    IodPretrained::ConditioningSignal cond{};
    {
        const auto out = model.getOutput({makeToken(100, 40, 60)}, {}, cond);
        EXPECT_TRUE(out.empty());
        EXPECT_TRUE(model.tokenHistory().empty());
    }

    const auto out = model.getOutput({makeToken(200, 50, 60), makeToken(300, 50, 64)}, {7}, cond);
    for (const auto &n: out) {
        EXPECT_EQ(n.localInstrumentId, 7);
        EXPECT_TRUE(n.token.time == 200 || n.token.time == 300) << n.token.time;
        const int32_t base = n.token.time == 200 ? 60 : 64;
        const int32_t dp = n.token.getPitch() - base;
        EXPECT_TRUE(dp == -12 || dp == 0 || dp == 12) << dp;
    }
    EXPECT_EQ(model.tokenHistory().size(), out.size());
}

} // namespace
} // namespace virtual_orch_test
