#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include <VirtualOrch/MusicToken.h>
#include <VirtualOrch/ProjectPaths.h>
#include <VirtualOrch/orchestration-models/IodPretrained.h>
#include <VirtualOrch/orchestration-models/IodPretrainedTypes.h>

namespace virtual_orch_test {
namespace {

auto onnxPath() -> juce::File {
    return modelsDir().getChildFile("ot_legacy.onnx");
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
    const auto logits = model.runNoteLogits(window, 0); // the split export when it is next to the model
    ASSERT_FALSE(logits.empty());
    const size_t expected = static_cast<size_t>(IodPretrainedTypes::comboLen)
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

TEST(IodPretrainedOrt, TheSplitExportGivesTheFullModelsLogits) {
    // ot_robust_enc.onnx + ot_robust_dec.onnx (Nov 24 2026): the backbone once per window, the decoder per
    // step for one note. Loaded through a folder holding only a link to ot_robust.onnx, the same model runs
    // whole; both must give the same logits for every step of a note.
    const auto full = modelsDir().getChildFile("ot_robust.onnx");
    if (! full.existsAsFile() || ! modelsDir().getChildFile("ot_robust_dec.onnx").existsAsFile())
        GTEST_SKIP() << "no ot_robust split export";
    const auto dir = juce::File::createTempFile("ot_full_only");
    ASSERT_TRUE(dir.createDirectory());
    const auto link = dir.getChildFile("ot_robust.onnx");
    ASSERT_TRUE(full.createSymbolicLink(link, true));
    IodPretrained whole, split;
    whole.init(link.getFullPathName().toRawUTF8());
    split.init(full.getFullPathName().toRawUTF8());
    if (! whole.isModelLoaded() || ! split.isModelLoaded())
        GTEST_SKIP() << "ORT session failed";
    EXPECT_FALSE(whole.usesSplitModel());
    ASSERT_TRUE(split.usesSplitModel());
    std::vector<Token> tokens;
    for (int n = 0; n < 60; ++n)
        tokens.push_back(makeToken(25 * n, 20 + (n * 7) % 60, 40 + (n * 5) % 40));
    auto window = IodPretrainedTypes::packEncoderWindow(tokens);
    for (int n = 0; n < 50; ++n) // earlier notes' choices
        for (int st = 1; st < IodPretrainedTypes::comboLen; ++st)
            window.comboIn[static_cast<size_t>(n)][static_cast<size_t>(st)] = 1 + (n * 3 + st * 11) % 30;
    for (const int note: {0, 17, 49}) {
        const auto a = whole.runNoteLogits(window, note);
        const auto b = split.runNoteLogits(window, note);
        ASSERT_EQ(a.size(), b.size());
        ASSERT_FALSE(a.empty());
        float maxd = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
            maxd = std::max(maxd, std::abs(a[i] - b[i]));
        EXPECT_LT(maxd, 1e-3f) << "note " << note;
    }
    dir.deleteRecursively();
}

} // namespace virtual_orch_test
