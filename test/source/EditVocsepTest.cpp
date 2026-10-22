#include <gtest/gtest.h>

#include <set>

#include <VirtualOrch/OrchestrationTransformer.h>
#include <VirtualOrch/ProjectPaths.h>
#include <VirtualOrch/vocsep/VoiceSeparation.h>

namespace virtual_orch_test {

namespace {

auto keyboardNote(int32_t onset, int32_t pitch, int32_t durCs, int32_t keyId) -> Token {
    Token t{onset, static_cast<int32_t>(Vocab::DurOffset) + durCs,
            static_cast<int32_t>(Vocab::NoteOffset) + pitch, 80};
    t.keyId = keyId;
    return t;
}

} // namespace

TEST(EditVocsep, withoutAModelUserNotesKeepNoVoice) {
    OrchestrationTransformer ot;
    std::vector<Token> batch{keyboardNote(100, 60, 50, 1)};
    ot.stampInputVoices(batch);
    EXPECT_EQ(batch[0].voiceId, -1);
    EXPECT_EQ(batch[0].noteId, -1);
}

TEST(EditVocsep, userNotesGetVoicesFromTheirOwnVocsep) {
    // The test binary has no app dir: look for vocsep.onnx where the app's models live.
    juce::File onnx;
    std::vector<juce::File> dirs{documentsModelsDir(), onnxExportDir(), onnxExportFallbackDir()};
    for (const auto &d: onnxExportDirs())
        dirs.push_back(d);
    for (const auto &d: dirs)
        if (const auto f = d.getChildFile("vocsep.onnx"); onnx == juce::File() && f.existsAsFile())
            onnx = f;
    if (onnx == juce::File())
        GTEST_SKIP() << "vocsep.onnx not found";
    VoiceSeparation separation;
    separation.init(onnx.getFullPathName().toRawUTF8());
    ASSERT_TRUE(separation.isLoaded());
    OrchestrationTransformer ot;
    ot.inputVoiceSeparation = &separation;

    // Two lines: a melody stepping up high, a bass holding low.
    int32_t key = 0;
    for (int i = 0; i < 8; ++i) {
        std::vector<Token> batch{keyboardNote(100 + i * 40, 72 + i, 35, ++key),
                                 keyboardNote(100 + i * 40, 40 - (i % 2), 35, ++key)};
        ot.stampInputVoices(batch);
        for (const auto &t: batch) {
            EXPECT_GE(t.voiceId, 0) << "every user note gets a voice";
            EXPECT_GE(t.noteId, OrchestrationTransformer::kInputNoteIdBase) << "ids apart from the reduction's";
        }
    }
    const auto &history = ot.inputVocsepHistoryForTest();
    ASSERT_EQ(history.size(), 16u);
    std::set<int32_t> voices;
    for (const auto &t: history)
        voices.insert(t.voiceId);
    EXPECT_GE(voices.size(), 2u) << "the two lines are told apart";
}

} // namespace virtual_orch_test
