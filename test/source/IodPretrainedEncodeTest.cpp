#include <gtest/gtest.h>

#include <VirtualOrch/MusicToken.h>
#include <VirtualOrch/orchestration-models/IodPretrainedTypes.h>

namespace virtual_orch_test {
namespace {

auto makeToken(int32_t onset, int32_t durCs, int32_t pitch, int32_t velocity = 80) -> Token {
    Token t;
    t.time = onset;
    t.duration = static_cast<int32_t>(Vocab::DurOffset) + durCs;
    t.note = static_cast<int32_t>(Vocab::NoteOffset) + pitch;
    t.velocity = velocity;
    return t;
}

TEST(IodPretrainedEncode, EmptyInput) {
    const auto w = IodPretrainedTypes::packEncoderWindow({});
    EXPECT_EQ(w.nNotes, 0);
    EXPECT_EQ(w.inputIds[0], 0);
    EXPECT_EQ(w.attentionMask[0], 0);
    EXPECT_EQ(w.noteValid[0], 0);
    EXPECT_EQ(w.comboIn[0][0], IodPretrainedTypes::bos);
}

TEST(IodPretrainedEncode, ThreeNoteFixturePadsRest) {
    std::vector<Token> tokens = {
        makeToken(10, 40, 60, 90),
        makeToken(20, 50, 64, 100),
        makeToken(30, 60, 67, 70),
    };
    const auto w = IodPretrainedTypes::packEncoderWindow(tokens);

    EXPECT_EQ(w.nNotes, 3);

    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(w.noteValid[static_cast<size_t>(i)], 1) << i;
        EXPECT_EQ(w.comboIn[static_cast<size_t>(i)][0], IodPretrainedTypes::bos);
        for (int s = 1; s < IodPretrainedTypes::comboLen; ++s)
            EXPECT_EQ(w.comboIn[static_cast<size_t>(i)][static_cast<size_t>(s)],
                      IodPretrainedTypes::bos);
    }
    for (int i = 3; i < IodPretrainedTypes::nNotesMax; ++i) {
        EXPECT_EQ(w.noteValid[static_cast<size_t>(i)], 0) << i;
        const size_t base = static_cast<size_t>(i) * 3;
        EXPECT_EQ(w.inputIds[base], 0);
        EXPECT_EQ(w.attentionMask[base], 0);
    }

    EXPECT_EQ(w.inputIds[0], 10);
    EXPECT_EQ(w.inputIds[1], IodPretrainedTypes::encoderDurOffset + 40);
    EXPECT_EQ(w.inputIds[2], IodPretrainedTypes::encoderNoteOffset + 60);
    EXPECT_EQ(w.attentionMask[0], 1);
    EXPECT_EQ(w.attentionMask[1], 1);
    EXPECT_EQ(w.attentionMask[2], 1);

    EXPECT_EQ(w.onsetCs[0], 10);
    EXPECT_EQ(w.basePitch[0], 60);
    EXPECT_EQ(w.velocity[0], 90);
    EXPECT_EQ(w.basePitch[2], 67);
    EXPECT_EQ(w.velocity[2], 70);

    for (size_t i = 0; i < 9; ++i)
        EXPECT_EQ(w.attentionMask[i], 1) << i;
    for (size_t i = 9; i < IodPretrainedTypes::noteSeqLen; ++i)
        EXPECT_EQ(w.attentionMask[i], 0) << i;
}

TEST(IodPretrainedEncode, DropsInstrumentBandFromPitch) {
    Token t;
    t.time = 5;
    t.duration = Vocab::DurOffset + 10;
    t.note = static_cast<int32_t>(Vocab::NoteOffset) + 128 * 3 + 60;
    t.velocity = 80;

    const auto w = IodPretrainedTypes::packEncoderWindow({t});
    ASSERT_EQ(w.nNotes, 1);
    EXPECT_EQ(w.inputIds[2], IodPretrainedTypes::encoderNoteOffset + 60);
    EXPECT_EQ(w.basePitch[0], 60);
}

TEST(IodPretrainedEncode, WindowStartsChunking) {
    EXPECT_TRUE(IodPretrainedTypes::encoderWindowStarts(0).empty());
    EXPECT_EQ(IodPretrainedTypes::encoderWindowStarts(1), (std::vector<size_t>{0}));
    EXPECT_EQ(IodPretrainedTypes::encoderWindowStarts(128), (std::vector<size_t>{0}));
    EXPECT_EQ(IodPretrainedTypes::encoderWindowStarts(129), (std::vector<size_t>{0, 128}));
    EXPECT_EQ(IodPretrainedTypes::encoderWindowStarts(256), (std::vector<size_t>{0, 128}));
    EXPECT_EQ(IodPretrainedTypes::encoderWindowStarts(257), (std::vector<size_t>{0, 128, 256}));
}

TEST(IodPretrainedEncode, SecondWindowSlice) {
    std::vector<Token> tokens;
    tokens.reserve(130);
    for (int32_t i = 0; i < 130; ++i)
        tokens.push_back(makeToken(i, 10, 60 + (i % 12)));

    const auto w0 = IodPretrainedTypes::packEncoderWindow(tokens, 0);
    EXPECT_EQ(w0.nNotes, 128);
    EXPECT_EQ(w0.onsetCs[0], 0);
    EXPECT_EQ(w0.onsetCs[127], 127);

    const auto w1 = IodPretrainedTypes::packEncoderWindow(tokens, 128);
    EXPECT_EQ(w1.nNotes, 2);
    EXPECT_EQ(w1.onsetCs[0], 128);
    EXPECT_EQ(w1.onsetCs[1], 129);
    EXPECT_EQ(w1.noteValid[2], 0);
}

} // namespace
} // namespace virtual_orch_test
