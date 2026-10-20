#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include <VirtualOrch/MusicToken.h>
#include <VirtualOrch/reduction/DenseTypes.h>

namespace virtual_orch_test {
namespace {

TEST(DenseQuantize, SnapDurationFiveCsGrid) {
    EXPECT_EQ(DenseQuantize::snapDurationCs(0), 10);
    EXPECT_EQ(DenseQuantize::snapDurationCs(1), 10);
    EXPECT_EQ(DenseQuantize::snapDurationCs(5), 10);
    EXPECT_EQ(DenseQuantize::snapDurationCs(7), 10);
    EXPECT_EQ(DenseQuantize::snapDurationCs(8), 10);
    EXPECT_EQ(DenseQuantize::snapDurationCs(10), 10);
    EXPECT_EQ(DenseQuantize::snapDurationCs(12), 10);
    EXPECT_EQ(DenseQuantize::snapDurationCs(13), 15);
    EXPECT_EQ(DenseQuantize::snapDurationCs(400), 400);
    EXPECT_EQ(DenseQuantize::snapDurationCs(997), 995);
    EXPECT_EQ(DenseQuantize::snapDurationCs(1000), 995);
}

TEST(DenseQuantize, SnapVelocityStepEight) {
    EXPECT_EQ(DenseQuantize::snapVelocity(0), 0);
    EXPECT_EQ(DenseQuantize::snapVelocity(3), 0);
    EXPECT_EQ(DenseQuantize::snapVelocity(4), 8);
    EXPECT_EQ(DenseQuantize::snapVelocity(100), 104);
    EXPECT_EQ(DenseQuantize::snapVelocity(120), 120);
    EXPECT_EQ(DenseQuantize::snapVelocity(127), 120);
}

TEST(DenseQuantize, SnapPianoPitchEightyEightKeys) {
    EXPECT_EQ(DenseQuantize::snapPianoPitch(0), 21);
    EXPECT_EQ(DenseQuantize::snapPianoPitch(21), 21);
    EXPECT_EQ(DenseQuantize::snapPianoPitch(60), 60);
    EXPECT_EQ(DenseQuantize::snapPianoPitch(108), 108);
    EXPECT_EQ(DenseQuantize::snapPianoPitch(127), 108);
}

TEST(DenseQuantize, LegalityHelpersMatchGrids) {
    EXPECT_FALSE(DenseQuantize::isLegalDurationCs(5));
    EXPECT_TRUE(DenseQuantize::isLegalDurationCs(10));
    EXPECT_TRUE(DenseQuantize::isLegalDurationCs(995));
    EXPECT_FALSE(DenseQuantize::isLegalDurationCs(0));
    EXPECT_FALSE(DenseQuantize::isLegalDurationCs(7));
    EXPECT_FALSE(DenseQuantize::isLegalDurationCs(1000));

    EXPECT_TRUE(DenseQuantize::isLegalVelocity(0));
    EXPECT_TRUE(DenseQuantize::isLegalVelocity(120));
    EXPECT_FALSE(DenseQuantize::isLegalVelocity(100));
    EXPECT_FALSE(DenseQuantize::isLegalVelocity(127));

    EXPECT_TRUE(DenseQuantize::isLegalPianoPitch(21));
    EXPECT_TRUE(DenseQuantize::isLegalPianoPitch(108));
    EXPECT_FALSE(DenseQuantize::isLegalPianoPitch(20));
    EXPECT_FALSE(DenseQuantize::isLegalPianoPitch(109));
}

TEST(DenseQuantize, SnapPackedEvents) {
    std::vector<int32_t> data = {
        10,
        static_cast<int32_t>(DenseVocab::DurOffset) + 7,
        static_cast<int32_t>(DenseVocab::NoteOffset) + 10,
        static_cast<int32_t>(DenseVocab::VelocityOffset) + 100,
    };
    DenseQuantize::snapPackedEvents(data);
    EXPECT_EQ(data[1], static_cast<int32_t>(DenseVocab::DurOffset) + 10);
    EXPECT_EQ(data[2], static_cast<int32_t>(DenseVocab::NoteOffset) + 21);
    EXPECT_EQ(data[3], static_cast<int32_t>(DenseVocab::VelocityOffset) + 104);
}

TEST(DenseQuantize, MaskIllegalDurationLogits) {
    std::vector<float> logits(DenseVocab::VocabSize, 0.0f);
    DenseQuantize::maskIllegalDurationLogits(logits);

    int legal = 0;
    for (int32_t cs = 0; cs < DenseConfig::MaxDur; ++cs) {
        const bool ok = std::isfinite(logits[DenseVocab::DurOffset + static_cast<size_t>(cs)]);
        if (DenseQuantize::isLegalDurationCs(cs)) {
            EXPECT_TRUE(ok) << "cs=" << cs;
            ++legal;
        } else {
            EXPECT_FALSE(ok) << "cs=" << cs;
        }
    }
    EXPECT_EQ(legal, (995 - 10) / 5 + 1);
    EXPECT_TRUE(std::isfinite(logits[DenseVocab::TimeOffset]));
    EXPECT_TRUE(std::isfinite(logits[DenseVocab::NoteOffset + 60]));
}

TEST(DenseQuantize, MaskIllegalPianoNoteLogits) {
    std::vector<float> logits(DenseVocab::VocabSize, 0.0f);
    DenseQuantize::maskIllegalPianoNoteLogits(logits);

    for (int32_t pitch = 0; pitch < DenseConfig::MaxPitch; ++pitch) {
        const bool ok =
            std::isfinite(logits[DenseVocab::NoteOffset + static_cast<size_t>(pitch)]);
        EXPECT_EQ(ok, DenseQuantize::isLegalPianoPitch(pitch)) << "pitch=" << pitch;
    }
    // Second instrument band also clamped.
    const size_t band1 = DenseVocab::NoteOffset + static_cast<size_t>(DenseConfig::MaxPitch);
    EXPECT_FALSE(std::isfinite(logits[band1 + 20]));
    EXPECT_TRUE(std::isfinite(logits[band1 + 60]));
}

TEST(DenseQuantize, MaskIllegalVelocityLogits) {
    std::vector<float> logits(DenseVocab::VocabSize, 0.0f);
    DenseQuantize::maskIllegalVelocityLogits(logits);

    int legal = 0;
    for (int32_t vel = 0; vel < DenseConfig::MaxVelocity; ++vel) {
        const bool ok =
            std::isfinite(logits[DenseVocab::VelocityOffset + static_cast<size_t>(vel)]);
        if (DenseQuantize::isLegalVelocity(vel)) {
            EXPECT_TRUE(ok) << "vel=" << vel;
            ++legal;
        } else {
            EXPECT_FALSE(ok) << "vel=" << vel;
        }
    }
    EXPECT_EQ(legal, 120 / 8 + 1);
}

/** V2 ingests snapped velocity; note-off oldNote keeps raw MIDI vel — must snap to match. */
TEST(DenseQuantize, SnappedVelocityMatchesNoteOffOldNote) {
    constexpr int32_t onset = 100;
    constexpr int32_t dur = static_cast<int32_t>(DenseVocab::DurOffset) + 400;
    constexpr int32_t note = static_cast<int32_t>(DenseVocab::NoteOffset) + 60;
    constexpr int32_t rawVel = 77;
    const int32_t snappedVel = DenseQuantize::snapVelocity(rawVel);
    ASSERT_NE(snappedVel, rawVel);

    const Token inHistory{onset, dur, note, snappedVel};
    const Token oldNoteRaw{onset, dur, note, rawVel};
    EXPECT_FALSE(tokenEquals(inHistory, oldNoteRaw));
    EXPECT_TRUE(tokenEquals(inHistory, Token{onset, dur, note, snappedVel}));
}

} // namespace
} // namespace virtual_orch_test
