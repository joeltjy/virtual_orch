#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include <VirtualOrch/MusicToken.h>
#include <VirtualOrch/reduction/DenseTypes.h>

namespace virtual_orch_test {
namespace {

TEST(PianoReductionInstrumentMask, LeavesOnlyInstrument1Finite) {
    std::vector<float> logits(DenseVocab::VocabSize, 1.0f);
    ASSERT_TRUE(DenseQuantize::maskPianoReductionGenerationInstrument(logits));

    // Time / dur / vel still finite here — safeLogits clears those in RT before this mask.
    // Note bands: only instrument 1 may remain finite.
    for (int32_t pitch = 0; pitch < DenseConfig::MaxPitch; ++pitch) {
        const size_t i0 =
            DenseVocab::NoteOffset + static_cast<size_t>(pitch); // instr 0
        const size_t i1 =
            DenseVocab::NoteOffset
            + static_cast<size_t>(DenseConfig::MaxPitch) * 1
            + static_cast<size_t>(pitch);
        EXPECT_FALSE(std::isfinite(logits[i0]));
        EXPECT_TRUE(std::isfinite(logits[i1]));
    }
    for (int instr = 2; instr < DenseConfig::MaxInstr; ++instr) {
        for (int32_t pitch = 0; pitch < DenseConfig::MaxPitch; ++pitch) {
            const size_t idx =
                DenseVocab::NoteOffset
                + static_cast<size_t>(instr) * DenseConfig::MaxPitch
                + static_cast<size_t>(pitch);
            EXPECT_FALSE(std::isfinite(logits[idx]));
        }
    }
}

TEST(PianoReductionInstrumentMask, SampleTopPNeverPicksInstrument0AfterFullMasks) {
    std::vector<float> logits(DenseVocab::VocabSize, 0.0f);
    // Match RT note-step masking: kill time/dur/vel, illegal pitches, then instr bands.
    std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::TimeOffset),
              logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::DurOffset),
              -std::numeric_limits<float>::infinity());
    std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::DurOffset),
              logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::NoteOffset),
              -std::numeric_limits<float>::infinity());
    std::fill(logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::VelocityOffset),
              logits.begin() + static_cast<std::ptrdiff_t>(DenseVocab::VocabSize),
              -std::numeric_limits<float>::infinity());
    DenseQuantize::maskIllegalPianoNoteLogits(logits);
    ASSERT_TRUE(DenseQuantize::maskPianoReductionGenerationInstrument(logits));

    // Boost one legal instr-1 pitch so sampling is deterministic under T→0.
    const int32_t pitch = 60;
    const size_t i1 =
        DenseVocab::NoteOffset
        + static_cast<size_t>(DenseConfig::MaxPitch)
              * DenseConfig::PianoReductionOutputInstrument
        + static_cast<size_t>(pitch);
    logits[i1] = 10.0f;

    for (int trial = 0; trial < 20; ++trial) {
        auto scores = logits; // sampleTopP mutates
        const int32_t token = sampleTopP(scores, 0.98f, 0.0f); // argmax on nucleus
        EXPECT_EQ(DenseQuantize::instrumentOfNoteToken(token),
                  DenseConfig::PianoReductionOutputInstrument);
        EXPECT_EQ(token, static_cast<int32_t>(i1));
    }
}

TEST(PianoReductionInstrumentMask, RejectsUndersizedLogits) {
    std::vector<float> logits(DenseVocab::NoteOffset + 50, 0.0f);
    EXPECT_FALSE(DenseQuantize::maskPianoReductionGenerationInstrument(logits));
}

} // namespace
} // namespace virtual_orch_test
