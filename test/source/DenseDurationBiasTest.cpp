#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <VirtualOrch/reduction/DenseDurationBias.h>
#include <VirtualOrch/reduction/DenseTypes.h>

namespace virtual_orch_test {
namespace {

auto denseNote(int32_t onset, int32_t durCs, int32_t instr = 0, int32_t pitch = 60)
    -> std::vector<int32_t> {
    return {onset,
            static_cast<int32_t>(DenseVocab::DurOffset) + durCs,
            static_cast<int32_t>(DenseVocab::NoteOffset) + instr * DenseConfig::MaxPitch + pitch,
            static_cast<int32_t>(DenseVocab::VelocityOffset) + 100};
}

auto appendNote(std::vector<int32_t> &data, int32_t onset, int32_t durCs, int32_t instr = 0)
    -> void {
    const auto n = denseNote(onset, durCs, instr);
    data.insert(data.end(), n.begin(), n.end());
}

} // namespace

TEST(DenseDurationBias, OnlyStepAlignedDurationsCounted) {
    std::vector<int32_t> data;
    appendNote(data, 100, 50); // legal
    appendNote(data, 200, 51); // snaps to 50
    appendNote(data, 300, 100);

    const auto stats = DenseDurationBias::collectDurationWindowAt(data, 300);
    EXPECT_EQ(stats.noteCount, 3);
    EXPECT_EQ(stats.lastDurationCs, 100);
    EXPECT_EQ(stats.uniqueDurations, (std::vector<int32_t>{50, 100}));
}

TEST(DenseDurationBias, BiasPenalizesFrequentDuration) {
    std::vector<int32_t> data;
    for (int i = 0; i < 8; ++i)
        appendNote(data, 100 + i * 20, 50);
    appendNote(data, 300, 100);

    const auto stats = DenseDurationBias::collectDurationWindowAt(data, 300);
    ASSERT_GT(stats.noteCount, 0);

    std::vector<float> logits(DenseVocab::VocabSize, 0.0f);
    DenseQuantize::maskIllegalDurationLogits(logits);

    DenseDurationBias::applyDurationLogitsBias(logits, stats, 1.0f, 0.0f);
    // Frequent 50cs gets a smaller boost than rare 100cs (π-smoothing).
    EXPECT_LT(logits[DenseVocab::DurOffset + 50], logits[DenseVocab::DurOffset + 100]);
    EXPECT_GT(logits[DenseVocab::DurOffset + 100], 0.0f);
}

TEST(DenseDurationBias, TauHoldThenNoteRamp) {
    DenseDurationBias::DurationTauScheduler sched;
    // countNote=false: time can end the hold, but ramp needs notes.
    EXPECT_FLOAT_EQ(sched.evaluate(0, {50}, {}, DenseDurationBias::TauBase,
                                   DenseDurationBias::TauPrimeBase, false)
                        .tau,
                    DenseDurationBias::TauBase);
    EXPECT_FLOAT_EQ(sched.evaluate(200, {50}, {}, DenseDurationBias::TauBase,
                                   DenseDurationBias::TauPrimeBase, false)
                        .tau,
                    DenseDurationBias::TauBase);
    // Past 2 s hold with 0 notes → still base (1.5^0).
    EXPECT_FLOAT_EQ(sched.evaluate(300, {50}, {}, DenseDurationBias::TauBase,
                                   DenseDurationBias::TauPrimeBase, false)
                        .tau,
                    DenseDurationBias::TauBase);

    DenseDurationBias::DurationTauScheduler schedNotes;
    // 10 counted notes ends hold early → base * 1.5^(10/10)
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(schedNotes
                            .evaluate(i * 5, {50}, {}, DenseDurationBias::TauBase,
                                      DenseDurationBias::TauPrimeBase, true)
                            .tau,
                        DenseDurationBias::TauBase);
    }
    EXPECT_FLOAT_EQ(schedNotes
                        .evaluate(45, {50}, {}, DenseDurationBias::TauBase,
                                  DenseDurationBias::TauPrimeBase, true)
                        .tau,
                    DenseDurationBias::TauBase * DenseDurationBias::TauRampPerSecond);
    // New duration resets hold.
    EXPECT_FLOAT_EQ(schedNotes
                        .evaluate(50, {50, 100}, {}, DenseDurationBias::TauBase,
                                  DenseDurationBias::TauPrimeBase, true)
                        .tau,
                    DenseDurationBias::TauBase);
    EXPECT_FLOAT_EQ(DenseDurationBias::TauPrimeBase, 0.0f);
}

TEST(DenseDurationBias, TauScheduleNowIsMonotonic) {
    DenseDurationBias::DurationTauScheduler sched;
    // Build 10 notes so ramp is active, then rewind now.
    for (int i = 0; i < 10; ++i)
        sched.evaluate(i * 5, {15}, {}, DenseDurationBias::TauBase,
                       DenseDurationBias::TauPrimeBase, true);
    const float ramped =
        DenseDurationBias::TauBase * DenseDurationBias::TauRampPerSecond;
    EXPECT_FLOAT_EQ(sched.evaluate(100, {15}, {}, DenseDurationBias::TauBase,
                                   DenseDurationBias::TauPrimeBase, false)
                        .tau,
                    ramped);
    EXPECT_FLOAT_EQ(sched.evaluate(0, {15}, {}, DenseDurationBias::TauBase,
                                   DenseDurationBias::TauPrimeBase, false)
                        .tau,
                    ramped);
}

TEST(DenseDurationBias, TauResetsWhenUniqueSetEmpty) {
    DenseDurationBias::DurationTauScheduler sched;
    for (int i = 0; i < 10; ++i)
        sched.evaluate(i * 5, {15}, {}, DenseDurationBias::TauBase,
                       DenseDurationBias::TauPrimeBase, true);
    EXPECT_GT(sched.evaluate(50, {15}, {}, DenseDurationBias::TauBase,
                             DenseDurationBias::TauPrimeBase, false)
                  .tau,
              DenseDurationBias::TauBase);
    EXPECT_FLOAT_EQ(sched.evaluate(60, {}, {}, DenseDurationBias::TauBase,
                                   DenseDurationBias::TauPrimeBase, false)
                        .tau,
                    DenseDurationBias::TauBase);
}

TEST(DenseDurationBias, OnlyInstrumentFilterIgnoresOtherInstr) {
    std::vector<int32_t> data;
    // Piano (instr 0) with many durations — must not enter the reduction window.
    appendNote(data, 100, 50, 0);
    appendNote(data, 150, 100, 0);
    appendNote(data, 200, 200, 0);
    // Reduction output instrument only uses 15cs.
    appendNote(data, 120, 15, DenseConfig::PianoReductionOutputInstrument);
    appendNote(data, 180, 15, DenseConfig::PianoReductionOutputInstrument);
    appendNote(data, 250, 15, DenseConfig::PianoReductionOutputInstrument);

    const auto all = DenseDurationBias::collectDurationWindowAt(data, 250);
    EXPECT_EQ(all.uniqueDurations, (std::vector<int32_t>{15, 50, 100, 200}));

    const auto outOnly = DenseDurationBias::collectDurationWindowAt(
        data, 250, -1, DenseConfig::PianoReductionOutputInstrument);
    EXPECT_EQ(outOnly.noteCount, 3);
    EXPECT_EQ(outOnly.uniqueDurations, (std::vector<int32_t>{15}));
    EXPECT_EQ(outOnly.lastDurationCs, 15);
}

} // namespace virtual_orch_test
