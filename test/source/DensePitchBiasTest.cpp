#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <VirtualOrch/reduction/DensePitchBias.h>
#include <VirtualOrch/reduction/DenseTypes.h>

namespace virtual_orch_test {
namespace {

auto denseNote(int32_t onset, int32_t pitch) -> std::vector<int32_t> {
    return {onset,
            static_cast<int32_t>(DenseVocab::DurOffset) + 50,
            static_cast<int32_t>(DenseVocab::NoteOffset) + pitch,
            static_cast<int32_t>(DenseVocab::VelocityOffset) + 100};
}

auto appendNote(std::vector<int32_t> &data, int32_t onset, int32_t pitch) -> void {
    const auto n = denseNote(onset, pitch);
    data.insert(data.end(), n.begin(), n.end());
}

} // namespace

TEST(DensePitchBias, WindowUsesMaxOfClockAndLastOnset) {
    std::vector<int32_t> data;
    appendNote(data, 600, 60);
    appendNote(data, 700, 62);
    appendNote(data, 800, 64); // last onset 800 — generation ahead of clock

    // Clock 400 < last onset 800 → now = 800. Window (300, 800] includes all three
    // (pure clock-now would have dropped them as onset > 400).
    const auto ahead = DensePitchBias::collectPitchWindow(data, 400);
    EXPECT_EQ(ahead.noteCount, 3);
    EXPECT_EQ(ahead.lastPitch, 64);
    EXPECT_EQ(ahead.uniquePitches, (std::vector<int32_t>{60, 62, 64}));

    // Clock past last onset → now = clock; only notes in (clock-500, clock].
    const auto caughtUp = DensePitchBias::collectPitchWindow(data, 1100);
    EXPECT_EQ(caughtUp.noteCount, 2); // (600, 1100] → 700, 800
    EXPECT_EQ(caughtUp.lastPitch, 64);
    EXPECT_EQ(caughtUp.uniquePitches, (std::vector<int32_t>{62, 64}));
}

TEST(DensePitchBias, WindowFallsBackToLastOnsetWithoutClock) {
    std::vector<int32_t> data;
    appendNote(data, 600, 60);
    appendNote(data, 700, 62);
    appendNote(data, 800, 64);

    const auto stats = DensePitchBias::collectPitchWindow(data, -1);
    EXPECT_EQ(stats.noteCount, 3); // now = 800 → (300, 800]
    EXPECT_EQ(stats.lastPitch, 64);
}

TEST(DensePitchBias, UnigramSmoothedAndDeltas) {
    std::vector<int32_t> data;
    appendNote(data, 10, 60);
    appendNote(data, 20, 60);
    appendNote(data, 25, 60);
    appendNote(data, 30, 62);

    const auto stats = DensePitchBias::collectPitchWindow(data, 30);
    EXPECT_EQ(stats.noteCount, 4);
    EXPECT_GT(stats.pi[60], stats.pi[62]);
    EXPECT_GT(stats.pi[62], stats.pi[64]); // unseen but epsilon mass

    // Deltas: 60→60 (0), 60→60 (0), 60→62 (+2) → mass at 0 > mass at +2
    EXPECT_GT(stats.sDelta[static_cast<size_t>(DensePitchBias::deltaIndex(0))],
              stats.sDelta[static_cast<size_t>(DensePitchBias::deltaIndex(2))]);
    EXPECT_EQ(stats.uniqueDeltas, (std::vector<int32_t>{0, 2}));
}

TEST(DensePitchBias, WideLeapsExcludedFromDeltaSet) {
    // Chromatic high line interleaved with distant low notes — offline drops
    // |Δ| > 12 so +28 / -41 never enter D (τ′ schedule).
    std::vector<int32_t> data;
    appendNote(data, 10, 79);
    appendNote(data, 20, 78); // -1
    appendNote(data, 30, 49); // -29 — ignored
    appendNote(data, 40, 77); // +28 — ignored
    appendNote(data, 50, 76); // -1

    const auto stats = DensePitchBias::collectPitchWindow(data, 50);
    EXPECT_EQ(stats.uniqueDeltas, (std::vector<int32_t>{-1}));
    EXPECT_EQ(stats.noteCount, 5);
}

TEST(DensePitchBias, TauHoldThenNoteRamp) {
    DensePitchBias::PitchTauScheduler sched;
    const std::vector<int32_t> s{60, 62};
    const std::vector<int32_t> deltas{2};

    // Without counting notes, past 2 s hold still yields base (2^0).
    EXPECT_FLOAT_EQ(sched.evaluate(0, s, deltas).tau, DensePitchBias::TauBase);
    EXPECT_FLOAT_EQ(sched.evaluate(200, s, deltas).tau, DensePitchBias::TauBase);
    EXPECT_FLOAT_EQ(sched.evaluate(300, s, deltas).tau, DensePitchBias::TauBase);

    DensePitchBias::PitchTauScheduler schedNotes;
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(
            schedNotes.evaluate(i * 5, s, deltas, 0.5f, 0.5f, true).tau, 0.5f);
    }
    // 10th note ends hold → 0.5 * 2^(10/10) = 1
    auto r10 = schedNotes.evaluate(45, s, deltas, 0.5f, 0.5f, true);
    EXPECT_FLOAT_EQ(r10.tau, 1.0f);
    EXPECT_FLOAT_EQ(r10.tauPrime, 1.0f);
    // 20th note → 0.5 * 2^2 = 2
    for (int i = 0; i < 10; ++i)
        r10 = schedNotes.evaluate(50 + i * 5, s, deltas, 0.5f, 0.5f, true);
    EXPECT_FLOAT_EQ(r10.tau, 2.0f);
}

TEST(DensePitchBias, TauScheduleNowIsMonotonic) {
    DensePitchBias::PitchTauScheduler sched;
    const std::vector<int32_t> s{60};
    for (int i = 0; i < 10; ++i)
        sched.evaluate(i * 5, s, {}, 0.5f, 0.5f, true);
    EXPECT_FLOAT_EQ(sched.evaluate(100, s, {}, 0.5f, 0.5f, false).tau, 1.0f);
    EXPECT_FLOAT_EQ(sched.evaluate(0, s, {}, 0.5f, 0.5f, false).tau, 1.0f);
}

TEST(DensePitchBias, TauResetsWhenUniqueSetEmpty) {
    DensePitchBias::PitchTauScheduler sched;
    for (int i = 0; i < 10; ++i)
        sched.evaluate(i * 5, {60}, {}, 0.5f, 0.5f, true);
    EXPECT_FLOAT_EQ(sched.evaluate(50, {60}, {}, 0.5f, 0.5f, false).tau, 1.0f);
    EXPECT_FLOAT_EQ(sched.evaluate(60, {}, {}, 0.5f, 0.5f, false).tau, 0.5f);
}

TEST(DensePitchBias, TauResetsWhenSGainsPitch) {
    DensePitchBias::PitchTauScheduler sched;
    for (int i = 0; i < 10; ++i)
        sched.evaluate(i * 5, {60}, {}, 0.5f, 0.5f, true);
    EXPECT_FLOAT_EQ(sched.evaluate(50, {60}, {}, 0.5f, 0.5f, false).tau, 1.0f);

    auto r = sched.evaluate(50, {60, 62}, {}, 0.5f, 0.5f, true); // new pitch → reset τ
    EXPECT_FLOAT_EQ(r.tau, 0.5f);
    r = sched.evaluate(60, {60, 62}, {}, 0.5f, 0.5f, true);
    EXPECT_FLOAT_EQ(r.tau, 0.5f);
}

TEST(DensePitchBias, TauPrimeResetsWhenNewIntervalEnters) {
    DensePitchBias::PitchTauScheduler sched;
    for (int i = 0; i < 10; ++i)
        sched.evaluate(i * 5, {60, 62}, {2}, 0.5f, 0.5f, true);
    auto r = sched.evaluate(50, {60, 62}, {2}, 0.5f, 0.5f, false);
    EXPECT_FLOAT_EQ(r.tau, 1.0f);
    EXPECT_FLOAT_EQ(r.tauPrime, 1.0f);

    // New interval resets τ′; τ keeps counting this note (11 → 0.5·2^1.1).
    r = sched.evaluate(50, {60, 62}, {-2, 2}, 0.5f, 0.5f, true);
    EXPECT_FLOAT_EQ(r.tau, 0.5f * std::pow(2.0f, 11.0f / 10.0f));
    EXPECT_FLOAT_EQ(r.tauPrime, 0.5f);
}

TEST(DensePitchBias, TauKeepsRampingWhenSOnlyShrinks) {
    DensePitchBias::PitchTauScheduler sched;
    for (int i = 0; i < 10; ++i)
        sched.evaluate(i * 5, {60, 62, 64}, {2}, 0.5f, 0.5f, true);
    EXPECT_FLOAT_EQ(sched.evaluate(50, {60, 62, 64}, {2}, 0.5f, 0.5f, false).tau, 1.0f);

    // Pitch ages out — must not reset; this note bumps count 10→11.
    auto r = sched.evaluate(60, {62, 64}, {2}, 0.5f, 0.5f, true);
    EXPECT_FLOAT_EQ(r.tau, 0.5f * std::pow(2.0f, 11.0f / 10.0f));
    // Interval set emptied → τ′ snaps to base.
    r = sched.evaluate(70, {64}, {}, 0.5f, 0.5f, true);
    EXPECT_FLOAT_EQ(r.tauPrime, 0.5f);
}

TEST(DensePitchBias, ApplyNoteLogitsBiasFavorsRarePitch) {
    std::vector<int32_t> data;
    appendNote(data, 10, 60);
    appendNote(data, 20, 60);
    appendNote(data, 25, 60);
    appendNote(data, 30, 62);
    const auto stats = DensePitchBias::collectPitchWindow(data, 30);

    std::vector<float> logits(DenseVocab::VocabSize, 0.0f);
    DensePitchBias::applyNoteLogitsBias(logits, stats, 1.0f, 0.0f);

    const size_t common =
        DenseVocab::NoteOffset + static_cast<size_t>(60); // instrument 0, pitch 60
    const size_t rare = DenseVocab::NoteOffset + static_cast<size_t>(62);
    const size_t otherBand =
        DenseVocab::NoteOffset + static_cast<size_t>(DenseConfig::MaxPitch) + 60;

    // Rare pitch gets a larger boost (subtracting τ·log(π) with smaller π).
    EXPECT_LT(logits[common], logits[rare]);
    // Same pitch adjustment in every instrument band.
    EXPECT_FLOAT_EQ(logits[common], logits[otherBand]);
    // Non-note vocab untouched.
    EXPECT_FLOAT_EQ(logits[0], 0.0f);
}

TEST(DensePitchBias, CollectPitchWindowAtUsesExactNow) {
    std::vector<int32_t> data;
    appendNote(data, 600, 60);
    appendNote(data, 700, 62);
    appendNote(data, 800, 64);

    // Exact prefix now = 750 (as if we just sampled onset 750): only notes ≤ 750.
    const auto at = DensePitchBias::collectPitchWindowAt(data, 750);
    EXPECT_EQ(at.noteCount, 2); // (250, 750] → 600, 700
    EXPECT_EQ(at.lastPitch, 62);
    EXPECT_EQ(at.uniquePitches, (std::vector<int32_t>{60, 62}));
}

TEST(DensePitchBias, CollectPitchWindowSkipsProvisionalDuration) {
    std::vector<int32_t> data;
    appendNote(data, 100, 60);
    // Held MIDI note still at provisional inputDuration (400) — must not enter π.
    data.push_back(200);
    data.push_back(static_cast<int32_t>(DenseVocab::DurOffset) + 400);
    data.push_back(static_cast<int32_t>(DenseVocab::NoteOffset) + 72);
    data.push_back(static_cast<int32_t>(DenseVocab::VelocityOffset) + 100);
    appendNote(data, 300, 62);

    const auto withSkip = DensePitchBias::collectPitchWindowAt(data, 300, 400);
    EXPECT_EQ(withSkip.noteCount, 2);
    EXPECT_EQ(withSkip.uniquePitches, (std::vector<int32_t>{60, 62}));
    EXPECT_EQ(withSkip.lastPitch, 62);

    const auto withoutSkip = DensePitchBias::collectPitchWindowAt(data, 300);
    EXPECT_EQ(withoutSkip.noteCount, 3);
    EXPECT_EQ(withoutSkip.lastPitch, 62);
}

TEST(DensePitchBias, ApplyNoteLogitsBiasNoopWhenEmpty) {
    DensePitchBias::PitchWindowStats empty;
    std::vector<float> logits(DenseVocab::VocabSize, 1.0f);
    DensePitchBias::applyNoteLogitsBias(logits, empty, 2.0f, 2.0f);
    EXPECT_FLOAT_EQ(logits[DenseVocab::NoteOffset], 1.0f);
}

} // namespace virtual_orch_test
