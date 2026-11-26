#include <gtest/gtest.h>

#include "VirtualOrch/MoodPresets.h"
#include "VirtualOrch/RegisterMix.h"

#include <cmath>
#include <vector>

using namespace RegisterMix;

namespace {

auto quad(std::vector<int32_t> &q, int32_t onset, int32_t pitch) -> void {
    q.push_back(onset);
    q.push_back(static_cast<int32_t>(DenseVocab::DurOffset) + 40);
    q.push_back(static_cast<int32_t>(DenseVocab::NoteOffset) + DenseConfig::MaxPitch + pitch); // instrument 1
    q.push_back(static_cast<int32_t>(DenseVocab::VelocityOffset) + 64);
}

constexpr int32_t kOrch0 = static_cast<int32_t>(DenseVocab::NoteOffset) + DenseConfig::MaxPitch;

} // namespace

TEST(RegisterMix, BandsAreSplitAtC3AndC5) {
    EXPECT_EQ(band(47), 0);
    EXPECT_EQ(band(48), 1); // C3
    EXPECT_EQ(band(71), 1);
    EXPECT_EQ(band(72), 2); // C5
}

TEST(RegisterMix, OffsetsMoveTowardTheTargetOncePerCommittedNote) {
    Controller c;
    const Mix target{0.2f, 0.4f, 0.4f};
    std::vector<int32_t> history;
    for (int i = 0; i < 6; ++i)
        quad(history, 10 * i, 60); // all in the middle band
    const Mix first = c.update(history, target);
    EXPECT_LT(first[1], 0.0f) << "too many middle notes: the middle band is held back";
    EXPECT_GT(first[2], 0.0f);
    EXPECT_GT(first[0], 0.0f);
    EXPECT_NEAR(first[0] + first[1] + first[2], 0.0f, 1e-5f);
    // the same history again (a precomputed branch, a second pick): no step
    const Mix again = c.update(history, target);
    EXPECT_EQ(again, first);
    // a new committed note: one more step
    quad(history, 70, 60);
    EXPECT_LT(c.update(history, target)[1], first[1]);
}

TEST(RegisterMix, NoStepWithTooFewNotesAndANewTargetStartsAtZero) {
    Controller c;
    std::vector<int32_t> history;
    for (int i = 0; i < kMinNotes - 1; ++i)
        quad(history, 10 * i, 60);
    EXPECT_EQ(c.update(history, Mix{0.5f, 0.0f, 0.5f}), Mix{});
    quad(history, 100, 60);
    EXPECT_NE(c.update(history, Mix{0.5f, 0.0f, 0.5f}), Mix{});
    EXPECT_EQ(c.update(history, Mix{0.1f, 0.8f, 0.1f}), Mix{}) << "a new target: offsets 0 until the next note";
}

TEST(RegisterMix, OffsetsStayBounded) {
    Controller c;
    std::vector<int32_t> history;
    for (int i = 0; i < 400; ++i) {
        quad(history, 10 * i, 60);
        c.update(history, Mix{1.0f, 0.0f, 0.0f});
    }
    for (const float o: c.offsets())
        EXPECT_LE(std::abs(o), kMaxOffset + 1e-4f);
}

TEST(RegisterMix, ApplyShiftsWhichRegisterButKeepsTheBlocksMass) {
    std::vector<float> logits(static_cast<size_t>(kOrch0 + 200), 0.0f);
    logits[static_cast<size_t>(kOrch0 + 130)] = 5.0f; // another token, outside the block
    const Mix offsets{1.0f, 0.0f, -1.0f};
    const auto lse = [&](const std::vector<float> &l) {
        double s = 0;
        for (int p = 0; p < 128; ++p)
            s += std::exp(static_cast<double>(l[static_cast<size_t>(kOrch0 + p)]));
        return std::log(s);
    };
    const double before = lse(logits);
    apply(logits, kOrch0, offsets, 1.0f, Mix{0.3f, 0.4f, 0.3f});
    EXPECT_NEAR(lse(logits), before, 1e-4);
    EXPECT_GT(logits[static_cast<size_t>(kOrch0 + 40)], logits[static_cast<size_t>(kOrch0 + 60)]);
    EXPECT_GT(logits[static_cast<size_t>(kOrch0 + 60)], logits[static_cast<size_t>(kOrch0 + 80)]);
    EXPECT_FLOAT_EQ(logits[static_cast<size_t>(kOrch0 + 130)], 5.0f) << "outside the block: untouched";
}

TEST(RegisterMix, TargetIsNormalisedAndClearable) {
    Target::set(Mix{1.0f, 2.0f, 1.0f});
    const auto t = Target::get();
    ASSERT_TRUE(t.has_value());
    EXPECT_FLOAT_EQ((*t)[1], 0.5f);
    Target::set(std::nullopt);
    EXPECT_FALSE(Target::get().has_value());
}

TEST(RegisterMix, EveryMoodPresetHasAMixThatSumsToOne) {
    for (const auto &p: MoodPresets::all()) {
        ASSERT_TRUE(p.registerMix.has_value()) << p.name;
        EXPECT_NEAR((*p.registerMix)[0] + (*p.registerMix)[1] + (*p.registerMix)[2], 1.0f, 1e-5f) << p.name;
    }
}

TEST(RegisterMix, AZeroShareMasksItsBandButNeverEverything) {
    std::vector<float> logits(static_cast<size_t>(kOrch0 + 128), 0.0f);
    apply(logits, kOrch0, Mix{}, 1.0f, Mix{0.0f, 0.6f, 0.4f});
    for (int p = 0; p < 128; ++p)
        EXPECT_EQ(std::isfinite(logits[static_cast<size_t>(kOrch0 + p)]), p >= kC3) << p;
    // only low pitches allowed (e.g. a register clamp): the zero share would leave nothing, so nothing is masked
    std::vector<float> lowOnly(static_cast<size_t>(kOrch0 + 128), -INFINITY);
    for (int p = 30; p < 40; ++p)
        lowOnly[static_cast<size_t>(kOrch0 + p)] = 0.0f;
    apply(lowOnly, kOrch0, Mix{}, 1.0f, Mix{0.0f, 0.6f, 0.4f});
    EXPECT_TRUE(std::isfinite(lowOnly[static_cast<size_t>(kOrch0 + 35)]));
}

TEST(RegisterMix, TheOptionIsOffByDefaultAndTurningItOffClearsTheTarget) {
    EXPECT_FALSE(Target::presetsUseMix());
    Target::setPresetsUseMix(true);
    Target::set(Mix{0.2f, 0.6f, 0.2f});
    ASSERT_TRUE(Target::get().has_value());
    Target::setPresetsUseMix(false);
    EXPECT_FALSE(Target::get().has_value());
}
