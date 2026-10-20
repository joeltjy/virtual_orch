#include <gtest/gtest.h>

#include <vector>

#include <VirtualOrch/reduction/DenseTypes.h>

namespace virtual_orch_test {
namespace {

auto denseNote(int32_t onset, int32_t durCs, int32_t instr, int32_t pitch) -> std::vector<int32_t> {
    return {onset,
            static_cast<int32_t>(DenseVocab::DurOffset) + durCs,
            static_cast<int32_t>(DenseVocab::NoteOffset) + instr * DenseConfig::MaxPitch + pitch,
            static_cast<int32_t>(DenseVocab::VelocityOffset) + 100};
}

auto appendNote(std::vector<int32_t> &data, int32_t onset, int32_t durCs, int32_t instr,
                int32_t pitch) -> void {
    const auto n = denseNote(onset, durCs, instr, pitch);
    data.insert(data.end(), n.begin(), n.end());
}

} // namespace

TEST(DenseProvisionalWait, DetectsLiveInstrumentProvisional) {
    std::vector<int32_t> data;
    appendNote(data, 100, 400, 0, 60); // live provisional
    appendNote(data, 120, 50, 1, 64);  // generated reduction

    EXPECT_TRUE(DenseSampling::hasProvisionalDurationNotes(data, 400, 4, 0));
    EXPECT_FALSE(DenseSampling::hasProvisionalDurationNotes(data, 400, 4, 1));
}

TEST(DenseProvisionalWait, ClearedWhenLiveSnapped) {
    std::vector<int32_t> data;
    appendNote(data, 100, 80, 0, 60);
    appendNote(data, 120, 50, 1, 64);

    EXPECT_FALSE(DenseSampling::hasProvisionalDurationNotes(data, 400, 4, 0));
}

TEST(DenseProvisionalWait, GeneratedDurationAvoidsProvisionalSentinel) {
    const int32_t provisional = 400;
    const int32_t colliding =
        static_cast<int32_t>(DenseVocab::DurOffset) + provisional;
    const int32_t avoided =
        DenseSampling::snapGeneratedDurationToken(colliding, provisional);
    EXPECT_FALSE(DenseSampling::isProvisionalDurationToken(avoided, provisional));
    EXPECT_NE(avoided, colliding);
}

} // namespace virtual_orch_test
