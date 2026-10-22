#include <gtest/gtest.h>

#include <VirtualOrch/orchestration-models/IodPretrainedTypes.h>

namespace virtual_orch_test {

using namespace IodPretrainedTypes;

// The 22 Oct 09:02 run: the vocsep ensemble carried "trumpet an octave down" and "flute an
// octave down" onto low notes, so they played MIDI 18..37, far below their ranges.
TEST(EnsemblePlayable, pairsOutOfRangeOrSwitchedOffAreDropped) {
    const int32_t trumpet = 56, flute = 73, trombone = 57;
    const std::vector<int32_t> active{11, 7, 13}; // trumpet, flute, trombone
    const std::vector<int32_t> pairs{packProgramDelta(trumpet, -1), packProgramDelta(flute, -1),
                                     packProgramDelta(trombone, 0), packProgramDelta(trumpet, 0)};
    // pitch 45 (A2): trumpet -12 = 33 and flute -12 = 33 are out; trombone at 45 is in;
    // trumpet at 45 is below its E3 floor.
    EXPECT_EQ(filterPairsPlayable(pairs, 45, active), (std::vector<int32_t>{packProgramDelta(trombone, 0)}));
    // pitch 76 (E5): trumpet an octave down (64) and at pitch are fine, flute -12 = 64 is fine.
    EXPECT_EQ(filterPairsPlayable(pairs, 76, active).size(), 4u);
    // the trombone switched off
    EXPECT_EQ(filterPairsPlayable(pairs, 45, {11, 7}), std::vector<int32_t>{});
}

} // namespace virtual_orch_test
