#include <gtest/gtest.h>

#include "VirtualOrch/reduction/ForcedChordNote.h"

#include <cmath>
#include <vector>

TEST(ForcedChordNote, ShareIsTheAllowedPartOfTheModelsPitchDistribution) {
    // two blocks of 10 ids; the model puts almost everything on ids 2..4
    std::vector<float> logits(20, 0.0f);
    for (int id = 2; id <= 4; ++id)
        logits[static_cast<size_t>(id)] = 8.0f;
    const std::vector<std::pair<int32_t, int32_t>> blocks{{0, 10}, {10, 20}};
    // only ids 7..9 allowed (above the chord's last note): the model gave them almost nothing
    const double high = ForcedChordNote::share(logits, {{7, 10}}, blocks);
    EXPECT_LT(high, 0.01);
    EXPECT_LT(high, ForcedChordNote::kDefaultMinShare);
    // allowed from 3 up: most of the mass
    EXPECT_GT(ForcedChordNote::share(logits, {{3, 10}}, blocks), 0.6);
    // masked ids (-inf) count for nothing
    logits[2] = logits[3] = logits[4] = -INFINITY;
    EXPECT_NEAR(ForcedChordNote::share(logits, {{7, 10}}, blocks), 3.0 / 17.0, 1e-9);
}
