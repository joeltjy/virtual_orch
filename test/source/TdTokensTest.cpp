#include <gtest/gtest.h>

#include <VirtualOrch/LaunchpadGrid.h>
#include <VirtualOrch/LaunchpadLighting.h>
#include <VirtualOrch/TdTokens.h>

namespace virtual_orch_test {

TEST(TemporalDensityLevels, includesSixtyFourAndAuto) {
    EXPECT_EQ(TemporalDensity::kLevels.size(), 7u);
    EXPECT_EQ(TemporalDensity::kLevels.back(), 64);
    for (const int32_t level: TemporalDensity::kLevels)
        EXPECT_TRUE(TemporalDensity::isValid(level));
    EXPECT_TRUE(TemporalDensity::isValid(TemporalDensity::kAuto));
    EXPECT_FALSE(TemporalDensity::isValid(3));
    EXPECT_FALSE(TemporalDensity::isValid(128));
}

TEST(LaunchpadTd, sevenPadsOnRowOneAndEachHasAColour) {
    EXPECT_EQ(LaunchpadGrid::kTdLevelCount, 7);
    for (int col = 0; col < LaunchpadGrid::kTdLevelCount; ++col) {
        const auto idx = LaunchpadGrid::tdLevelIndexForPad(LaunchpadGrid::kTdRow, col);
        ASSERT_TRUE(idx.has_value());
        EXPECT_EQ(*idx, col);
        EXPECT_NE(LaunchpadLighting::colourForTdLevel(col, true), LaunchpadLighting::kOff);
        EXPECT_EQ(LaunchpadLighting::colourForTdLevel(col, false), LaunchpadLighting::kOff);
    }
    EXPECT_FALSE(LaunchpadGrid::tdLevelIndexForPad(LaunchpadGrid::kTdRow, 7).has_value());
    EXPECT_FALSE(LaunchpadGrid::tdLevelIndexForPad(0, 0).has_value());
}

TEST(TdTokens, denseIdsFollowTheVocabWithAndWithoutLevel64) {
    EXPECT_EQ(TdTokens::denseIdForLevel(0, 11775), 11768);
    EXPECT_EQ(TdTokens::denseIdForLevel(1, 11775), 11769);
    EXPECT_EQ(TdTokens::denseIdForLevel(32, 11775), 11774);
    EXPECT_EQ(TdTokens::denseIdForLevel(64, 11775), 11774) << "no id for 64: clamps to 32";
    EXPECT_EQ(TdTokens::denseIdForLevel(64, 11776), 11775);
    EXPECT_EQ(TdTokens::denseIdForLevel(3, 11776), -1);
}

TEST(TdTokens, denseAutoRepeatsTheLastChosenLevel) {
    TdTokens::DenseTdResolver resolver;
    EXPECT_EQ(resolver.resolve(TemporalDensity::kAuto, 11776),
              TdTokens::denseIdForLevel(8, 11776))
        << "default before any choice";
    EXPECT_EQ(resolver.resolve(16, 11776), TdTokens::denseIdForLevel(16, 11776));
    EXPECT_EQ(resolver.resolve(TemporalDensity::kAuto, 11776), TdTokens::denseIdForLevel(16, 11776));
    EXPECT_EQ(resolver.resolve(64, 11775), TdTokens::denseIdForLevel(32, 11775));
}

TEST(TdTokens, amtNativeIdsFollowTheSameLevels) {
    EXPECT_EQ(TdTokens::amtIdForLevel(0, 55293), 55285);
    EXPECT_EQ(TdTokens::amtIdForLevel(8, 55293), 55289);
    EXPECT_EQ(TdTokens::amtIdForLevel(64, 55293), 55292);
    EXPECT_EQ(TdTokens::amtIdForLevel(3, 55293), -1);
    TdTokens::AmtTdResolver resolver;
    EXPECT_EQ(resolver.resolve(TemporalDensity::kAuto, 55293), 55289) << "default level 8";
    EXPECT_EQ(resolver.resolve(32, 55293), 55291);
    EXPECT_EQ(resolver.resolve(TemporalDensity::kAuto, 55293), 55291) << "AUTO repeats the last level";
}

TEST(TdTokens, iodIdsAndUnknown) {
    EXPECT_EQ(TdTokens::iodIdForLaunchpad(TemporalDensity::kAuto), 20177);
    EXPECT_EQ(TdTokens::iodIdForLaunchpad(1), 20170);
    EXPECT_EQ(TdTokens::iodIdForLaunchpad(2), 20171);
    EXPECT_EQ(TdTokens::iodIdForLaunchpad(8), 20173);
    EXPECT_EQ(TdTokens::iodIdForLaunchpad(32), 20175);
    EXPECT_EQ(TdTokens::iodIdForLaunchpad(64), 20175) << "no id for 64: written as 32";
    EXPECT_EQ(TdTokens::iodIdForLaunchpad(5), 20177);
    EXPECT_EQ(TdTokens::kBoundaryCs % 200, 0);
}

} // namespace virtual_orch_test
