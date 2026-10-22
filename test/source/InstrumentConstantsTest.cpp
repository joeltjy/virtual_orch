#include <gtest/gtest.h>

#include <VirtualOrch/InstrumentConstants.h>

namespace virtual_orch_test {

TEST(InstrumentConstants, BrassVelocityReducedByDefault) {
    EXPECT_TRUE(InstrumentConstants::isBrassLocalInstrumentId(11));
    EXPECT_TRUE(InstrumentConstants::isBrassLocalInstrumentId(14));
    EXPECT_FALSE(InstrumentConstants::isBrassLocalInstrumentId(10));
    EXPECT_FALSE(InstrumentConstants::isBrassLocalInstrumentId(15));

    EXPECT_EQ(InstrumentConstants::scaleVelocityForInstrument(11, 100), 35);
    EXPECT_EQ(InstrumentConstants::scaleVelocityForInstrument(6, 100), 100); // strings ×1
    EXPECT_EQ(InstrumentConstants::scaleVelocityForInstrument(12, 0), 0);
}

TEST(InstrumentConstants, WoodwindVelocityBoostedAndCapped) {
    EXPECT_TRUE(InstrumentConstants::isWoodwindLocalInstrumentId(7));
    EXPECT_TRUE(InstrumentConstants::isWoodwindLocalInstrumentId(10));
    EXPECT_FALSE(InstrumentConstants::isWoodwindLocalInstrumentId(6));
    EXPECT_FALSE(InstrumentConstants::isWoodwindLocalInstrumentId(11));

    EXPECT_EQ(InstrumentConstants::scaleVelocityForInstrument(7, 100), 127); // 100×1.5 capped
    EXPECT_EQ(InstrumentConstants::scaleVelocityForInstrument(9, 80), 120);
    EXPECT_EQ(InstrumentConstants::scaleVelocityForInstrument(8, 60), 90);
}

TEST(InstrumentConstants, MapsTaxonomyLocalIdsToGmPrograms) {
    ASSERT_EQ(InstrumentConstants::kInstrumentMappings.size(), 20u);

    EXPECT_EQ(InstrumentConstants::toGmInstrumentId(0), 40); // violin
    EXPECT_EQ(InstrumentConstants::toGmInstrumentId(6), 48); // string ensemble 1
    EXPECT_EQ(InstrumentConstants::toGmInstrumentId(7), 73); // flute
    EXPECT_EQ(InstrumentConstants::toGmInstrumentId(8), 68); // oboe
    EXPECT_EQ(InstrumentConstants::toGmInstrumentId(9), 71); // clarinet
    EXPECT_EQ(InstrumentConstants::toGmInstrumentId(12), 60); // french horn
    EXPECT_EQ(InstrumentConstants::toGmInstrumentId(19), 52); // choir

    EXPECT_EQ(InstrumentConstants::toLocalInstrumentId(68), 8);
    EXPECT_FALSE(InstrumentConstants::toGmInstrumentId(20).has_value());
    EXPECT_FALSE(InstrumentConstants::toGmInstrumentId(-1).has_value());
}

TEST(InstrumentConstants, ContrabassHasChannelFifteen) {
    const auto channel = [](int32_t localId, int32_t pitch) {
        return InstrumentConstants::outputChannelForLocalInstrumentId(localId, pitch);
    };

    EXPECT_EQ(channel(0, 60), 1); // violin
    EXPECT_EQ(channel(1, 60), 2); // viola
    EXPECT_EQ(channel(2, 48), 3); // cello
    EXPECT_EQ(channel(3, 30), 15); // contrabass, below the cello too
    EXPECT_EQ(channel(4, 60), 4); // tremolo
    EXPECT_EQ(channel(5, 60), 5); // pizzicato
    EXPECT_EQ(channel(7, 72), 6); // flute
    EXPECT_EQ(channel(8, 72), 7); // oboe
    EXPECT_EQ(channel(9, 60), 8); // clarinet
    EXPECT_EQ(channel(15, 60), 16); // piano monitor bus

    // String Ensemble (local 6) resolves by pitch to a solo string channel.
    EXPECT_EQ(channel(6, 96), 1); // violin range
    EXPECT_EQ(channel(6, 40), 3); // cello range
    EXPECT_EQ(channel(6, 30), 15); // below the cello: contrabass

    EXPECT_FALSE(channel(17, 60).has_value()); // snare gave its channel to the contrabass
    EXPECT_FALSE(channel(18, 60).has_value()); // guitar takes no channel
    EXPECT_FALSE(channel(19, 60).has_value()); // choir takes no channel
}

TEST(InstrumentConstants, ReductionPlaybackLocalIdMapsToChannelSixteen) {
    EXPECT_EQ(InstrumentConstants::kReductionPlaybackLocalId, InstrumentConstants::kPianoLocalId);
    EXPECT_EQ(InstrumentConstants::kReductionMonitorMidiChannel, 16);
    const auto channel = InstrumentConstants::outputChannelForLocalInstrumentId(
        InstrumentConstants::kReductionPlaybackLocalId, 60);
    ASSERT_TRUE(channel.has_value());
    EXPECT_EQ(*channel, 16);
}

} // namespace virtual_orch_test
