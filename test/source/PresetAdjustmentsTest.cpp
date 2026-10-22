#include <gtest/gtest.h>

#include <VirtualOrch/AppSession.h>
#include <VirtualOrch/BiasSettings.h>
#include <VirtualOrch/reduction/TdOnsetBias.h>

namespace virtual_orch_test {

TEST(PresetAdjustments, aPresetSetsTheHybridsLogitAdjustmentsAndSavesThemBack) {
    const juce::ScopedJuceInitialiser_GUI gui;
    AppSession session;
    session.setMusicModelArch(MusicModelArch::DenseHybrid);
    const float strengthBefore = TdOnsetBias::strength();

    // version1.0's adjustments: W3 side off, v1 side on, LogitTD strength 5.5.
    const auto preset = juce::JSON::parse(R"({
        "biasesOn": true, "biasesOnlyWhenPianoSilent": false, "logitTdStrength": 5.5,
        "model": {"pitchBias": false, "durationBias": false, "onsetGapBias": false,
                  "shortGapPenalty": false, "logitTd": false, "pitchTau": 0.2, "waitMinPlay": 0.3},
        "hybridV1": {"pitchBias": true, "durationBias": true, "onsetGapBias": true,
                     "shortGapPenalty": true, "logitTd": false}})");
    session.applyLogitAdjustments(preset);

    auto &h = session.reductionTransformerHybrid;
    EXPECT_FLOAT_EQ(TdOnsetBias::strength(), 5.5f);
    EXPECT_TRUE(BiasSettings::enabled());
    EXPECT_FALSE(h.isPitchBiasEnabled()) << "W side";
    EXPECT_FLOAT_EQ(h.getWaitMinPlay(), 0.3f);
    EXPECT_TRUE(h.hybridV1PitchBias.load()) << "v1 side";
    EXPECT_TRUE(h.hybridV1ShortGapPenalty.load());
    EXPECT_FALSE(h.hybridV1LogitTd.load()) << "follows the preset, not the default";

    const auto saved = session.logitAdjustmentsToVar();
    EXPECT_FLOAT_EQ(static_cast<float>(saved["logitTdStrength"]), 5.5f);
    EXPECT_FALSE(static_cast<bool>(saved["model"]["pitchBias"]));
    EXPECT_TRUE(static_cast<bool>(saved["hybridV1"]["onsetGapBias"]));
    EXPECT_FALSE(static_cast<bool>(saved["hybridV1"]["logitTd"]));

    TdOnsetBias::setStrength(strengthBefore);
}

TEST(PresetAdjustments, theV1SideIsOnlySetForTheHybrid) {
    const juce::ScopedJuceInitialiser_GUI gui;
    AppSession session;
    session.setMusicModelArch(MusicModelArch::DenseAnticipatedPiano);
    session.applyLogitAdjustments(juce::JSON::parse(R"({"model": {"pitchBias": true},
                                                        "hybridV1": {"pitchBias": false}})"));
    EXPECT_TRUE(session.reductionTransformerAnticipated.isPitchBiasEnabled()) << "T3'' gets 'model'";
    EXPECT_TRUE(session.reductionTransformerHybrid.hybridV1PitchBias.load()) << "hybrid untouched";
    EXPECT_FALSE(session.logitAdjustmentsToVar().hasProperty("hybridV1"));
}

} // namespace virtual_orch_test
