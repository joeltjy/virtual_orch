#include <gtest/gtest.h>

#include <VirtualOrch/BiasSettings.h>
#include <VirtualOrch/PerformanceActivity.h>
#include <VirtualOrch/reduction/AnticipationStage.h>
#include <VirtualOrch/reduction/ReductionTransformer.h>

namespace {

struct Restore {
    ~Restore() {
        BiasSettings::setEnabled(true);
        BiasSettings::setOnlyWhenPianoSilent(false);
    }
};

class StubReduction : public ReductionTransformer {
public:
    explicit StubReduction(ModelConfig &config) : ReductionTransformer("stub", config) {}
    [[nodiscard]] auto isModelLoaded() const -> bool override { return true; }
    void run() override {}
};

/** Dense-vocab fake backend: flat logits. */
class FlatBackend : public StreamingKv::Backend {
public:
    FlatBackend() { d.layers = 1; d.heads = 1; d.headDim = 1; d.maxRows = 512; d.sinkRows = 4; }
    [[nodiscard]] auto dims() const -> StreamingKv::Dims override { return d; }
    auto run(const StreamingKv::State &past, const std::vector<int64_t> &ids,
             StreamingKv::State &present, std::vector<float> &last) -> bool override {
        present = past;
        for (const auto id: ids) {
            present.key.push_back(static_cast<float>(id));
            present.value.push_back(static_cast<float>(id));
            ++present.rows;
        }
        last.assign(11776, 0.0f);
        return true;
    }
    StreamingKv::Dims d;
};

} // namespace

namespace virtual_orch_test {

TEST(BiasSettings, offMeansNeverAndAlwaysMeansWheneverOn) {
    Restore restore;
    BiasSettings::setEnabled(false);
    EXPECT_FALSE(BiasSettings::allowsNow(0));
    EXPECT_FALSE(BiasSettings::allowsNow(60000));

    BiasSettings::setEnabled(true);
    BiasSettings::setOnlyWhenPianoSilent(false);
    EXPECT_TRUE(BiasSettings::allowsNow(0)) << "ungated: also while the piano plays";
    EXPECT_TRUE(BiasSettings::allowsNow(60000));
}

TEST(BiasSettings, theGateAllowsTheBiasesOnlyAfterMoreThanOneSecondOfPianoSilence) {
    Restore restore;
    BiasSettings::setEnabled(true);
    BiasSettings::setOnlyWhenPianoSilent(true);
    EXPECT_FALSE(BiasSettings::allowsNow(0)) << "a key is down";
    EXPECT_FALSE(BiasSettings::allowsNow(500));
    EXPECT_FALSE(BiasSettings::allowsNow(1000)) << "more than 1 s, not exactly 1 s";
    EXPECT_TRUE(BiasSettings::allowsNow(1001));
    EXPECT_TRUE(BiasSettings::allowsNow(INT64_MAX)) << "nothing played yet";
}

TEST(BiasSettings, reductionTransformersAskTheSettingsWithTheirPianoSilenceTimer) {
    Restore restore;
    ModelConfig config;
    StubReduction rt(config);
    int64_t silent = 0;
    rt.pianoSilentForMs = [&] { return silent; };

    BiasSettings::setOnlyWhenPianoSilent(true);
    EXPECT_FALSE(rt.biasesActiveNow());
    silent = 2500;
    EXPECT_TRUE(rt.biasesActiveNow());
    BiasSettings::setEnabled(false);
    EXPECT_FALSE(rt.biasesActiveNow());

}

TEST(BiasSettings, theActivityLogReportsHowLongThePianoHasBeenSilent) {
    PerformanceActivity::ActivityLog log;
    EXPECT_EQ(log.pianoSilentForMs(5000), INT64_MAX) << "nothing played";
    log.keyboardNoteOn(1000, 60);
    EXPECT_EQ(log.pianoSilentForMs(9000), 0) << "a key is down";
    log.keyboardNoteOff(2000, 60);
    EXPECT_EQ(log.pianoSilentForMs(2400), 400);
    EXPECT_EQ(log.pianoSilentForMs(4000), 2000);
    log.keyboardNoteOn(5000, 62);
    EXPECT_EQ(log.pianoSilentForMs(5100), 0);
}

TEST(BiasSettings, theStageTurnsAllThreeBiasesOffWhenTheGateSaysSo) {
    AnticipationStage stage(AnticipatedStream::Vocab::Dense, false);
    stage.setBackend(std::make_unique<FlatBackend>(), 11776);
    stage.reset(1000);
    stage.stream.addControl({1000, 40, 60, 64}, false);
    AnticipationStage::Params params;
    params.now = 1100;
    params.lowerAbs = 1100;

    params.biases = true;
    ASSERT_TRUE(stage.generate(params).has_value());
    EXPECT_TRUE(stage.sampler.pitchBias);
    EXPECT_TRUE(stage.sampler.durationBias);
    EXPECT_TRUE(stage.sampler.onsetGapBias);

    params.biases = false;
    ASSERT_TRUE(stage.generate(params).has_value());
    EXPECT_FALSE(stage.sampler.pitchBias);
    EXPECT_FALSE(stage.sampler.durationBias);
    EXPECT_FALSE(stage.sampler.onsetGapBias);
}

} // namespace virtual_orch_test
