#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cmath>

#include <VirtualOrch/ProjectPaths.h>
#include <VirtualOrch/TdTokens.h>
#include <VirtualOrch/reduction/OnnxModelKind.h>
#include <VirtualOrch/reduction/ReductionTransformerPianoReduction.h>
#include <VirtualOrch/reduction/StreamingKvCache.h>
#include <VirtualOrch/reduction/TwoInstrumentAnticipatedReductionTransformer.h>

namespace virtual_orch_test {

namespace {

/**
 * The T3'' ONNX the app would load: app/t3_anticipated_reduction.json's "onnx" stem, looked up in
 * the app's model folders (onnx_export_* newest first). The test binary is built without
 * VIRTUAL_ORCH_APP_DIR, so the JSON is read from the repo next to this file.
 */
auto findT3Onnx(juce::String &stem) -> juce::File {
    const auto json = juce::File(__FILE__).getParentDirectory().getParentDirectory()
                          .getParentDirectory().getChildFile("app/t3_anticipated_reduction.json");
    stem = juce::JSON::parse(json.loadFileAsString()).getProperty("onnx", {}).toString();
    std::vector<juce::File> dirs{documentsModelsDir(), onnxExportDir(), onnxExportFallbackDir()};
    for (const auto &d: onnxExportDirs())
        dirs.push_back(d);
    for (const auto &d: dirs)
        if (const auto f = d.getChildFile(stem + ".onnx"); f.existsAsFile())
            return f;
    return {};
}

} // namespace

TEST(T3ModelFile, configPointsAtRopeCacheRtAndItLoads) {
    juce::String stem;
    const auto onnx = findT3Onnx(stem);
    EXPECT_EQ(stem, juce::String("rope_cache_RT"));
    if (! onnx.existsAsFile())
        GTEST_SKIP() << "rope_cache_RT.onnx is not in Documents/ or Desktop/onnx_export_* here";

    StreamingKv::OnnxBackend backend;
    std::string error;
    ASSERT_TRUE(backend.init(onnx.getFullPathName().toRawUTF8(), error)) << error;
    EXPECT_EQ(backend.vocabSize(), 11776u) << "T3'' v2 vocab (TD level 64 included)";
    EXPECT_FALSE(backend.usesPositionIds()) << "RoPE export";

    // One step from an empty cache: a TD token (level 8), then a piano note quad.
    StreamingKv::State past;
    StreamingKv::State present;
    std::vector<float> logits;
    const int64_t td = TdTokens::denseIdForLevel(8, 11776);
    const std::vector<int64_t> ids{td, 100, 10000 + 50, 11000 + 60, 11640 + 64};
    ASSERT_TRUE(backend.run(past, ids, present, logits)) << backend.lastError;
    ASSERT_EQ(logits.size(), 11776u);
    EXPECT_TRUE(std::all_of(logits.begin(), logits.end(), [](float l) { return std::isfinite(l); }));
    EXPECT_EQ(present.rows, static_cast<int>(ids.size()));
}

TEST(OnnxModelKind, kvExportsAreTheOnesWithPastInputs) {
    EXPECT_TRUE(OnnxModelKind::isKvExport({"input_ids", "past_key", "past_value"}));
    EXPECT_FALSE(OnnxModelKind::isKvExport({"input_ids"}));
    EXPECT_FALSE(OnnxModelKind::isKvExport({"input_ids", "attention_mask"}));
}

TEST(T3ModelFile, wrongPairingsAreRefusedWithAReason) {
    const auto docs = juce::File::getSpecialLocation(juce::File::userDocumentsDirectory);
    const auto kv = docs.getChildFile("onnx_export_101826/rope_cache_RT.onnx");
    const auto full = docs.getChildFile("onnx_export_092126/amt_causal_2instrument.onnx");
    if (! kv.existsAsFile() || ! full.existsAsFile())
        GTEST_SKIP() << "needs rope_cache_RT.onnx and amt_causal_2instrument.onnx";
    ModelConfig config;

    ReductionTransformerPianoReduction piano(config);
    EXPECT_FALSE(piano.init(kv.getFullPathName().toRawUTF8())) << "a KV export in Piano Reduction";
    EXPECT_FALSE(piano.isModelLoaded());
    EXPECT_NE(piano.lastInitError.find("KV-cache export"), std::string::npos) << piano.lastInitError;
    EXPECT_TRUE(piano.init(full.getFullPathName().toRawUTF8())) << piano.lastInitError;
    EXPECT_TRUE(piano.lastInitError.empty());

    TwoInstrumentAnticipatedReductionTransformer t3(config);
    EXPECT_FALSE(t3.init(full.getFullPathName().toRawUTF8())) << "a full-sequence model in T3''";
    EXPECT_NE(t3.lastInitError.find("full-sequence model"), std::string::npos) << t3.lastInitError;
}

TEST(WaitModelFiles, w2AndW3LoadOnlyAsWaitModelsAndTheHybridPairLoads) {
    const auto docs = juce::File::getSpecialLocation(juce::File::userDocumentsDirectory);
    const auto v1 = docs.getChildFile("onnx_export_101826/rope_cache_RT.onnx");
    const auto w2 = docs.getChildFile("onnx_export_102126/rope_cache_RT_v2.onnx");
    const auto w3 = docs.getChildFile("onnx_export_102126/rope_cache_RT_v3.onnx");
    if (! v1.existsAsFile() || ! w2.existsAsFile() || ! w3.existsAsFile())
        GTEST_SKIP() << "needs rope_cache_RT{,_v2,_v3}.onnx";
    using Variant = TwoInstrumentAnticipatedReductionTransformer::Variant;
    ModelConfig config;

    TwoInstrumentAnticipatedReductionTransformer t3(config, Variant::T3);
    EXPECT_FALSE(t3.init(w2.getFullPathName().toRawUTF8())) << "T3'' refuses a WAIT model";
    EXPECT_NE(t3.lastInitError.find("WAIT"), std::string::npos) << t3.lastInitError;

    TwoInstrumentAnticipatedReductionTransformer wait(config, Variant::Wait);
    EXPECT_FALSE(wait.init(v1.getFullPathName().toRawUTF8())) << "the WAIT type refuses v1";
    EXPECT_NE(wait.lastInitError.find("no WAIT"), std::string::npos) << wait.lastInitError;
    EXPECT_TRUE(wait.init(w3.getFullPathName().toRawUTF8())) << wait.lastInitError;
    EXPECT_TRUE(wait.init(w2.getFullPathName().toRawUTF8())) << wait.lastInitError;

    // W2 live: one piano note, a few steps: it plays or waits, never errors.
    int32_t now = 1000;
    wait.nowCsProvider = [&now] { return now; };
    wait.tdValueProvider = [] { return 8; };
    Token key{1000, static_cast<int32_t>(Vocab::DurOffset) + 40,
              static_cast<int32_t>(Vocab::NoteOffset) + 60};
    key.velocity = 80;
    wait.inputTokenQueue.push(key);
    int produced = 0;
    for (int i = 0; i < 20; ++i, now += 10)
        produced += wait.stepOnce() ? 1 : 0;
    std::printf("[w2] %d notes in 20 steps (the rest waited)\n", produced);
    Token t{};
    while (wait.outputTokenQueue.pull(t))
        if (t.note != static_cast<int32_t>(Vocab::ClearQueue)) {
            EXPECT_GE(t.time, 1000);
            EXPECT_GE(t.getPitch(), 21);
            EXPECT_LE(t.getPitch(), 108);
        }

    TwoInstrumentAnticipatedReductionTransformer hybrid(config, Variant::Hybrid);
    EXPECT_FALSE(hybrid.initHybrid(w2.getFullPathName().toRawUTF8(), v1.getFullPathName().toRawUTF8()))
        << "the pair the wrong way round is refused";
    EXPECT_TRUE(hybrid.initHybrid(v1.getFullPathName().toRawUTF8(), w3.getFullPathName().toRawUTF8()))
        << hybrid.lastInitError;
    EXPECT_TRUE(hybrid.isModelLoaded());
}

} // namespace virtual_orch_test
