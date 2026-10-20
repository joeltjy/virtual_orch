#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include <VirtualOrch/InstrumentConstants.h>
#include <VirtualOrch/MusicToken.h>
#include <VirtualOrch/OrchestrationTransformer.h>
#include <VirtualOrch/ProjectPaths.h>
#include <VirtualOrch/orchestration-models/IodPretrained.h>

namespace virtual_orch_test {
namespace {

auto onnxPath() -> juce::File {
    return onnxExportDir().getChildFile("instrument_octave_predictions.onnx");
}

auto makeToken(int32_t onset, int32_t durCs, int32_t pitch) -> Token {
    Token t;
    t.time = onset;
    t.duration = static_cast<int32_t>(Vocab::DurOffset) + durCs;
    t.note = static_cast<int32_t>(Vocab::NoteOffset) + pitch;
    t.velocity = 80;
    return t;
}

TEST(IodPretrainedQueueWiring, NotesPushOntoOtOutputQueue) {
    const auto path = onnxPath();
    if (! path.existsAsFile()) {
        GTEST_SKIP() << "missing " << path.getFullPathName();
    }

    IodPretrained model;
    model.init(path.getFullPathName().toRawUTF8());
    if (! model.isModelLoaded()) {
        GTEST_SKIP() << "ORT session failed (need CUDA for Trilu in this export)";
    }

    IodPretrained::ConditioningSignal cond{};
    const auto notes = model.getOutput({makeToken(500, 40, 60)}, {7}, cond);
    ASSERT_FALSE(notes.empty()) << "expected at least one flute orchestration note";

    OrchestrationTransformer ot;
    ot.orchestrationModel = &model;

    auto packed = notes;
    std::sort(packed.begin(), packed.end(),
              [](const OrchestrationNote &a, const OrchestrationNote &b) {
                  if (a.token.time != b.token.time)
                      return a.token.time < b.token.time;
                  return a.token.duration < b.token.duration;
              });
    for (const auto &n: packed)
        ot.outputTokenQueue.push(n);

    size_t pulled = 0;
    OrchestrationNote got{};
    while (ot.outputTokenQueue.pull(got)) {
        EXPECT_EQ(got.localInstrumentId, 7);
        EXPECT_EQ(got.token.time, 500);
        EXPECT_GE(got.velocity, 1);
        const auto channel =
            InstrumentConstants::outputChannelForLocalInstrumentId(got.localInstrumentId,
                                                                   got.token.getPitch());
        ASSERT_TRUE(channel.has_value()) << "flute should map to a MIDI channel";
        EXPECT_EQ(*channel, 6);
        ++pulled;
    }
    EXPECT_EQ(pulled, notes.size());
}

TEST(IodPretrainedQueueWiring, EmptyPadProducesNothingOnQueue) {
    const auto path = onnxPath();
    if (! path.existsAsFile()) {
        GTEST_SKIP() << "missing " << path.getFullPathName();
    }

    IodPretrained model;
    model.init(path.getFullPathName().toRawUTF8());
    if (! model.isModelLoaded()) {
        GTEST_SKIP() << "ORT session failed (need CUDA for Trilu in this export)";
    }

    IodPretrained::ConditioningSignal cond{};
    const auto notes = model.getOutput({makeToken(0, 40, 60)}, {}, cond);
    EXPECT_TRUE(notes.empty());

    OrchestrationTransformer ot;
    for (const auto &n: notes)
        ot.outputTokenQueue.push(n);

    OrchestrationNote got{};
    EXPECT_FALSE(ot.outputTokenQueue.pull(got));
}

} // namespace
} // namespace virtual_orch_test
