#include <gtest/gtest.h>

#include <VirtualOrch/reduction/ReductionTransformer.h>

// AMT, V1, V2 and Piano Reduction: a held key gets the median length of its recent neighbours
// within an octave, generation does not wait, and its note-off finds the same context entry.

namespace {

/** Reduction stub with a dense (width 4) context. */
class StubReduction : public ReductionTransformer {
public:
    explicit StubReduction(ModelConfig &config) : ReductionTransformer("stub", config) {}
    [[nodiscard]] auto isModelLoaded() const -> bool override { return true; }
    [[nodiscard]] auto inputEventWidth() const -> size_t override { return 4; }
    void run() override {}
    using ReductionTransformer::isHeldInput;
    using ReductionTransformer::provisionalLengthCs;
    using ReductionTransformer::rememberHeldKey;
    using ReductionTransformer::resolveHeldKey;
    using ReductionTransformer::heldKeysForTest;

    auto add(int32_t onset, int32_t lengthCs, int32_t pitch, int32_t velocity = 64) -> void {
        inputData.push_back(onset);
        inputData.push_back(static_cast<int32_t>(Vocab::DurOffset) + lengthCs);
        inputData.push_back(static_cast<int32_t>(Vocab::NoteOffset) + pitch);
        inputData.push_back(static_cast<int32_t>(DenseVocab::VelocityOffset) + velocity);
    }
    /** The context as Tokens, like the models' applyUpdatesFromFilter builds it. */
    [[nodiscard]] auto history() const -> std::vector<Token> {
        std::vector<Token> out;
        for (size_t i = 0; i + 3 < inputData.size(); i += 4)
            out.push_back(Token{inputData[i], inputData[i + 1], inputData[i + 2], 64});
        return out;
    }
};

auto liveToken(int32_t onset, int32_t pitch, int32_t lengthCs, int32_t keyId) -> Token {
    return Token{.time = onset,
                 .duration = static_cast<int32_t>(Vocab::DurOffset) + lengthCs,
                 .note = static_cast<int32_t>(Vocab::NoteOffset) + pitch,
                 .velocity = 90,
                 .keyId = keyId};
}

auto lengthOf(const Token &t) -> int32_t { return t.getRealDuration(); }

} // namespace

namespace virtual_orch_test {

TEST(HeldKeyLength, aHeldKeyIsDirectInputWithThePlaceholderLength) {
    ModelConfig config;
    config.inputMode = InputMode::Direct;
    StubReduction rt(config);
    EXPECT_TRUE(rt.isHeldInput(liveToken(100, 60, config.inputDuration, 1)));
    EXPECT_FALSE(rt.isHeldInput(liveToken(100, 60, 120, 1))) << "released before the flush";
    config.inputMode = InputMode::Buffer;
    EXPECT_FALSE(rt.isHeldInput(liveToken(100, 60, config.inputDuration, 1)));
}

TEST(HeldKeyLength, medianOfTheLastTenSecondsWithinAnOctave) {
    ModelConfig config;
    StubReduction rt(config);
    const int32_t onset = 5000;
    rt.add(onset - 1200, 700, 60);  // older than 10 s: ignored
    rt.add(onset - 900, 50, 62);    // within an octave
    rt.add(onset - 600, 999, 85);   // more than an octave away: ignored
    rt.add(onset - 400, 300, 48);   // exactly an octave below: counted
    rt.add(onset - 100, 100, 70);
    rt.add(onset + 200, 20, 60);    // after the onset (generated ahead): ignored
    EXPECT_EQ(rt.provisionalLengthCs(onset, 60), 100) << "median of 50, 100, 300";

    rt.add(onset - 50, 200, 61); // even count: the two middles are 100 and 200
    EXPECT_EQ(rt.provisionalLengthCs(onset, 60), 150);
}

TEST(HeldKeyLength, fallsBackToAllRecentNotesThenToTheSetting) {
    ModelConfig config;
    config.inputDuration = 400;
    StubReduction rt(config);
    EXPECT_EQ(rt.provisionalLengthCs(1000, 60), 400) << "nothing in the context yet";
    rt.add(800, 80, 90);
    rt.add(900, 40, 95);
    EXPECT_EQ(rt.provisionalLengthCs(1000, 60), 60) << "none within an octave: all of them";
}

TEST(HeldKeyLength, otherHeldKeysDoNotCountTowardTheMedian) {
    ModelConfig config;
    StubReduction rt(config);
    rt.add(900, 50, 60);
    rt.add(950, 120, 64); // written for a held key
    rt.rememberHeldKey(liveToken(950, 64, config.inputDuration, 7));
    EXPECT_EQ(rt.provisionalLengthCs(1000, 62), 50);
}

TEST(HeldKeyLength, noteOffFindsTheLiveKeyEvenNextToAnIdenticalGeneratedNote) {
    ModelConfig config;
    StubReduction rt(config);
    rt.add(900, 50, 60);
    rt.add(1000, 50, 62); // the held key, written with its median length (50)
    rt.rememberHeldKey(liveToken(1000, 62, config.inputDuration, 3));
    rt.add(1000, 50, 62); // generated at the same onset, pitch and length

    auto history = rt.history();
    // Held for 1.6 s: the real length is LONGER than the provisional one.
    const TokenUpdate off{.oldNote = liveToken(1000, 62, config.inputDuration, 3),
                          .newNote = liveToken(1000, 62, 160, 3)};
    ASSERT_TRUE(rt.resolveHeldKey(history, off));
    EXPECT_EQ(lengthOf(history[1]), 160) << "the live key gets its real length";
    EXPECT_EQ(lengthOf(history[2]), 50) << "the generated twin is untouched";
    EXPECT_TRUE(rt.heldKeysForTest().empty());
    EXPECT_FALSE(rt.resolveHeldKey(history, off)) << "a second note-off is not a held key";
}

TEST(HeldKeyLength, withoutAKeyIdItMatchesByOnsetAndPitch) {
    ModelConfig config;
    StubReduction rt(config);
    rt.add(1000, 70, 62);
    rt.rememberHeldKey(liveToken(1000, 62, config.inputDuration, -1));
    rt.add(1000, 80, 65);
    rt.rememberHeldKey(liveToken(1000, 65, config.inputDuration, -1));

    auto history = rt.history();
    const TokenUpdate off{.oldNote = liveToken(1000, 65, config.inputDuration, -1),
                          .newNote = liveToken(1000, 65, 30, -1)};
    ASSERT_TRUE(rt.resolveHeldKey(history, off));
    EXPECT_EQ(lengthOf(history[0]), 70);
    EXPECT_EQ(lengthOf(history[1]), 30) << "shorter than provisional is fine too";
    ASSERT_EQ(rt.heldKeysForTest().size(), 1u);
}

TEST(HeldKeyLength, anUpdateForANoteThatWasNotHeldIsLeftToTheFieldMatch) {
    ModelConfig config;
    StubReduction rt(config);
    rt.add(1000, 70, 62);
    auto history = rt.history();
    const TokenUpdate other{.oldNote = liveToken(1000, 62, 70, 9), .newNote = liveToken(1000, 62, 90, 9)};
    EXPECT_FALSE(rt.resolveHeldKey(history, other));
    EXPECT_EQ(lengthOf(history[0]), 70);
}

} // namespace virtual_orch_test
