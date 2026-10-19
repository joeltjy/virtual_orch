#include <gtest/gtest.h>

#include <VirtualOrch/Clock.h>
#include <VirtualOrch/Metrics.h>
#include <VirtualOrch/orchestration-models/InstrumentCombinations.h>
#include <VirtualOrch/orchestration-models/TestModel.h>
#include <VirtualOrch/reduction/ReductionTransformer.h>

namespace {

using Keys = std::vector<int32_t>;

auto noteToken(int32_t onset, int32_t pitch, int32_t voiceId) -> Token {
    Token token{onset, static_cast<int32_t>(Vocab::DurOffset) + 50,
                static_cast<int32_t>(Vocab::NoteOffset) + pitch};
    token.voiceId = voiceId;
    return token;
}

class IcProbe : public InstrumentCombinations {
public:
    using OrchestrationModel::inputNotes;
    using OrchestrationModel::recordIncoming;
};

class TestModelProbe : public TestModel {
public:
    using OrchestrationModel::inputNotes;
    using OrchestrationModel::recordIncoming;
};

/** `n` sampled 9-token groups; note k uses instruments chosen[k]. */
auto sampledFor(const std::vector<Keys> &chosen) -> std::vector<int32_t> {
    std::vector<int32_t> tokens;
    int32_t onset = 0;
    for (const auto &keys: chosen) {
        const size_t base = tokens.size();
        tokens.insert(tokens.end(), {onset, 10000 + 50, 11000 + 60, 11256 + 80, 0, 0, 0, 0, 0});
        InstrumentCombinations::writeComboTokens(tokens, base, keys);
        onset += 10;
    }
    return tokens;
}

class StubReduction : public ReductionTransformer {
public:
    explicit StubReduction(ModelConfig &config) : ReductionTransformer("stub", config) {}
    [[nodiscard]] auto isModelLoaded() const -> bool override { return true; }
    void run() override {}
    using ReductionTransformer::forwardRestamp;
    using ReductionTransformer::pushOutputToken;
};

} // namespace

namespace virtual_orch_test {

TEST(InstrumentCombinationsEnsemble, comboTokensRoundTripThroughInstrumentKeys) {
    for (const Keys &keys: {Keys{0}, Keys{0, 2, 6}, Keys{7, 10}, Keys{11, 14}, Keys{15, 19},
                            Keys{0, 7, 11, 15}, Keys{3, 4, 5, 8, 9, 12, 13, 16, 17, 18}}) {
        auto tokens = sampledFor({keys});
        EXPECT_EQ(InstrumentCombinations::instrumentKeysFromTokens(tokens, 0), keys);
    }
}

TEST(InstrumentCombinationsEnsemble, writtenCombosDecodeToTheSameInstruments) {
    InstrumentCombinations ic;
    auto tokens = sampledFor({{0, 7, 12}});
    const auto notes = ic.encodedCombinationToInstruments(tokens);
    std::vector<int32_t> ids;
    for (const auto &n: notes)
        ids.push_back(n.localInstrumentId);
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ(ids, (Keys{0, 7, 12}));
}

TEST(InstrumentCombinationsEnsemble, offLeavesTheSampledChoiceUntouched) {
    IcProbe ic;
    ic.setVocsepEnsembleEnabled(false);
    std::vector<Token> in;
    for (int i = 0; i < 4; ++i)
        in.push_back(noteToken(i * 10, 60, 1));
    ic.recordIncoming(in);
    auto tokens = sampledFor({{0}, {0}, {0}, {1}});
    const auto before = tokens;
    ic.applyEnsembleToSampled(tokens, {0, 1});
    EXPECT_EQ(tokens, before);
    ASSERT_EQ(ic.inputNotes.size(), 4u);
    EXPECT_TRUE(ic.inputNotes[3].hasResult);
    EXPECT_EQ(ic.inputNotes[3].keysB, ic.inputNotes[3].keysA);
}

TEST(InstrumentCombinationsEnsemble, onSmoothsAFlickerWithinAVoice) {
    IcProbe ic;
    ic.setVocsepEnsembleEnabled(true);
    std::vector<Token> in;
    for (int i = 0; i < 8; ++i)
        in.push_back(noteToken(i * 10, 60, 1));
    ic.recordIncoming(in);
    // Seven notes on instrument 0, then one flicker on instrument 1.
    auto tokens = sampledFor({{0}, {0}, {0}, {0}, {0}, {0}, {0}, {1}});
    ic.applyEnsembleToSampled(tokens, {0, 1});
    EXPECT_EQ(InstrumentCombinations::instrumentKeysFromTokens(tokens, 7 * 9), (Keys{0}))
        << "the flicker is replaced by the voice's instrument";
    EXPECT_EQ(ic.inputNotes[7].keysA, (Keys{1}));
    EXPECT_EQ(ic.inputNotes[7].keysB, (Keys{0}));
}

TEST(InstrumentCombinationsEnsemble, notesOfDifferentVoicesAreNotMixed) {
    IcProbe ic;
    ic.setVocsepEnsembleEnabled(true);
    std::vector<Token> in;
    for (int i = 0; i < 6; ++i)
        in.push_back(noteToken(i * 10, 60, i % 2)); // alternating voices 0 / 1
    ic.recordIncoming(in);
    auto tokens = sampledFor({{0}, {7}, {0}, {7}, {0}, {7}});
    ic.applyEnsembleToSampled(tokens, {0, 7});
    for (size_t k = 0; k < 6; ++k) {
        EXPECT_EQ(InstrumentCombinations::instrumentKeysFromTokens(tokens, k * 9),
                  k % 2 == 0 ? Keys{0} : Keys{7});
    }
}

TEST(InstrumentCombinationsEnsemble, disallowedInstrumentsAreDroppedAndEmptyFallsBackToA) {
    IcProbe ic;
    ic.setVocsepEnsembleEnabled(true);
    std::vector<Token> in;
    for (int i = 0; i < 4; ++i)
        in.push_back(noteToken(i * 10, 60, 1));
    ic.recordIncoming(in);
    auto tokens = sampledFor({{0}, {0}, {0}, {0}});
    // Only instrument 5 is allowed: the smoothed choice {0} is filtered out -> fall back to A.
    ic.applyEnsembleToSampled(tokens, {5});
    EXPECT_EQ(InstrumentCombinations::instrumentKeysFromTokens(tokens, 3 * 9), (Keys{0}));
}

TEST(TestModelEnsemble, smoothsTheSingleInstrumentPerVoice) {
    TestModelProbe model;
    model.setVocsepEnsembleEnabled(true);
    std::vector<Token> in;
    for (int i = 0; i < 8; ++i)
        in.push_back(noteToken(i * 10, 60, 2));
    model.recordIncoming(in);

    std::vector<OrchestrationNote> sampled;
    for (int i = 0; i < 8; ++i) {
        OrchestrationNote n;
        n.token = in[static_cast<size_t>(i)];
        n.localInstrumentId = i == 7 ? 9 : 4; // one flicker at the end
        sampled.push_back(n);
    }
    const auto out = model.applyEnsemble(sampled, {4, 9});
    ASSERT_EQ(out.size(), 8u);
    EXPECT_EQ(out[7].localInstrumentId, 4);
}

TEST(TestModelEnsemble, offReturnsTheSampledNotes) {
    TestModelProbe model;
    model.setVocsepEnsembleEnabled(false);
    model.recordIncoming({noteToken(0, 60, 1), noteToken(10, 62, 1)});
    std::vector<OrchestrationNote> sampled(2);
    sampled[0].token = noteToken(0, 60, 1);
    sampled[0].localInstrumentId = 3;
    sampled[1].token = noteToken(10, 62, 1);
    sampled[1].localInstrumentId = 5;
    const auto out = model.applyEnsemble(sampled, {3, 5});
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].localInstrumentId, 3);
    EXPECT_EQ(out[1].localInstrumentId, 5);
}

TEST(ModelSupport, everyOrchestrationModelDeclaresEnsembleSupport) {
    InstrumentCombinations ic;
    TestModel tm;
    EXPECT_TRUE(ic.supportsVocsepEnsemble());
    EXPECT_TRUE(tm.supportsVocsepEnsemble());
}

TEST(VoiceRestampForwarding, reductionAssignsNoteIdsAndForwardsRestamps) {
    ModelConfig config;
    StubReduction rt(config);
    CircularFifo<VoiceRestamp> restamps;
    rt.orchestrationRestampsIncoming = &restamps;

    rt.pushOutputToken(noteToken(0, 60, 0));
    rt.pushOutputToken(noteToken(10, 62, 1));
    rt.pushOutputToken(noteToken(20, 64, 2));
    rt.pushOutputToken(Token{0, static_cast<int32_t>(Vocab::DurOffset),
                             static_cast<int32_t>(Vocab::ClearQueue)});

    auto history = rt.getOutputHistory();
    ASSERT_EQ(history.size(), 4u);
    EXPECT_EQ(history[0].noteId, 0);
    EXPECT_EQ(history[1].noteId, 1);
    EXPECT_EQ(history[2].noteId, 2);
    EXPECT_EQ(history[3].noteId, -1) << "ClearQueue is not a note";

    rt.forwardRestamp(1, 9);
    history = rt.getOutputHistory();
    EXPECT_EQ(history[1].voiceId, 9) << "reduction's own history is rewritten";
    VoiceRestamp r{};
    ASSERT_TRUE(restamps.pull(r));
    EXPECT_EQ(r.noteId, 1);
    EXPECT_EQ(r.newVoiceId, 9);

    rt.forwardRestamp(99, 3); // out of range: ignored
    EXPECT_FALSE(restamps.pull(r));
}

TEST(VoiceRestampForwarding, tokenHelperPatchesOnlyTheMatchingNote) {
    std::vector<Token> tokens{noteToken(0, 60, 1), noteToken(10, 62, 1)};
    tokens[0].noteId = 5;
    tokens[1].noteId = 6;
    EXPECT_EQ(applyVoiceRestampToTokens(tokens, VoiceRestamp{6, 4}), 1);
    EXPECT_EQ(tokens[0].voiceId, 1);
    EXPECT_EQ(tokens[1].voiceId, 4);
    EXPECT_EQ(applyVoiceRestampToTokens(tokens, VoiceRestamp{6, 4}), 0) << "already applied";
    EXPECT_EQ(applyVoiceRestampToTokens(tokens, VoiceRestamp{-1, 4}), 0);
}

} // namespace virtual_orch_test
