#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>

#include <VirtualOrch/orchestration-models/IodPretrained.h>

// End-to-end tests of IodPretrained. By default they run on a tiny IOD-shaped dummy graph
// (test/resources/iod_dummy.onnx, made by tools/make_dummy_iod_onnx.py), which exercises the
// whole C++ pipeline. Set VIRTUAL_ORCH_TEST_MODELS to a directory with the real iod.onnx to also
// run the same checks on it (skipped when it is missing or this ONNX Runtime cannot load it).
namespace {

auto dummyModelPath() -> std::string {
    return (std::filesystem::path(VIRTUAL_ORCH_TEST_RESOURCES) / "iod_dummy.onnx").string();
}

auto realModelPath() -> std::string {
    const char *dir = std::getenv("VIRTUAL_ORCH_TEST_MODELS");
    if (dir == nullptr)
        return {};
    const auto path = std::filesystem::path(dir) / "iod.onnx";
    return std::filesystem::exists(path) ? path.string() : std::string{};
}

auto reductionNote(int32_t onset, int32_t pitch, int32_t voice, int32_t noteId) -> Token {
    Token token{onset, static_cast<int32_t>(Vocab::DurOffset) + 60,
                static_cast<int32_t>(Vocab::NoteOffset) + pitch};
    token.velocity = 90;
    token.voiceId = voice;
    token.noteId = noteId;
    return token;
}

const std::vector<int32_t> kInstruments{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 19};

} // namespace

namespace virtual_orch_test {

void runEnsembleChecks(const std::string &path) {
    for (const bool ensemble: {false, true}) {
        IodPretrained iod;
        iod.init(path.c_str());
        ASSERT_TRUE(iod.isModelLoaded());
        iod.setVocsepEnsembleEnabled(ensemble);
        iod.currentStream = OrchestrationModel::InputStream::Reduction;
        const OrchestrationModel::ConditioningSignal signal{};

        size_t totalOut = 0;
        int32_t noteId = 0;
        for (int batch = 0; batch < 4; ++batch) {
            std::vector<Token> in;
            for (int k = 0; k < 6; ++k) {
                const int32_t i = batch * 6 + k;
                in.push_back(reductionNote(i * 25, 48 + (i * 5) % 30, i % 2, noteId++));
            }
            const auto out = iod.getOutput(in, kInstruments, signal);
            totalOut += out.size();
            for (const auto &n: out) {
                EXPECT_GE(n.localInstrumentId, 0);
                EXPECT_LT(n.localInstrumentId, 20);
                EXPECT_GE(n.token.time, 0);
            }
        }
        EXPECT_GT(totalOut, 0u) << "ensemble " << ensemble;

        const auto &list = iod.getInputNotes();
        ASSERT_EQ(list.size(), 24u);
        for (const auto &rec: list) {
            EXPECT_TRUE(rec.hasResult);
            EXPECT_FALSE(rec.keysA.empty());
            EXPECT_FALSE(rec.keysB.empty()) << "B falls back to A, never empty";
            EXPECT_EQ(rec.dataA.size(), static_cast<size_t>(IodPretrainedTypes::comboLen));
            if (! ensemble)
                EXPECT_EQ(rec.keysA, rec.keysB);
        }
    }
}

void runTwoStreamChecks(const std::string &path) {
    IodPretrained iod;
    iod.init(path.c_str());
    iod.setVocsepEnsembleEnabled(true);
    const OrchestrationModel::ConditioningSignal signal{};

    std::vector<Token> reduction, user;
    for (int i = 0; i < 6; ++i)
        reduction.push_back(reductionNote(i * 30, 60 + i, 1, 100 + i));
    for (int i = 0; i < 4; ++i) {
        Token t{i * 30 + 5, static_cast<int32_t>(Vocab::DurOffset) + 40,
                static_cast<int32_t>(Vocab::NoteOffset) + 55 + i};
        user.push_back(t);
    }
    iod.currentStream = OrchestrationModel::InputStream::User;
    EXPECT_FALSE(iod.getOutput(user, kInstruments, signal).empty());
    iod.currentStream = OrchestrationModel::InputStream::Reduction;
    EXPECT_FALSE(iod.getOutput(reduction, kInstruments, signal).empty());

    iod.applyVoiceRestamp(102, 7);
    bool found = false;
    for (const auto &rec: iod.getInputNotes()) {
        if (rec.token.noteId == 102) {
            EXPECT_EQ(rec.token.voiceId, 7);
            EXPECT_EQ(rec.stream, OrchestrationModel::InputStream::Reduction);
            found = true;
        }
    }
    EXPECT_TRUE(found);

    // A note more than 5 s after the last one clears the list (IOD's history gap rule).
    iod.getOutput({reductionNote(2000, 60, 0, 200)}, kInstruments, signal);
    EXPECT_EQ(iod.getInputNotes().size(), 1u);
}

TEST(IodPretrainedSmoke, ensembleOffAndOnProduceNotesAndFillTheSharedList) {
    runEnsembleChecks(dummyModelPath());
}

TEST(IodPretrainedSmoke, twoStreamsAndARestampDoNotBreakTheModel) {
    runTwoStreamChecks(dummyModelPath());
}

TEST(IodPretrainedReal, sameChecksOnTheRealModelWhenAvailable) {
    const auto path = realModelPath();
    if (path.empty())
        GTEST_SKIP() << "VIRTUAL_ORCH_TEST_MODELS/iod.onnx not found";
    IodPretrained probe;
    probe.init(path.c_str());
    if (! probe.isModelLoaded())
        GTEST_SKIP() << "this ONNX Runtime build cannot load the real IOD export";
    runEnsembleChecks(path);
    runTwoStreamChecks(path);
}

} // namespace virtual_orch_test
