#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>

#include <VirtualOrch/orchestration-models/IodPretrainedTD.h>

namespace {

using namespace IodPretrainedTypes;

auto dummyModelPath() -> std::string {
    return (std::filesystem::path(VIRTUAL_ORCH_TEST_RESOURCES) / "iod_td_dummy.onnx").string();
}

auto note(int32_t onset, int32_t pitch, int32_t durCs = 60, int32_t velocity = 90) -> Token {
    Token token{onset, static_cast<int32_t>(Vocab::DurOffset) + durCs,
                static_cast<int32_t>(Vocab::NoteOffset) + pitch};
    token.velocity = velocity;
    return token;
}

const std::vector<int32_t> kInstruments{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 19};

} // namespace

namespace virtual_orch_test {

TEST(IodPretrainedTD, slotsAndCompactRoundTrip) {
    const std::array<std::array<int32_t, comboLen>, 3> combos{{
        comboFromPairs({packProgramDelta(44, 0), packProgramDelta(73, 1)}), // strings + woodwind
        comboFromPairs({packProgramDelta(58, -1)}),                          // brass only
        comboFromPairs({}),                                                   // silent
    }};
    for (const auto &compact: combos) {
        const auto slots = IodPretrainedTD::slotsFromCompact(compact);
        EXPECT_EQ(IodPretrainedTD::compactFromSlots(slots), compact);
        for (const auto id: slots) {
            EXPECT_GE(id, 11384);
            EXPECT_LE(id, IodPretrainedTD::kEndNote);
        }
    }
    // strings + woodwind: GROUPS 0b0011 -> id 11384 + 3 - 1, brass/other slots are END_NOTE
    const auto slots = IodPretrainedTD::slotsFromCompact(combos[0]);
    EXPECT_EQ(slots[0], 11384 + 3 - 1);
    EXPECT_GE(slots[1], 11399);
    EXPECT_LT(slots[1], 11911);
    EXPECT_GE(slots[2], 11911);
    EXPECT_LT(slots[2], 16007);
    EXPECT_EQ(slots[3], IodPretrainedTD::kEndNote);
    EXPECT_EQ(slots[4], IodPretrainedTD::kEndNote);
}

TEST(IodPretrainedTD, quadIdsAreQ5V8AndRebased) {
    const auto quad = IodPretrainedTD::quadIds(note(1230, 60, 63, 93), 1200);
    EXPECT_EQ(quad[0], 30);
    EXPECT_EQ(quad[1], 10000 + 65);
    EXPECT_EQ(quad[2], 11000 + 60);
    EXPECT_EQ(quad[3], 11256 + 96);
    const auto loud = IodPretrainedTD::quadIds(note(0, 127, 5000, 127), 0);
    EXPECT_EQ(loud[1], 10000 + 995) << "duration clamps below 1000";
    EXPECT_EQ(loud[2], 11000 + 127);
    EXPECT_EQ(loud[3], 11256 + 120) << "velocity clamps at 120";
}

TEST(IodPretrainedTD, writesOneTdTokenPerBoundaryAfterTheNoteGroup) {
    IodPretrainedTD model;
    model.init(dummyModelPath().c_str());
    ASSERT_TRUE(model.isModelLoaded());
    int32_t level = 8;
    model.temporalDensitySource = [&] { return level; };
    model.currentStream = OrchestrationModel::InputStream::Reduction;

    // onsets 1210, 1350, 1630 (boundaries 1200, 1400, 1600 at relative 0, 200, 400)
    std::vector<Token> in{note(1210, 60), note(1350, 62)};
    (void) model.getOutput(in, kInstruments, {});
    level = 1;
    (void) model.getOutput({note(1630, 64)}, kInstruments, {});

    const auto &values = model.boundaryValues();
    ASSERT_EQ(values.size(), 3u);
    EXPECT_EQ(values.at(1200), TdTokens::iodIdForLaunchpad(8));
    EXPECT_EQ(values.at(1400), TdTokens::iodIdForLaunchpad(1)) << "read when first needed";
    EXPECT_EQ(values.at(1600), TdTokens::iodIdForLaunchpad(1));

    const auto &list = model.getInputNotes();
    ASSERT_EQ(list.size(), 3u);
    // Context for sampling a 4th note: 3 groups of 9 tokens + the TDs + its own quad.
    const auto ids = model.buildContext(0, 2, 1200);
    // note 0: quad(4) + 5 slots + TD(1200); note 1: quad + slots, no new boundary (1350 < 1400);
    // note 2: only its quad
    ASSERT_EQ(ids.size(), 4u + 5u + 1u + 4u + 5u + 4u);
    EXPECT_EQ(ids[9], TdTokens::iodIdForLaunchpad(8));
    for (size_t i = 0; i < ids.size(); ++i)
        EXPECT_LT(ids[i], IodPretrainedTD::kVocabSize);

    const auto full = model.buildContext(0, 2, 1200);
    EXPECT_EQ(full.back(), 11256 + 88) << "ends with the quad of the target (velocity id)";
}

TEST(IodPretrainedTD, autoWritesUnknown) {
    IodPretrainedTD model;
    model.init(dummyModelPath().c_str());
    ASSERT_TRUE(model.isModelLoaded());
    model.temporalDensitySource = [] { return TemporalDensity::kAuto; };
    (void) model.getOutput({note(0, 60), note(450, 64)}, kInstruments, {});
    ASSERT_EQ(model.boundaryValues().size(), 3u);
    for (const auto &[time, id]: model.boundaryValues())
        EXPECT_EQ(id, TdTokens::kIodUnknown) << time;
}

TEST(IodPretrainedTD, producesLegalNotesAndKeepsListConsistent) {
    for (const bool ensemble: {false, true}) {
        IodPretrainedTD model;
        model.init(dummyModelPath().c_str());
        ASSERT_TRUE(model.isModelLoaded());
        model.setVocsepEnsembleEnabled(ensemble);
        model.currentStream = OrchestrationModel::InputStream::Reduction;
        size_t total = 0;
        int32_t noteId = 0;
        for (int batch = 0; batch < 5; ++batch) {
            std::vector<Token> in;
            for (int k = 0; k < 5; ++k) {
                const int32_t i = batch * 5 + k;
                auto token = note(i * 30, 48 + (i * 7) % 30);
                token.voiceId = i % 2;
                token.noteId = noteId++;
                in.push_back(token);
            }
            const auto out = model.getOutput(in, kInstruments, {});
            total += out.size();
            for (const auto &n: out) {
                EXPECT_GE(n.localInstrumentId, 4);
                EXPECT_LT(n.localInstrumentId, 20);
                EXPECT_GE(n.token.getPitch(), 0);
                EXPECT_LE(n.token.getPitch(), 127);
            }
        }
        EXPECT_GT(total, 0u);
        const auto &list = model.getInputNotes();
        ASSERT_EQ(list.size(), 25u);
        for (const auto &rec: list) {
            EXPECT_TRUE(rec.hasResult);
            EXPECT_EQ(rec.dataB.size(), static_cast<size_t>(comboLen));
            EXPECT_FALSE(rec.keysB.empty());
            if (! ensemble)
                EXPECT_EQ(rec.keysA, rec.keysB);
        }
    }
}

TEST(IodPretrainedTD, longStreamsStayInsideTheContextAndOnsetRange) {
    IodPretrainedTD model;
    model.init(dummyModelPath().c_str());
    ASSERT_TRUE(model.isModelLoaded());
    // 150 notes, one every 0.4 s: 60 s of music, more than one context window.
    std::vector<Token> in;
    for (int i = 0; i < 150; ++i)
        in.push_back(note(i * 40, 40 + (i * 3) % 40));
    const auto out = model.getOutput(in, kInstruments, {});
    EXPECT_GT(out.size(), 0u);
    EXPECT_EQ(model.getInputNotes().size(), 150u);
    for (const auto &rec: model.getInputNotes())
        EXPECT_TRUE(rec.hasResult);

    // a stream beyond 100 s: every window is rebased into 0..9999
    std::vector<Token> slow;
    for (int i = 0; i < 40; ++i)
        slow.push_back(note(20000 + i * 400, 60));
    IodPretrainedTD other;
    other.init(dummyModelPath().c_str());
    (void) other.getOutput(slow, kInstruments, {});
    const auto ids = other.buildContext(0, 39, 20000);
    EXPECT_LE(ids.size(), static_cast<size_t>(IodPretrainedTD::kMaxContext) + 100u);
}

TEST(IodPretrainedTD, aGapClearsTheTdBoundaries) {
    IodPretrainedTD model;
    model.init(dummyModelPath().c_str());
    ASSERT_TRUE(model.isModelLoaded());
    (void) model.getOutput({note(0, 60), note(300, 62)}, kInstruments, {});
    EXPECT_EQ(model.boundaryValues().size(), 2u);
    (void) model.getOutput({note(5000, 64)}, kInstruments, {}); // gap > 5 s: new stream
    EXPECT_EQ(model.getInputNotes().size(), 1u);
    EXPECT_EQ(model.boundaryValues().size(), 1u);
    EXPECT_EQ(model.boundaryValues().begin()->first, 5000);
}

// Optional: the real T14 export, when VIRTUAL_ORCH_TEST_MODELS points at a directory holding it
// (skipped when missing or when this ONNX Runtime cannot load it).
TEST(IodPretrainedTD, realExportSamplesLegalNotes) {
    const char *dir = std::getenv("VIRTUAL_ORCH_TEST_MODELS");
    if (dir == nullptr)
        GTEST_SKIP() << "VIRTUAL_ORCH_TEST_MODELS not set";
    const auto path = std::filesystem::path(dir)
                      / "td_grpo_unk_T14_dummy_from_td_explicit_given_ckpt18000.onnx";
    if (! std::filesystem::exists(path))
        GTEST_SKIP() << "no real T14 export";
    IodPretrainedTD model;
    model.init(path.string().c_str());
    if (! model.isModelLoaded())
        GTEST_SKIP() << "this ONNX Runtime cannot load the export";
    std::vector<Token> in;
    for (int i = 0; i < 12; ++i)
        in.push_back(note(i * 50, 55 + (i * 3) % 20));
    const auto out = model.getOutput(in, kInstruments, {});
    EXPECT_GT(out.size(), 0u);
    for (const auto &rec: model.getInputNotes())
        EXPECT_TRUE(rec.hasResult);
}

} // namespace virtual_orch_test
