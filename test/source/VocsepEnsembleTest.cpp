#include <gtest/gtest.h>

#include <random>
#include <vector>

#include <VirtualOrch/orchestration-models/OrchestrationModel.h>
#include <VirtualOrch/orchestration-models/VocsepEnsemble.h>

namespace {

using Keys = std::vector<int32_t>;

/** Verbatim logic of the former IodPretrained::ensembleComboB, minus the IOD-only filters,
 *  on plain key sets. The new shared core must reproduce it. */
struct LegacyNote {
    int32_t voiceId = -1;
    bool hasCombo = false;
    Keys a;
    Keys b;
};

auto legacyContains(const Keys &sorted, int32_t key) -> bool {
    return std::binary_search(sorted.begin(), sorted.end(), key);
}

auto legacyB(const std::vector<LegacyNote> &stream, size_t noteIdx, float addP, float removeP)
    -> Keys {
    const int32_t voiceId = stream[noteIdx].voiceId;
    Keys b0 = stream[noteIdx].a;
    if (voiceId >= 0) {
        for (size_t i = noteIdx; i-- > 0;) {
            if (stream[i].voiceId != voiceId || ! stream[i].hasCombo)
                continue;
            b0 = stream[i].b;
            break;
        }
    }
    std::vector<size_t> sameVoice;
    if (voiceId < 0) {
        sameVoice.push_back(noteIdx);
    } else {
        for (size_t i = 0; i <= noteIdx; ++i) {
            if (stream[i].voiceId != voiceId)
                continue;
            if (i < noteIdx && ! stream[i].hasCombo)
                continue;
            sameVoice.push_back(i);
        }
    }
    if (sameVoice.size() > 10)
        sameVoice.erase(sameVoice.begin(), sameVoice.end() - 10);
    const float n = static_cast<float>(sameVoice.size());
    if (n <= 0.0f)
        return stream[noteIdx].a;

    Keys candidates = b0;
    for (const size_t idx: sameVoice)
        candidates.insert(candidates.end(), stream[idx].a.begin(), stream[idx].a.end());
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

    Keys b = b0;
    std::sort(b.begin(), b.end());
    for (const int32_t key: candidates) {
        int32_t count = 0;
        for (const size_t idx: sameVoice) {
            if (legacyContains(stream[idx].a, key))
                ++count;
        }
        const float prop = static_cast<float>(count) / n;
        const bool inB = legacyContains(b, key);
        if (prop > addP && ! inB) {
            b.push_back(key);
            std::sort(b.begin(), b.end());
        } else if (prop < removeP && inB) {
            b.erase(std::remove(b.begin(), b.end(), key), b.end());
        }
    }
    return b;
}

class StubModel : public OrchestrationModel {
public:
    [[nodiscard]] auto getName() const -> std::string override { return "Stub"; }
    [[nodiscard]] auto tokenHistory() const -> const std::vector<OrchestrationNote> & override {
        return history;
    }
    [[nodiscard]] auto getOutput(const std::vector<Token> &, const std::vector<int32_t> &,
                                 const ConditioningSignal &, OrchestrationBalanceTracker *,
                                 const OrchestrationBalanceTracker::BiasView *)
        -> std::vector<OrchestrationNote> override {
        return {};
    }
    using OrchestrationModel::ensembleKeysB;
    using OrchestrationModel::inputNotes;
    using OrchestrationModel::recordIncoming;

    std::vector<OrchestrationNote> history;
};

auto noteToken(int32_t onset, int32_t pitch, int32_t voiceId = -1, int32_t noteId = -1) -> Token {
    Token token{onset, static_cast<int32_t>(Vocab::DurOffset) + 50,
                static_cast<int32_t>(Vocab::NoteOffset) + pitch};
    token.voiceId = voiceId;
    token.noteId = noteId;
    return token;
}

auto input(int32_t voiceId, Keys a, OrchestrationModel::InputStream stream =
                                       OrchestrationModel::InputStream::Reduction)
    -> OrchestrationModel::InputNote {
    OrchestrationModel::InputNote note;
    note.token = noteToken(0, 60, voiceId);
    note.stream = stream;
    note.keysA = std::move(a);
    return note;
}

} // namespace

namespace virtual_orch_test {

TEST(VocsepEnsemble, matchesTheFormerIodAlgorithmOnRandomStreams) {
    std::mt19937 rng(12345);
    for (int trial = 0; trial < 300; ++trial) {
        std::vector<LegacyNote> legacy;
        std::vector<OrchestrationModel::InputNote> notes;
        const float addP = 0.67f, removeP = 0.3f;
        for (int i = 0; i < 40; ++i) {
            const int32_t voice = static_cast<int32_t>(rng() % 4) - 1; // -1..2
            Keys a;
            const int count = 1 + static_cast<int>(rng() % 4);
            for (int k = 0; k < count; ++k)
                a.push_back(static_cast<int32_t>(rng() % 8));
            std::sort(a.begin(), a.end());
            a.erase(std::unique(a.begin(), a.end()), a.end());

            legacy.push_back(LegacyNote{voice, false, a, a});
            notes.push_back(input(voice, a));

            const Keys expected = legacyB(legacy, legacy.size() - 1, addP, removeP);
            const Keys got = VocsepEnsemble::computeKeysB(notes, notes.size() - 1, addP, removeP);
            ASSERT_EQ(got, expected) << "trial " << trial << " note " << i;

            legacy.back().b = expected;
            legacy.back().hasCombo = true;
            notes.back().keysB = got;
            notes.back().hasResult = true;
        }
    }
}

TEST(VocsepEnsemble, noteWithoutVoiceIsItsOwnVoice) {
    std::vector<OrchestrationModel::InputNote> notes;
    notes.push_back(input(-1, {3}));
    notes.back().keysB = {3};
    notes.back().hasResult = true;
    notes.push_back(input(-1, {5}));
    // share of key 5 in its own history is 1.0 > addP; previous B ({3}) is not inherited.
    EXPECT_EQ(VocsepEnsemble::computeKeysB(notes, 1, 0.67f, 0.3f), (Keys{5}));
}

TEST(VocsepEnsemble, smoothsAFlickerWithinOneVoice) {
    std::vector<OrchestrationModel::InputNote> notes;
    for (int i = 0; i < 6; ++i) {
        notes.push_back(input(1, {1, 2}));
        notes.back().keysB = VocsepEnsemble::computeKeysB(notes, notes.size() - 1, 0.67f, 0.3f);
        notes.back().hasResult = true;
    }
    notes.push_back(input(1, {1})); // a single note drops key 2
    EXPECT_EQ(VocsepEnsemble::computeKeysB(notes, notes.size() - 1, 0.67f, 0.3f), (Keys{1, 2}))
        << "key 2 is still in 6 of the last 7 A choices";
}

TEST(VocsepEnsemble, userAndReductionStreamsDoNotMixInAVoice) {
    using Stream = OrchestrationModel::InputStream;
    std::vector<OrchestrationModel::InputNote> notes;
    for (int i = 0; i < 5; ++i) {
        notes.push_back(input(2, {9}, Stream::User));
        notes.back().keysB = {9};
        notes.back().hasResult = true;
    }
    notes.push_back(input(2, {4}, Stream::Reduction));
    // Reduction note of voice 2 must not inherit B = {9} or count the user notes.
    EXPECT_EQ(VocsepEnsemble::computeKeysB(notes, notes.size() - 1, 0.67f, 0.3f), (Keys{4}));
}

TEST(OrchestrationModelList, recordsOnlySoundingNotesAndTagsTheStream) {
    StubModel model;
    model.currentStream = OrchestrationModel::InputStream::User;
    std::vector<Token> in{noteToken(0, 60), noteToken(10, 62)};
    in.push_back(Token{0, static_cast<int32_t>(Vocab::DurOffset),
                       static_cast<int32_t>(Vocab::ClearQueue)});
    EXPECT_EQ(model.recordIncoming(in), 0u);
    ASSERT_EQ(model.inputNotes.size(), 2u);
    EXPECT_EQ(model.inputNotes[0].stream, OrchestrationModel::InputStream::User);

    model.currentStream = OrchestrationModel::InputStream::Reduction;
    EXPECT_EQ(model.recordIncoming({noteToken(20, 64)}), 2u);
    EXPECT_EQ(model.inputNotes[2].stream, OrchestrationModel::InputStream::Reduction);
}

TEST(OrchestrationModelList, gapClearDropsTheListAndReportsIt) {
    StubModel model;
    model.recordIncoming({noteToken(0, 60), noteToken(100, 62)}, 500);
    bool cleared = false;
    const size_t first = model.recordIncoming({noteToken(900, 64)}, 500, &cleared);
    EXPECT_TRUE(cleared);
    EXPECT_EQ(first, 0u);
    ASSERT_EQ(model.inputNotes.size(), 1u);
    EXPECT_EQ(model.inputNotes[0].token.time, 900);
}

TEST(OrchestrationModelList, capsTheListAndShiftsTheFirstNewIndex) {
    StubModel model;
    std::vector<Token> many;
    for (int i = 0; i < 250; ++i)
        many.push_back(noteToken(i, 60));
    model.recordIncoming(many);
    std::vector<Token> more;
    for (int i = 250; i < 270; ++i)
        more.push_back(noteToken(i, 61));
    const size_t first = model.recordIncoming(more);
    EXPECT_EQ(model.inputNotes.size(), OrchestrationModel::kInputNoteCap);
    EXPECT_EQ(first, OrchestrationModel::kInputNoteCap - 20);
    EXPECT_EQ(model.inputNotes.front().token.time, 270 - static_cast<int32_t>(OrchestrationModel::kInputNoteCap));
}

TEST(OrchestrationModelList, restampUpdatesTheVoiceIdByNoteId) {
    StubModel model;
    model.recordIncoming({noteToken(0, 60, 1, 11), noteToken(10, 62, 1, 12), noteToken(20, 64, 2, 13)});
    model.applyVoiceRestamp(12, 7);
    EXPECT_EQ(model.inputNotes[0].token.voiceId, 1);
    EXPECT_EQ(model.inputNotes[1].token.voiceId, 7);
    EXPECT_EQ(model.inputNotes[2].token.voiceId, 2);
    model.applyVoiceRestamp(999, 3); // unknown id: ignored
    model.applyVoiceRestamp(-1, 3);
    EXPECT_EQ(model.inputNotes[0].token.voiceId, 1);
}

} // namespace virtual_orch_test
