#include <gtest/gtest.h>

#include <VirtualOrch/MusicToken.h>
#include <VirtualOrch/reduction/GroupChord.h>
#include <VirtualOrch/reduction/GroupTokenEngine.h>

// Hybrid v1.4 group tokens: the shape table must match orchestration overnight/pkg/groups4.py (the models' vocab).

TEST(GroupChord, TableMatchesGroups4) {
    // 43 shapes, 328 SHAPE x N tokens
    EXPECT_EQ(GroupChord::shape(0).name, "unison/octave");
    EXPECT_EQ(GroupChord::shape(0).n, 2);
    EXPECT_EQ(GroupChord::shape(327).name, "dim7");
    EXPECT_EQ(GroupChord::shape(327).n, 12);
    // groups4 offsets: monad 0-3, dyad +i at 4 + 5 (i - 1), triad maj root 59-66, ..., dim7 319-327
    EXPECT_EQ(GroupChord::offsetOf({0}, 2), 0);
    EXPECT_EQ(GroupChord::offsetOf({0, 4}, 3), 20);
    EXPECT_EQ(GroupChord::offsetOf({0, 4, 7}, 4), 60);
    EXPECT_EQ(GroupChord::offsetOf({0, 3, 8}, 3), 67);   // maj 1st
    EXPECT_EQ(GroupChord::offsetOf({0, 4, 7, 10}, 4), 175); // 7 root
    EXPECT_EQ(GroupChord::offsetOf({0, 3, 6, 9}, 4), 319);
    EXPECT_EQ(GroupChord::offsetOf({0, 4, 7}, 11), -1);  // N past the triad range
    EXPECT_EQ(GroupChord::offsetOf({0, 1, 2}, 3), -1);   // not a triad
    size_t total = 0;
    for (int k = 1; k <= 4; ++k)
        total += GroupChord::offsetsWithPitchClasses(k).size();
    EXPECT_EQ(total, static_cast<size_t>(GroupChord::kShapes));
    EXPECT_EQ(GroupTokenEngine::kGroupVocab, 12239);
    EXPECT_EQ(GroupTokenEngine::kContinuationVocab, 12096);
}

TEST(GroupChord, PitchClassMaskAndLabel) {
    // C3 + maj 1st (0, 3, 8) = C Eb Ab
    const auto m = GroupChord::pitchClassMask(48, GroupChord::offsetOf({0, 3, 8}, 3));
    EXPECT_EQ(m, (1u << 0) | (1u << 3) | (1u << 8));
    EXPECT_EQ(GroupChord::label(48, GroupChord::offsetOf({0, 4, 7}, 4)), "C3 maj root x4");
}

TEST(GroupChord, ExpandWithoutBankIsClosePosition) {
    GroupChord::setSharedBank(nullptr);
    Token t{100, static_cast<int32_t>(Vocab::DurOffset) + 50, static_cast<int32_t>(Vocab::NoteOffset) + 48, 80};
    t.noteId = 7;
    t.chordShape = GroupChord::offsetOf({0, 4, 7}, 4); // C major, 4 notes, root position
    const auto notes = GroupChord::expand(t);
    ASSERT_EQ(notes.size(), 4u);
    EXPECT_EQ(notes[0].getPitch(), 48);
    EXPECT_EQ(notes[1].getPitch(), 52);
    EXPECT_EQ(notes[2].getPitch(), 55);
    EXPECT_EQ(notes[3].getPitch(), 60);
    for (const auto &n: notes) {
        EXPECT_EQ(n.time, 100);
        EXPECT_EQ(n.duration, t.duration);
        EXPECT_EQ(n.velocity, 80);
        EXPECT_EQ(n.noteId, 7);
        EXPECT_EQ(n.chordShape, -1);
    }
    // a plain note passes through
    Token plain = t;
    plain.chordShape = -1;
    EXPECT_EQ(GroupChord::expand(plain).size(), 1u);
}

TEST(GroupChord, BankDecodeIsSeeded) {
    auto bank = std::make_shared<GroupChord::VoicingBankV4>();
    std::string error;
    const int32_t off = GroupChord::offsetOf({0, 4, 7}, 3);
    const std::string json = "{\"bank\": {\"" + std::to_string(off) + "\": [[[0, 7, 16], 3], [[0, 4, 19], 1]]}}";
    ASSERT_TRUE(bank->loadJson(json, error)) << error;
    const auto a = bank->decode(48, off, 11);
    const auto b = bank->decode(48, off, 11);
    EXPECT_EQ(a, b); // the same seed, the same voicing (OT and playback agree)
    ASSERT_EQ(a.size(), 3u);
    EXPECT_EQ(a[0], 48);
    EXPECT_TRUE((a == std::vector<int32_t>{48, 55, 64}) || (a == std::vector<int32_t>{48, 52, 67}));
}
