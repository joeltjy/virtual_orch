#include <gtest/gtest.h>

#include <VirtualOrch/PolyphonyCap.h>

namespace {

auto note(int32_t onset, int32_t durCs, int32_t pitch = 60) -> Token {
    return Token{onset, static_cast<int32_t>(Vocab::DurOffset) + durCs,
                 static_cast<int32_t>(Vocab::NoteOffset) + pitch};
}

} // namespace

namespace virtual_orch_test {

TEST(PolyphonyCap, noCapOrRoomLeftChangesNothing) {
    const std::vector<Token> notes{note(0, 100), note(20, 100)};
    EXPECT_EQ(PolyphonyCap::earliestOnsetCs(notes, 50, 0), 50);
    EXPECT_EQ(PolyphonyCap::earliestOnsetCs(notes, 50, 3), 50);
    EXPECT_EQ(PolyphonyCap::earliestOnsetCs({}, 50, 1), 50);
}

TEST(PolyphonyCap, fullMeansTheNextOffset) {
    const std::vector<Token> notes{note(0, 100), note(20, 60)}; // end 100 and 80
    EXPECT_EQ(PolyphonyCap::earliestOnsetCs(notes, 50, 2), 80);
    EXPECT_EQ(PolyphonyCap::earliestOnsetCs(notes, 50, 1), 100)
        << "cap 1 with two sounding: the 2nd earliest offset, not the next one";
}

TEST(PolyphonyCap, overTheCapNeedsSeveralOffsets) {
    const std::vector<Token> notes{note(0, 90), note(0, 70), note(10, 100), note(10, 30)};
    // sounding at 20, ends: 40, 70, 90, 110
    EXPECT_EQ(PolyphonyCap::earliestOnsetCs(notes, 20, 4), 40);
    EXPECT_EQ(PolyphonyCap::earliestOnsetCs(notes, 20, 3), 70) << "(4-3+1)=2nd earliest";
    EXPECT_EQ(PolyphonyCap::earliestOnsetCs(notes, 20, 2), 90) << "3rd earliest";
    EXPECT_EQ(PolyphonyCap::earliestOnsetCs(notes, 20, 1), 110) << "4th earliest";
}

TEST(PolyphonyCap, onlyNotesSoundingAtTheHeadCount) {
    const std::vector<Token> notes{note(0, 30),   // over before the head
                                   note(200, 50), // not started yet
                                   note(40, 100)};
    EXPECT_EQ(PolyphonyCap::earliestOnsetCs(notes, 50, 1), 140);
    EXPECT_EQ(PolyphonyCap::earliestOnsetCs(notes, 140, 1), 140) << "ends exactly at the head";
}

TEST(PolyphonyCap, ignoresClearTokensAndRests) {
    std::vector<Token> notes{note(0, 100)};
    notes.push_back(Token{10, static_cast<int32_t>(Vocab::DurOffset) + 500,
                          static_cast<int32_t>(Vocab::ClearQueue)});
    notes.push_back(Token{10, static_cast<int32_t>(Vocab::DurOffset) + 500,
                          static_cast<int32_t>(Vocab::Rest)});
    EXPECT_EQ(PolyphonyCap::earliestOnsetCs(notes, 50, 2), 50);
}

} // namespace virtual_orch_test
