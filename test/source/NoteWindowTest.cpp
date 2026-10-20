#include <gtest/gtest.h>

#include "VirtualOrch/NoteWindow.h"

namespace {

auto noteAt(int32_t time, int32_t durationCs) -> Token {
    Token token{};
    token.time = time;
    token.duration = static_cast<int32_t>(Vocab::DurOffset) + durationCs;
    token.note = static_cast<int32_t>(Vocab::NoteOffset) + 60;
    token.velocity = 100;
    return token;
}

} // namespace

TEST(NoteWindow, CutoffTrailsTheClockByTheVisibleSpan) {
    EXPECT_EQ(NoteWindow::cutoffCs(10000), 10000 - NoteWindow::visibleSpanCs);
}

TEST(NoteWindow, DropsNotesThatEndedBeforeTheCutoff) {
    const std::vector<Token> history{noteAt(0, 50), noteAt(900, 50), noteAt(1200, 50)};

    const auto kept = NoteWindow::filteredTokens(history, 1000);

    ASSERT_EQ(kept.size(), 1U);
    EXPECT_EQ(kept.front().time, 1200);
}

TEST(NoteWindow, KeepsNotesStartedBeforeCutoffButStillSounding) {
    const std::vector<Token> history{noteAt(500, 800)};

    EXPECT_EQ(NoteWindow::filteredTokens(history, 1000).size(), 1U);
}

TEST(NoteWindow, FilteringIsBoundedByTheWindowNotHistoryLength) {
    std::vector<Token> history;
    for (int32_t i = 0; i < 20000; ++i)
        history.push_back(noteAt(i * 10, 20));

    const int32_t now = 199990;
    const auto kept = NoteWindow::filteredTokens(history, NoteWindow::cutoffCs(now));

    EXPECT_LE(static_cast<int32_t>(kept.size()), (NoteWindow::visibleSpanCs + 20) / 10 + 1);
}

TEST(NoteWindow, SortStridedByOnsetOrdersEventsAndKeepsSameOnsetTies) {
    // Packed dense-style events: [onset, dur, note, vel] × 3, out of onset order.
    std::vector<int32_t> data{
        200, 10, 60, 100,
        100, 10, 61, 100,
        100, 10, 62, 100,
    };

    NoteWindow::sortStridedByOnset(data, 4);

    ASSERT_EQ(data.size(), 12U);
    EXPECT_EQ(data[0], 100);
    EXPECT_EQ(data[2], 61); // first of the same-onset pair preserved
    EXPECT_EQ(data[4], 100);
    EXPECT_EQ(data[6], 62);
    EXPECT_EQ(data[8], 200);
    EXPECT_EQ(data[10], 60);
}
