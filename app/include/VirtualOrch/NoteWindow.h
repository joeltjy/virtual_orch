#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include "VirtualOrch/MusicToken.h"

/**
 * Piano-roll widgets can only ever draw a bounded span around the clock, but the
 * histories they read from are append-only for the whole session. Filtering to the
 * drawable span before copying keeps refresh cost constant instead of growing until
 * the message thread is starved and the UI stops responding.
 */
namespace NoteWindow {

/** Widest span a roll can show; matches PianoRollView's zoom-out cap. */
inline constexpr int32_t visibleSpanCs = 6000;

/** Cap for append-only session histories (OT / RT / playback / prompt). */
inline constexpr size_t maxHistoryNotes = 256;

/** Drop the oldest entries so at most `maxNotes` remain. */
template <typename Item>
auto trimToLastNotes(std::vector<Item> &items, size_t maxNotes = maxHistoryNotes) -> void {
    if (items.size() <= maxNotes)
        return;
    items.erase(items.begin(),
                items.begin() + static_cast<std::ptrdiff_t>(items.size() - maxNotes));
}

/**
 * Same as trimToLastNotes for packed event streams (AMT = 3 ints/note, dense = 4).
 * Erases whole events only.
 */
inline auto trimStridedToLastNotes(std::vector<int32_t> &data,
                                   size_t eventWidth,
                                   size_t maxNotes = maxHistoryNotes) -> void {
    if (eventWidth == 0)
        return;
    const size_t maxInts = maxNotes * eventWidth;
    if (data.size() <= maxInts)
        return;
    const size_t drop = data.size() - maxInts;
    const size_t alignedDrop = (drop / eventWidth) * eventWidth;
    if (alignedDrop == 0)
        return;
    data.erase(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(alignedDrop));
}

/**
 * Stable-sort packed events by onset (first int of each event). Same-onset ties keep
 * their relative order. Used on the lookback copy just before model inference.
 */
inline auto sortStridedByOnset(std::vector<int32_t> &data, size_t eventWidth) -> void {
    if (eventWidth == 0 || data.size() < eventWidth * 2)
        return;
    if (data.size() % eventWidth != 0)
        return;

    const size_t n = data.size() / eventWidth;
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return data[a * eventWidth] < data[b * eventWidth];
    });

    bool alreadySorted = true;
    for (size_t i = 0; i < n; ++i) {
        if (order[i] != i) {
            alreadySorted = false;
            break;
        }
    }
    if (alreadySorted)
        return;

    std::vector<int32_t> sorted;
    sorted.reserve(data.size());
    for (const size_t idx: order) {
        const auto base = static_cast<std::ptrdiff_t>(idx * eventWidth);
        sorted.insert(sorted.end(),
                      data.begin() + base,
                      data.begin() + base + static_cast<std::ptrdiff_t>(eventWidth));
    }
    data.swap(sorted);
}

/** Earliest time (1/100s) that can still be drawn when the clock reads nowCs. */
[[nodiscard]] inline auto cutoffCs(int32_t nowCs) -> int32_t {
    return nowCs - visibleSpanCs;
}

/**
 * Copy the entries still reaching cutoff. `range` returns {onset, durationCs} for an
 * entry; a note that started earlier but is still sounding is kept.
 */
template <typename Item, typename Range>
[[nodiscard]] auto filtered(const std::vector<Item> &items, int32_t cutoff, Range range)
    -> std::vector<Item> {
    // Histories are onset-ordered and these run with a source lock held, so seek to the
    // start of the window instead of scanning (and reallocating across) the whole history.
    // A note starting a further visibleSpanCs before the cutoff would need a >60s duration
    // to still be audible, so searching from there is a safe lower bound.
    const auto searchFrom = static_cast<int64_t>(cutoff) - visibleSpanCs;
    const auto first = std::lower_bound(items.begin(),
                                        items.end(),
                                        searchFrom,
                                        [&range](const Item &item, int64_t key) {
                                            return range(item).first < key;
                                        });

    std::vector<Item> out;
    out.reserve(static_cast<size_t>(std::distance(first, items.end())));
    for (auto it = first; it != items.end(); ++it) {
        const auto span = range(*it);
        if (span.first + std::max(1, span.second) >= cutoff)
            out.push_back(*it);
    }
    return out;
}

[[nodiscard]] inline auto filteredTokens(const std::vector<Token> &tokens, int32_t cutoff)
    -> std::vector<Token> {
    return filtered(tokens, cutoff, [](const Token &token) {
        return std::pair<int32_t, int32_t>{token.time, token.getRealDuration()};
    });
}

} // namespace NoteWindow
