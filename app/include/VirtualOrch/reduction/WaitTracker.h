#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "VirtualOrch/reduction/InformativeWaitEngine.h"

/**
 * What the WAIT Visualizer shows, built from an InformativeWaitEngine's events (the drawing
 * model's: Informative WAIT, or Hybrid v1.1's playing side). Written on the reduction thread,
 * read by the UI (snapshot), so everything is under one lock.
 *
 *  - pending: the WAIT now waiting for the pianist. Drawn at `drawnAtCs` (the clock when the
 *    model drew it), predicted at `predictedCs` = ref + d (the event before the WAIT + its
 *    length); the window is predictedCs +- InformativeWaitEngine::kEps. A line mode's
 *    provisional WAIT after a note counts too (COUNT 1, predicted at the note's end).
 *  - results: one per closed window, newest last. Hit: the pianist played in it; `rank` is
 *    where their first note ranked in the anticipated piano pitches (1.. ; 0 = no anticipation
 *    was ready). Expired: nothing came (EXPIRED written, or a provisional WAIT removed).
 *    A hit's placeUs: the time from the pianist's key landing to the last of its COUNT orchestra
 *    notes placed (only a window the pianist's onset fired has one).
 */
class WaitTracker {
public:
    static constexpr size_t kRecent = 8;

    struct Pending {
        int32_t drawnAtCs = 0;
        int32_t predictedCs = 0;
        int32_t d = 0;
        int32_t k = 1;
        bool provisional = false;
    };
    struct Result {
        enum class Kind { Hit, Expired };
        Kind kind = Kind::Expired;
        int rank = 0; // Hit: 1.. (rank of the played pitch), 0 = no anticipation ready
        int32_t atCs = 0;
        int32_t k = 1;
        /** Hit: microseconds from the pianist's key landing to its last orchestra note placed (-1 = not yet). */
        int64_t placeUs = -1;
        int32_t placed = 0;       // orchestra notes placed on the onset
        bool precomputed = false; // all of them came from the anticipation branch
        bool fromKey = true;      // false: the key's landing was unknown, timed from the fire
    };
    struct Snapshot {
        std::optional<Pending> pending;
        std::vector<Result> recent; // oldest first
        uint64_t results = 0;       // how many windows have closed
    };

    /** One engine event; `nowCs` is the app clock when it happened. */
    auto onEvent(const InformativeWaitEngine::Event &e, int32_t nowCs) -> void {
        const std::lock_guard<std::mutex> lock(mutex);
        if (e.what == "wait") {
            pending = Pending{nowCs, e.atCs + e.a, e.a, e.b, false};
            lastRank = 0;
        } else if (e.what == "provisional") {
            pending = Pending{nowCs, e.atCs + e.a, e.a, 1, true};
            lastRank = 0;
        } else if (e.what == "anticipate" && e.b >= 1) {
            lastRank = e.b; // scored at the fire, just before "fire"
        } else if (e.what == "fire") {
            close(Result{Result::Kind::Hit, lastRank, e.atCs, e.a});
        } else if (e.what == "placed") {
            if (! recent.empty() && recent.back().kind == Result::Kind::Hit && recent.back().placeUs < 0) {
                recent.back().placeUs = e.b;
                recent.back().placed = e.a;
                recent.back().precomputed = e.text.rfind("precomputed", 0) == 0;
                recent.back().fromKey = e.text.find("|fire") == std::string::npos;
            }
        } else if (e.what == "expired" || e.what == "removed") {
            close(Result{Result::Kind::Expired, 0, e.atCs, pending.has_value() ? pending->k : e.b});
        }
    }

    /** The model stopped drawing (a Hybrid switch, a restart): its pending WAIT is gone. */
    auto cancel() -> void {
        const std::lock_guard<std::mutex> lock(mutex);
        pending.reset();
        lastRank = 0;
    }

    /** A new stream: no pending WAIT and no results. */
    auto clear() -> void {
        const std::lock_guard<std::mutex> lock(mutex);
        pending.reset();
        recent.clear();
        lastRank = 0;
        results = 0;
    }

    [[nodiscard]] auto snapshot() const -> Snapshot {
        const std::lock_guard<std::mutex> lock(mutex);
        return Snapshot{pending, std::vector<Result>(recent.begin(), recent.end()), results};
    }

private:
    auto close(const Result &r) -> void {
        pending.reset();
        lastRank = 0;
        recent.push_back(r);
        while (recent.size() > kRecent)
            recent.pop_front();
        ++results;
    }

    mutable std::mutex mutex;
    std::optional<Pending> pending;
    std::deque<Result> recent;
    int lastRank = 0;
    uint64_t results = 0;
};
