#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <VirtualOrch/MusicToken.h>
#include <VirtualOrch/reduction/DenseTypes.h>
#include <VirtualOrch/vocsep/VocsepGraph.h>

namespace virtual_orch_test {
namespace {

auto makeNote(int32_t onset, int32_t dur, int32_t pitch) -> Vocsep::Note {
    Vocsep::Note n;
    n.onsetCs = onset;
    n.durCs = dur;
    n.pitch = pitch;
    n.velocity = 100;
    return n;
}

auto hasPotEdge(const Vocsep::GraphTensors &g, int src, int dst) -> bool {
    for (int64_t e = 0; e < g.targetEdgeCount; ++e) {
        const int s = static_cast<int>(g.targetEdgeIndex[static_cast<size_t>(e)]);
        const int d =
            static_cast<int>(g.targetEdgeIndex[static_cast<size_t>(g.targetEdgeCount + e)]);
        if (s == src && d == dst)
            return true;
    }
    return false;
}

} // namespace

TEST(VocsepGraph, VoiceFeaturesDimAndPc) {
    const std::vector<Vocsep::Note> notes{makeNote(0, 50, 60), makeNote(50, 50, 64)};
    const auto g = Vocsep::buildGraph(notes);
    ASSERT_EQ(g.numNodes, 2);
    ASSERT_EQ(g.x.size(), static_cast<size_t>(2 * Vocsep::InFeats));
    // pitch 60 → pc 0, octave 5
    EXPECT_NEAR(g.x[1 + 0], 1.0f, 1e-5f);
    EXPECT_NEAR(g.x[13 + 5], 1.0f, 1e-5f);
    // pitch 64 → pc 4, octave 5
    EXPECT_NEAR(g.x[Vocsep::InFeats + 1 + 4], 1.0f, 1e-5f);
}

TEST(VocsepGraph, PitchScoreSamePitchIsOne) {
    const std::vector<Vocsep::Note> notes{makeNote(0, 50, 60), makeNote(100, 50, 60),
                                         makeNote(200, 50, 72)};
    const auto g = Vocsep::buildGraph(notes);
    ASSERT_GT(g.targetEdgeCount, 0);
    // Find edge 0→1 (same pitch)
    bool found = false;
    for (int64_t e = 0; e < g.targetEdgeCount; ++e) {
        const int s = static_cast<int>(g.targetEdgeIndex[static_cast<size_t>(e)]);
        const int d =
            static_cast<int>(g.targetEdgeIndex[static_cast<size_t>(g.targetEdgeCount + e)]);
        if (s == 0 && d == 1) {
            EXPECT_NEAR(g.pitchScore[static_cast<size_t>(e)], 1.0f, 1e-5f);
            found = true;
        }
    }
    EXPECT_TRUE(found);
}

TEST(VocsepGraph, HungarianPerfectChain) {
    // Three notes; high scores on 0→1 and 1→2.
    const int n = 3;
    std::vector<int64_t> edges{0, 1, 0, 1, 2, 2}; // src: 0,1,0 then dst: 1,2,2 → wait reshape
    // target_edge_index is [2, E]: all sources then all dests.
    // edges: (0,1), (1,2), (0,2)
    const std::vector<int64_t> target{0, 1, 0, 1, 2, 2};
    const std::vector<float> scores{0.95f, 0.92f, 0.1f};
    const auto parents = Vocsep::hungarianParents(scores, target, 3, n, 0.5f);
    ASSERT_EQ(parents.size(), 3u);
    EXPECT_EQ(parents[0], -1);
    EXPECT_EQ(parents[1], 0);
    EXPECT_EQ(parents[2], 1);
}

TEST(VocsepGraph, NoteFromTokenDensePitch) {
    Token t;
    t.time = 120;
    t.duration = static_cast<int32_t>(DenseVocab::DurOffset) + 40;
    t.note = static_cast<int32_t>(DenseVocab::NoteOffset) + 67;
    t.velocity = 90;
    t.voiceId = 2;
    const auto n = Vocsep::noteFromToken(t);
    EXPECT_EQ(n.onsetCs, 120);
    EXPECT_EQ(n.durCs, 40);
    EXPECT_EQ(n.pitch, 67);
    EXPECT_EQ(n.velocity, 90);
    EXPECT_EQ(n.voiceId, 2);
}

TEST(VocsepGraph, SameOnsetChordHasNoPotEdge) {
    // Long and short same-onset pairs must never be pot-linked (Python skips oi >= oj).
    for (const int32_t dur: {5, 50, 400}) {
        const auto g = Vocsep::buildGraph({makeNote(100, dur, 60), makeNote(100, dur, 64)});
        EXPECT_FALSE(hasPotEdge(g, 0, 1)) << "dur=" << dur;
        EXPECT_FALSE(hasPotEdge(g, 1, 0)) << "dur=" << dur;
    }
}

TEST(VocsepGraph, NearAbutReaddsPotEdge) {
    // A ends at 100, B onset 105 → |offset−onset|=5 ≤ NearConsecutiveTolCs.
    const auto g = Vocsep::buildGraph({makeNote(0, 100, 60), makeNote(105, 50, 64)});
    EXPECT_TRUE(hasPotEdge(g, 0, 1));
    EXPECT_FALSE(hasPotEdge(g, 1, 0));
}

TEST(VocsepGraph, ExactConsecutiveStillInPots) {
    // Exact abut (onset_B == offset_A) remains a pot edge via MCMA consecutive re-add.
    const auto g = Vocsep::buildGraph({makeNote(0, 100, 60), makeNote(100, 50, 64)});
    EXPECT_TRUE(hasPotEdge(g, 0, 1));
}

} // namespace virtual_orch_test
