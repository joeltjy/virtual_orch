#pragma once

#include <cstdint>
#include <vector>

#include "VirtualOrch/MusicToken.h"

namespace Vocsep {

inline constexpr int InFeats = 43;
inline constexpr int VoiceFeatDim = 23;
inline constexpr int PosEncDim = 20;
inline constexpr int PotEdgesMaxDist = 16;
/**
 * Python `NEAR_CONSECUTIVE_TOL_CS`: after MCMA pots, re-add directed A→B when
 * A.onset < B.onset and |A.offset − B.onset| ≤ this (100 ms). Hetero consecutive
 * stays exact isclose (paper); this only unions pot edges.
 */
inline constexpr int NearConsecutiveTolCs = 10;
inline constexpr float Threshold = 0.3f;
inline constexpr float PitchScoreAlpha = 3.1f;
inline constexpr float CsPerBeat = 50.0f; // 120 BPM
inline constexpr float TsBeats = 4.0f;

struct Note {
    int32_t onsetCs = 0;
    int32_t durCs = 1;
    int32_t pitch = 0;
    int32_t velocity = 100;
    int32_t voiceId = -1;
};

struct GraphTensors {
    std::vector<int64_t> targetEdgeIndex; // [2, E_tgt] row-major: all sources then all dests
    int64_t targetEdgeCount = 0;
    std::vector<float> x; // [N, InFeats]
    int64_t numNodes = 0;
    std::vector<int64_t> embedEdgeIndex; // [2, E]
    int64_t embedEdgeCount = 0;
    std::vector<int64_t> edgeType; // [E]
    std::vector<float> pitchScore; // [E_tgt]
    std::vector<float> onsetScore; // [E_tgt * 2]
};

/** Build ONNX inputs for a window of notes (copied and sorted by onset). */
[[nodiscard]] auto buildGraph(const std::vector<Note> &notes) -> GraphTensors;

/**
 * Hungarian maximize on sparse pot edges; keep assignments with score > threshold.
 * Returns parent index per node (-1 if none).
 */
[[nodiscard]] auto hungarianParents(const std::vector<float> &edgeScores,
                                    const std::vector<int64_t> &targetEdgeIndex,
                                    int64_t targetEdgeCount,
                                    int numNodes,
                                    float threshold = Threshold) -> std::vector<int32_t>;

[[nodiscard]] auto noteFromToken(const Token &token) -> Note;

} // namespace Vocsep
