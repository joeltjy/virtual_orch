#include "VirtualOrch/vocsep/VocsepGraph.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <utility>

namespace Vocsep {
namespace {

[[nodiscard]] auto isclose(float a, float b, float atol = 1e-4f) -> bool {
    return std::abs(a - b) <= atol;
}

/** Dense symmetric Jacobi eigendecomposition; eigenvectors as columns of V. */
auto jacobiEigensystem(std::vector<float> &A, int n, std::vector<float> &eigenvalues,
                       std::vector<float> &V) -> void {
    eigenvalues.assign(static_cast<size_t>(n), 0.0f);
    V.assign(static_cast<size_t>(n * n), 0.0f);
    for (int i = 0; i < n; ++i)
        V[static_cast<size_t>(i * n + i)] = 1.0f;

    constexpr int maxSweeps = 40;
    for (int sweep = 0; sweep < maxSweeps; ++sweep) {
        float off = 0.0f;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q)
                off += std::abs(A[static_cast<size_t>(p * n + q)]);
        }
        if (off < 1e-8f)
            break;

        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const float apq = A[static_cast<size_t>(p * n + q)];
                if (std::abs(apq) < 1e-12f)
                    continue;
                const float app = A[static_cast<size_t>(p * n + p)];
                const float aqq = A[static_cast<size_t>(q * n + q)];
                const float tau = (aqq - app) / (2.0f * apq);
                const float t =
                    (tau >= 0.0f ? 1.0f : -1.0f)
                    / (std::abs(tau) + std::sqrt(1.0f + tau * tau));
                const float c = 1.0f / std::sqrt(1.0f + t * t);
                const float s = t * c;

                for (int k = 0; k < n; ++k) {
                    if (k == p || k == q)
                        continue;
                    const float akp = A[static_cast<size_t>(k * n + p)];
                    const float akq = A[static_cast<size_t>(k * n + q)];
                    A[static_cast<size_t>(k * n + p)] = A[static_cast<size_t>(p * n + k)] =
                        c * akp - s * akq;
                    A[static_cast<size_t>(k * n + q)] = A[static_cast<size_t>(q * n + k)] =
                        s * akp + c * akq;
                }
                A[static_cast<size_t>(p * n + p)] = app - t * apq;
                A[static_cast<size_t>(q * n + q)] = aqq + t * apq;
                A[static_cast<size_t>(p * n + q)] = A[static_cast<size_t>(q * n + p)] = 0.0f;

                for (int k = 0; k < n; ++k) {
                    const float vip = V[static_cast<size_t>(k * n + p)];
                    const float viq = V[static_cast<size_t>(k * n + q)];
                    V[static_cast<size_t>(k * n + p)] = c * vip - s * viq;
                    V[static_cast<size_t>(k * n + q)] = s * vip + c * viq;
                }
            }
        }
    }

    for (int i = 0; i < n; ++i)
        eigenvalues[static_cast<size_t>(i)] = A[static_cast<size_t>(i * n + i)];
}

auto laplacianPosEnc(const std::vector<int64_t> &edgeIndex, int64_t edgeCount, int numNodes,
                     int posEncDim) -> std::vector<float> {
    std::vector<float> pe(static_cast<size_t>(numNodes * posEncDim), 0.0f);
    if (numNodes <= posEncDim + 1 || edgeCount <= 0)
        return pe;

    std::vector<float> A(static_cast<size_t>(numNodes * numNodes), 0.0f);
    for (int64_t e = 0; e < edgeCount; ++e) {
        const int src = static_cast<int>(edgeIndex[static_cast<size_t>(e)]);
        const int dst = static_cast<int>(edgeIndex[static_cast<size_t>(edgeCount + e)]);
        if (src < 0 || dst < 0 || src >= numNodes || dst >= numNodes)
            continue;
        A[static_cast<size_t>(src * numNodes + dst)] = 1.0f;
    }

    std::vector<float> deg(static_cast<size_t>(numNodes), 0.0f);
    for (int i = 0; i < numNodes; ++i) {
        float s = 0.0f;
        for (int j = 0; j < numNodes; ++j)
            s += A[static_cast<size_t>(i * numNodes + j)];
        deg[static_cast<size_t>(i)] = s;
    }

    std::vector<float> L(static_cast<size_t>(numNodes * numNodes), 0.0f);
    for (int i = 0; i < numNodes; ++i) {
        const float di = std::sqrt(std::max(deg[static_cast<size_t>(i)], 1.0f));
        for (int j = 0; j < numNodes; ++j) {
            const float dj = std::sqrt(std::max(deg[static_cast<size_t>(j)], 1.0f));
            const float aij = A[static_cast<size_t>(i * numNodes + j)];
            float v = (i == j ? 1.0f : 0.0f) - aij / (di * dj);
            L[static_cast<size_t>(i * numNodes + j)] = v;
        }
    }

    std::vector<float> evals;
    std::vector<float> V;
    try {
        jacobiEigensystem(L, numNodes, evals, V);
    } catch (...) {
        return pe;
    }

    std::vector<int> order(static_cast<size_t>(numNodes));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return evals[static_cast<size_t>(a)] < evals[static_cast<size_t>(b)];
    });

    // Skip the constant eigenvector (smallest); take next posEncDim.
    for (int d = 0; d < posEncDim; ++d) {
        const int col = order[static_cast<size_t>(std::min(d + 1, numNodes - 1))];
        for (int i = 0; i < numNodes; ++i)
            pe[static_cast<size_t>(i * posEncDim + d)] = V[static_cast<size_t>(i * numNodes + col)];
    }
    return pe;
}

auto voiceFeatures(const std::vector<Note> &notes) -> std::vector<float> {
    const int n = static_cast<int>(notes.size());
    std::vector<float> x(static_cast<size_t>(n * VoiceFeatDim), 0.0f);
    for (int i = 0; i < n; ++i) {
        const float durBeat = static_cast<float>(std::max(notes[static_cast<size_t>(i)].durCs, 1))
                              / CsPerBeat;
        const float durFeat = 1.0f - std::tanh(durBeat / TsBeats);
        float *row = x.data() + static_cast<size_t>(i * VoiceFeatDim);
        row[0] = durFeat;
        const int pitch = std::clamp(notes[static_cast<size_t>(i)].pitch, 0, 127);
        row[1 + (pitch % 12)] = 1.0f;
        const int octave = std::clamp(pitch / 12, 0, 9);
        row[13 + octave] = 1.0f;
    }
    return x;
}

struct HeteroEdges {
    std::vector<int64_t> index; // [2, E]
    std::vector<int64_t> type;  // [E]
    int64_t count = 0;
};

auto buildHeteroEdges(const std::vector<Note> &notes) -> HeteroEdges {
    const int n = static_cast<int>(notes.size());
    std::vector<int> onsetDiv(static_cast<size_t>(n));
    std::vector<int> durDiv(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        onsetDiv[static_cast<size_t>(i)] = notes[static_cast<size_t>(i)].onsetCs;
        durDiv[static_cast<size_t>(i)] = std::max(notes[static_cast<size_t>(i)].durCs, 1);
    }

    std::vector<int64_t> src;
    std::vector<int64_t> dst;
    std::vector<int64_t> etype;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j)
                continue;
            if (isclose(static_cast<float>(onsetDiv[static_cast<size_t>(j)]),
                        static_cast<float>(onsetDiv[static_cast<size_t>(i)]))) {
                src.push_back(i);
                dst.push_back(j);
                etype.push_back(0); // onset
            }
        }
        const int end = onsetDiv[static_cast<size_t>(i)] + durDiv[static_cast<size_t>(i)];
        for (int j = 0; j < n; ++j) {
            if (i == j)
                continue;
            // Paper consecutive: onset_B ≈ offset_A (exact isclose). Near-abut slack
            // is applied later as a pot-edge union, not here.
            if (isclose(static_cast<float>(onsetDiv[static_cast<size_t>(j)]),
                        static_cast<float>(end))) {
                src.push_back(i);
                dst.push_back(j);
                etype.push_back(1); // consecutive
            }
        }
        for (int j = 0; j < n; ++j) {
            // During: B starts while A still sounds (MCMA strips these from pot).
            if (onsetDiv[static_cast<size_t>(i)] < onsetDiv[static_cast<size_t>(j)]
                && end > onsetDiv[static_cast<size_t>(j)]) {
                src.push_back(i);
                dst.push_back(j);
                etype.push_back(2); // during
            }
        }
    }

    // Rest edges: silent gaps between end times and next onsets.
    std::vector<int> endTimes;
    endTimes.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        endTimes.push_back(onsetDiv[static_cast<size_t>(i)] + durDiv[static_cast<size_t>(i)]);
    std::sort(endTimes.begin(), endTimes.end());
    endTimes.erase(std::unique(endTimes.begin(), endTimes.end()), endTimes.end());
    for (size_t ei = 0; ei + 1 < endTimes.size(); ++ei) {
        const int et = endTimes[ei];
        bool etIsOnset = false;
        for (int j = 0; j < n; ++j) {
            if (onsetDiv[static_cast<size_t>(j)] == et) {
                etIsOnset = true;
                break;
            }
        }
        if (etIsOnset)
            continue;
        std::vector<int> scr;
        for (int i = 0; i < n; ++i) {
            if (onsetDiv[static_cast<size_t>(i)] + durDiv[static_cast<size_t>(i)] == et)
                scr.push_back(i);
        }
        int minDiff = std::numeric_limits<int>::max();
        for (int j = 0; j < n; ++j) {
            const int diff = onsetDiv[static_cast<size_t>(j)] - et;
            if (diff > 0)
                minDiff = std::min(minDiff, diff);
        }
        if (minDiff == std::numeric_limits<int>::max())
            continue;
        std::vector<int> destinations;
        for (int j = 0; j < n; ++j) {
            if (onsetDiv[static_cast<size_t>(j)] - et == minDiff)
                destinations.push_back(j);
        }
        for (int i: scr) {
            for (int j: destinations) {
                src.push_back(i);
                dst.push_back(j);
                etype.push_back(3); // rest
            }
        }
    }

    // Reverse edges for types 1,2,3 → new types 4,5,6 (skip onset=0).
    const size_t baseCount = src.size();
    int64_t nextType = 3;
    for (int64_t type = 1; type <= 3; ++type) {
        ++nextType;
        for (size_t e = 0; e < baseCount; ++e) {
            if (etype[e] != type)
                continue;
            src.push_back(dst[e]);
            dst.push_back(src[e]);
            etype.push_back(nextType);
        }
    }

    HeteroEdges out;
    out.count = static_cast<int64_t>(src.size());
    out.index.resize(static_cast<size_t>(out.count * 2));
    out.type = std::move(etype);
    for (int64_t e = 0; e < out.count; ++e) {
        out.index[static_cast<size_t>(e)] = src[static_cast<size_t>(e)];
        out.index[static_cast<size_t>(out.count + e)] = dst[static_cast<size_t>(e)];
    }
    return out;
}

auto buildPotEdges(int numNodes, const HeteroEdges &hetero, int maxDist) -> std::vector<std::pair<int, int>> {
    std::vector<uint8_t> onsetDuring(static_cast<size_t>(numNodes * numNodes), 0);
    std::vector<uint8_t> consecutive(static_cast<size_t>(numNodes * numNodes), 0);
    for (int64_t e = 0; e < hetero.count; ++e) {
        const int s = static_cast<int>(hetero.index[static_cast<size_t>(e)]);
        const int d = static_cast<int>(hetero.index[static_cast<size_t>(hetero.count + e)]);
        const int t = static_cast<int>(hetero.type[static_cast<size_t>(e)]);
        if (s < 0 || d < 0 || s >= numNodes || d >= numNodes)
            continue;
        if (t == 0 || t == 2)
            onsetDuring[static_cast<size_t>(s * numNodes + d)] = 1;
        if (t == 1)
            consecutive[static_cast<size_t>(s * numNodes + d)] = 1;
    }

    std::vector<std::pair<int, int>> pots;
    for (int i = 0; i < numNodes; ++i) {
        for (int j = i + 1; j < numNodes; ++j) {
            // MCMA: strip onset/during, drop |j-i|>=max_dist, re-add exact consecutive only.
            // Exact consecutive never coincides with onset/during, so this does not undo the strip.
            bool keep = onsetDuring[static_cast<size_t>(i * numNodes + j)] == 0;
            if ((j - i) >= maxDist)
                keep = false;
            if (consecutive[static_cast<size_t>(i * numNodes + j)] != 0)
                keep = true;
            if (keep)
                pots.emplace_back(i, j);
        }
    }
    return pots;
}

/** Python `readd_near_consecutive_pot_edges`: union A→B when A.onset < B.onset and |offset_A-onset_B|≤tol. */
auto readdNearConsecutivePotEdges(std::vector<std::pair<int, int>> pots,
                                  const std::vector<Note> &notes,
                                  int tolCs) -> std::vector<std::pair<int, int>> {
    const int n = static_cast<int>(notes.size());
    if (n <= 1 || tolCs < 0)
        return pots;

    std::set<std::pair<int, int>> existing(pots.begin(), pots.end());
    for (int i = 0; i < n; ++i) {
        const int onsetI = notes[static_cast<size_t>(i)].onsetCs;
        const int offsetI =
            onsetI + std::max(notes[static_cast<size_t>(i)].durCs, 1);
        for (int j = 0; j < n; ++j) {
            if (i == j)
                continue;
            const int onsetJ = notes[static_cast<size_t>(j)].onsetCs;
            if (onsetI >= onsetJ)
                continue;
            if (std::abs(offsetI - onsetJ) > tolCs)
                continue;
            const auto key = std::pair{i, j};
            if (existing.insert(key).second)
                pots.push_back(key);
        }
    }
    return pots;
}

auto pitchScores(const std::vector<Note> &notes,
                 const std::vector<std::pair<int, int>> &pots) -> std::vector<float> {
    std::vector<float> fpitch(notes.size());
    constexpr float a = 440.0f;
    for (size_t i = 0; i < notes.size(); ++i) {
        const float mpitch = static_cast<float>(notes[i].pitch);
        fpitch[i] = (a / 32.0f) * std::pow(2.0f, (mpitch - 9.0f) / 12.0f);
    }
    std::vector<float> out(pots.size());
    for (size_t e = 0; e < pots.size(); ++e) {
        const float lo = std::min(fpitch[static_cast<size_t>(pots[e].first)],
                                  fpitch[static_cast<size_t>(pots[e].second)]);
        const float hi = std::max(fpitch[static_cast<size_t>(pots[e].first)],
                                  fpitch[static_cast<size_t>(pots[e].second)]);
        out[e] = std::pow(lo / std::max(hi, 1e-8f), PitchScoreAlpha);
    }
    return out;
}

auto onsetScores(const std::vector<Note> &notes,
                 const std::vector<std::pair<int, int>> &pots) -> std::vector<float> {
    const int n = static_cast<int>(notes.size());
    std::vector<float> onset(static_cast<size_t>(n));
    std::vector<float> duration(static_cast<size_t>(n));
    std::vector<float> onsetBeat(static_cast<size_t>(n));
    std::vector<float> durationBeat(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        onset[static_cast<size_t>(i)] = static_cast<float>(notes[static_cast<size_t>(i)].onsetCs);
        duration[static_cast<size_t>(i)] =
            static_cast<float>(std::max(notes[static_cast<size_t>(i)].durCs, 1));
        onsetBeat[static_cast<size_t>(i)] = onset[static_cast<size_t>(i)] / CsPerBeat;
        durationBeat[static_cast<size_t>(i)] = duration[static_cast<size_t>(i)] / CsPerBeat;
    }

    std::vector<float> out(pots.size() * 2);
    for (size_t e = 0; e < pots.size(); ++e) {
        const int s = pots[e].first;
        const int d = pots[e].second;
        const float offsetBeat = onsetBeat[static_cast<size_t>(s)] + durationBeat[static_cast<size_t>(s)];
        const float noteDistanceBeat = onsetBeat[static_cast<size_t>(d)] - offsetBeat;
        const float oscore =
            1.0f
            - (1.0f / (1.0f + std::exp(-2.0f * (noteDistanceBeat / TsBeats))) - 0.5f) * 2.0f;
        const float offset = onset[static_cast<size_t>(s)] + duration[static_cast<size_t>(s)];
        const float legato =
            (onset[static_cast<size_t>(d)] == offset) ? 1.0f : 0.0f;
        out[e * 2] = oscore;
        out[e * 2 + 1] = legato;
    }
    return out;
}

/**
 * Hungarian maximize on dense cost (n x n). Returns assignment col for each row (-1 unused).
 * Uses Kuhn–Munkres (Jonker–Volgenant-ish dense variant).
 */
auto hungarianMaximize(const std::vector<float> &cost, int n) -> std::vector<int> {
    // Convert maximize → minimize by negating.
    constexpr float inf = 1e9f;
    std::vector<float> u(static_cast<size_t>(n + 1), 0.0f);
    std::vector<float> v(static_cast<size_t>(n + 1), 0.0f);
    std::vector<int> p(static_cast<size_t>(n + 1), 0);
    std::vector<int> way(static_cast<size_t>(n + 1), 0);

    auto at = [&](int i, int j) -> float {
        // 1-based i,j into 0-based matrix; use -cost for maximization.
        return -cost[static_cast<size_t>((i - 1) * n + (j - 1))];
    };

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<float> minv(static_cast<size_t>(n + 1), inf);
        std::vector<char> used(static_cast<size_t>(n + 1), false);
        do {
            used[static_cast<size_t>(j0)] = true;
            const int i0 = p[static_cast<size_t>(j0)];
            float delta = inf;
            int j1 = 0;
            for (int j = 1; j <= n; ++j) {
                if (used[static_cast<size_t>(j)])
                    continue;
                const float cur = at(i0, j) - u[static_cast<size_t>(i0)] - v[static_cast<size_t>(j)];
                if (cur < minv[static_cast<size_t>(j)]) {
                    minv[static_cast<size_t>(j)] = cur;
                    way[static_cast<size_t>(j)] = j0;
                }
                if (minv[static_cast<size_t>(j)] < delta) {
                    delta = minv[static_cast<size_t>(j)];
                    j1 = j;
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[static_cast<size_t>(j)]) {
                    u[static_cast<size_t>(p[static_cast<size_t>(j)])] += delta;
                    v[static_cast<size_t>(j)] -= delta;
                } else {
                    minv[static_cast<size_t>(j)] -= delta;
                }
            }
            j0 = j1;
        } while (p[static_cast<size_t>(j0)] != 0);
        do {
            const int j1 = way[static_cast<size_t>(j0)];
            p[static_cast<size_t>(j0)] = p[static_cast<size_t>(j1)];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> assignment(static_cast<size_t>(n), -1);
    for (int j = 1; j <= n; ++j) {
        if (p[static_cast<size_t>(j)] != 0)
            assignment[static_cast<size_t>(p[static_cast<size_t>(j)] - 1)] = j - 1;
    }
    return assignment;
}

} // namespace

auto noteFromToken(const Token &token) -> Note {
    Note n;
    n.onsetCs = token.time;
    n.durCs = std::max(token.getRealDuration(), 1);
    n.pitch = std::clamp(token.getPitch(), 0, 127);
    n.velocity = token.velocity > 0 ? token.velocity : 100;
    n.voiceId = token.voiceId;
    return n;
}

auto buildGraph(const std::vector<Note> &notesIn) -> GraphTensors {
    // Keep caller order (window is already chronological). Sorting would break parent indices.
    const std::vector<Note> &notes = notesIn;
    GraphTensors g;
    const int n = static_cast<int>(notes.size());
    g.numNodes = n;
    if (n == 0)
        return g;

    auto feats = voiceFeatures(notes);
    const auto hetero = buildHeteroEdges(notes);
    g.embedEdgeIndex = hetero.index;
    g.embedEdgeCount = hetero.count;
    g.edgeType = hetero.type;

    auto pe = laplacianPosEnc(g.embedEdgeIndex, g.embedEdgeCount, n, PosEncDim);
    g.x.resize(static_cast<size_t>(n * InFeats));
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < VoiceFeatDim; ++d)
            g.x[static_cast<size_t>(i * InFeats + d)] =
                feats[static_cast<size_t>(i * VoiceFeatDim + d)];
        for (int d = 0; d < PosEncDim; ++d)
            g.x[static_cast<size_t>(i * InFeats + VoiceFeatDim + d)] =
                pe[static_cast<size_t>(i * PosEncDim + d)];
    }

    const auto pots = readdNearConsecutivePotEdges(
        buildPotEdges(n, hetero, PotEdgesMaxDist), notes, NearConsecutiveTolCs);
    g.targetEdgeCount = static_cast<int64_t>(pots.size());
    g.targetEdgeIndex.resize(static_cast<size_t>(g.targetEdgeCount * 2));
    for (int64_t e = 0; e < g.targetEdgeCount; ++e) {
        g.targetEdgeIndex[static_cast<size_t>(e)] = pots[static_cast<size_t>(e)].first;
        g.targetEdgeIndex[static_cast<size_t>(g.targetEdgeCount + e)] =
            pots[static_cast<size_t>(e)].second;
    }
    g.pitchScore = pitchScores(notes, pots);
    g.onsetScore = onsetScores(notes, pots);
    return g;
}

auto hungarianParents(const std::vector<float> &edgeScores,
                      const std::vector<int64_t> &targetEdgeIndex,
                      int64_t targetEdgeCount,
                      int numNodes,
                      float threshold) -> std::vector<int32_t> {
    std::vector<int32_t> parents(static_cast<size_t>(numNodes), -1);
    if (numNodes <= 0 || targetEdgeCount <= 0 || edgeScores.empty())
        return parents;

    std::vector<float> cost(static_cast<size_t>(numNodes * numNodes), 0.0f);
    for (int64_t e = 0; e < targetEdgeCount; ++e) {
        const int s = static_cast<int>(targetEdgeIndex[static_cast<size_t>(e)]);
        const int d = static_cast<int>(targetEdgeIndex[static_cast<size_t>(targetEdgeCount + e)]);
        if (s < 0 || d < 0 || s >= numNodes || d >= numNodes)
            continue;
        const float score = edgeScores[static_cast<size_t>(e)];
        cost[static_cast<size_t>(s * numNodes + d)] = score;
    }

    const auto assignment = hungarianMaximize(cost, numNodes);
    for (int row = 0; row < numNodes; ++row) {
        const int col = assignment[static_cast<size_t>(row)];
        if (col < 0)
            continue;
        const float score = cost[static_cast<size_t>(row * numNodes + col)];
        if (score > threshold)
            parents[static_cast<size_t>(col)] = row;
    }
    return parents;
}

} // namespace Vocsep
