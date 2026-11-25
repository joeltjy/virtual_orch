#include "VirtualOrch/reduction/ChordTheory.h"

#include <algorithm>
#include <cmath>

namespace ChordTheory {
namespace {

constexpr int kTypes = 18;
constexpr int kDom7 = 11;
constexpr int kMin7 = 12;

auto intervalsOf(int type) -> std::vector<int> {
    switch (type) {
        case 0: return {0};
        case 1: case 2: case 3: case 4: case 5: case 6: return {0, type};
        case 7: return {0, 4, 7};
        case 8: return {0, 3, 7};
        case 9: return {0, 3, 6};
        case 10: return {0, 4, 8};
        case 11: return {0, 4, 7, 10};
        case 12: return {0, 3, 7, 10};
        case 13: return {0, 4, 7, 11};
        case 14: return {0, 3, 7, 11};
        case 15: return {0, 3, 6, 10};
        case 16: return {0, 3, 6, 9};
        case 17: return {0, 4, 8, 11};
        default: return {0};
    }
}

auto has(Pcs s, int pc) -> bool { return ((s >> (((pc % 12) + 12) % 12)) & 1u) != 0; }

/** Every distinct chord (by pitch-class set), named as wait3 names it. */
auto allChords() -> const std::vector<std::pair<Pcs, TypeRoot>> & {
    static const auto chords = [] {
        std::vector<std::pair<Pcs, TypeRoot>> out;
        for (int t = 0; t < kTypes; ++t)
            for (int r = 0; r < 12; ++r) {
                const Pcs m = chordMask(t, r);
                if (std::none_of(out.begin(), out.end(), [m](const auto &c) { return c.first == m; }))
                    if (const auto tr = typeOf(m))
                        out.push_back({m, *tr});
            }
        return out;
    }();
    return chords;
}

auto scaleMask(Key key) -> Pcs {
    static constexpr int kMajor[] = {0, 2, 4, 5, 7, 9, 11};
    static constexpr int kMinor[] = {0, 2, 3, 5, 7, 8, 10, 11}; // natural minor + the harmonic raised 7th
    Pcs m = 0;
    if (key.minor)
        for (const int i: kMinor)
            m |= static_cast<Pcs>(1u << ((key.tonic + i) % 12));
    else
        for (const int i: kMajor)
            m |= static_cast<Pcs>(1u << ((key.tonic + i) % 12));
    return m;
}

/** The canonical entry of the key for this chord, if it is one. */
auto canonicalIn(const std::vector<Canonical> &canon, const TypeRoot &c) -> const Canonical * {
    for (const auto &e: canon)
        if (e.chord == c)
            return &e;
    return nullptr;
}

auto isCanonical(const std::vector<Canonical> &canon, Key key, Pcs mask, const TypeRoot &c) -> bool {
    if (c.type <= 6)
        return (mask & ~scaleMask(key)) == 0; // a unison / dyad: all its notes in the key's scale
    return canonicalIn(canon, c) != nullptr;
}

auto rootDegreeRank(Key key, int root) -> int {
    static constexpr int kMajor[] = {0, 2, 4, 5, 7, 9, 11};
    static constexpr int kMinor[] = {0, 2, 3, 5, 7, 8, 10};
    const int iv = ((root - key.tonic) % 12 + 12) % 12;
    for (int d = 0; d < 7; ++d)
        if ((key.minor ? kMinor : kMajor)[d] == iv)
            return degreeRank(d);
    if (key.minor && iv == 11)
        return degreeRank(6); // vii (harmonic)
    return 7;
}

template<typename KeyFn>
auto best(const std::vector<std::pair<Pcs, TypeRoot>> &cands, KeyFn keyOf) -> std::optional<TypeRoot> {
    std::optional<TypeRoot> out;
    std::vector<int> bestKey;
    for (const auto &[m, c]: cands) {
        auto k = keyOf(m, c);
        k.push_back(c.type);
        k.push_back(c.root);
        if (! out || k < bestKey) {
            out = c;
            bestKey = std::move(k);
        }
    }
    return out;
}

} // namespace

auto count(Pcs s) -> int {
    int n = 0;
    for (int i = 0; i < 12; ++i)
        n += has(s, i) ? 1 : 0;
    return n;
}

auto maskOf(const std::vector<int32_t> &pitches) -> Pcs {
    Pcs m = 0;
    for (const int32_t p: pitches)
        m |= static_cast<Pcs>(1u << (((p % 12) + 12) % 12));
    return m;
}

auto chordMask(int type, int root) -> Pcs {
    Pcs m = 0;
    for (const int iv: intervalsOf(type))
        m |= static_cast<Pcs>(1u << ((((root + iv) % 12) + 12) % 12));
    return m;
}

auto typeOf(Pcs s) -> std::optional<TypeRoot> {
    std::vector<int> pcs;
    for (int i = 0; i < 12; ++i)
        if (has(s, i))
            pcs.push_back(i);
    if (pcs.size() == 1)
        return TypeRoot{0, pcs[0]};
    if (pcs.size() == 2) {
        const int iv = pcs[1] - pcs[0];
        return iv <= 6 ? TypeRoot{iv, pcs[0]} : TypeRoot{12 - iv, pcs[1]};
    }
    if (pcs.size() < 3 || pcs.size() > 4)
        return std::nullopt;
    for (const int r: pcs) // ascending: for aug / dim7 the first match is the lowest pitch class
        for (int t = 7; t < kTypes; ++t)
            if (chordMask(t, r) == s)
                return TypeRoot{t, r};
    return std::nullopt;
}

auto recognized(Pcs s) -> bool { return s != 0 && (count(s) < 3 || typeOf(s).has_value()); }

auto estimateKey(const std::array<double, 12> &weights) -> Key {
    static constexpr double kMajor[12] = {6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88};
    static constexpr double kMinor[12] = {6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17};
    double wMean = 0;
    for (const double w: weights)
        wMean += w / 12.0;
    double wVar = 0;
    for (const double w: weights)
        wVar += (w - wMean) * (w - wMean);
    if (wVar <= 0)
        return Key{0, false};
    Key bestKey;
    double bestR = -2;
    for (int minor = 0; minor < 2; ++minor) {
        const double *prof = minor ? kMinor : kMajor;
        double pMean = 0;
        for (int i = 0; i < 12; ++i)
            pMean += prof[i] / 12.0;
        for (int tonic = 0; tonic < 12; ++tonic) {
            double num = 0;
            double pVar = 0;
            for (int pc = 0; pc < 12; ++pc) {
                const double pv = prof[((pc - tonic) % 12 + 12) % 12] - pMean;
                num += (weights[static_cast<size_t>(pc)] - wMean) * pv;
                pVar += pv * pv;
            }
            const double r = num / std::sqrt(wVar * pVar);
            if (r > bestR + 1e-12) {
                bestR = r;
                bestKey = Key{tonic, minor == 1};
            }
        }
    }
    return bestKey;
}

auto canonicalChords(Key key) -> std::vector<Canonical> {
    std::vector<Canonical> out;
    const auto add = [&](int degree, int iv, int type) { out.push_back({{type, (key.tonic + iv) % 12}, degree}); };
    if (! key.minor) {
        static constexpr int iv[] = {0, 2, 4, 5, 7, 9, 11};
        static constexpr int triad[] = {7, 8, 8, 7, 7, 8, 9};      // I ii iii IV V vi vii°
        static constexpr int seventh[] = {13, 12, 12, 13, 11, 12, 15}; // Imaj7 ii7 iii7 IVmaj7 V7 vi7 viiø7
        for (int d = 0; d < 7; ++d) {
            add(d, iv[d], triad[d]);
            add(d, iv[d], seventh[d]);
        }
    } else {
        static constexpr int iv[] = {0, 2, 3, 5, 7, 8, 10};
        static constexpr int triad[] = {8, 9, 7, 8, 8, 7, 7};       // i ii° III iv v VI VII
        static constexpr int seventh[] = {12, 15, 13, 12, 12, 13, 11}; // i7 iiø7 IIImaj7 iv7 v7 VImaj7 VII7
        for (int d = 0; d < 7; ++d) {
            add(d, iv[d], triad[d]);
            add(d, iv[d], seventh[d]);
        }
        add(4, 7, 7);   // V (harmonic)
        add(4, 7, 11);  // V7
        add(6, 11, 16); // vii°7
    }
    for (auto &c: out) // name it as wait3 does (dim7 root = the lowest pitch class)
        if (const auto tr = typeOf(chordMask(c.chord.type, c.chord.root)))
            c.chord = *tr;
    return out;
}

auto degreeRank(int degree) -> int {
    static constexpr int kRank[7] = {0, 3, 5, 2, 1, 4, 6}; // I V IV ii vi iii vii
    return degree >= 0 && degree < 7 ? kRank[degree] : 7;
}

auto infer(Pcs piano, int32_t lowestPitch, int k, const std::array<double, 12> &keyWeights) -> Inferred {
    Inferred out;
    out.key = estimateKey(keyWeights);
    const auto canon = canonicalChords(out.key);
    const int n = count(piano);
    const auto kMiss = [k](Pcs m) { return count(m) == k ? 0 : 1; };
    const auto degreeOf = [&](const TypeRoot &c) {
        if (const auto *e = canonicalIn(canon, c))
            return degreeRank(e->degree);
        return rootDegreeRank(out.key, c.root);
    };
    // steps 1 and 2: chords containing the piano notes (step 1: the unison / dyad itself, or its extension to k)
    {
        std::vector<std::pair<Pcs, TypeRoot>> cands;
        for (const auto &[m, c]: allChords()) {
            if ((m & piano) != piano)
                continue;
            if (n < 3 ? (m == piano || count(m) == k) : c.type >= 7)
                cands.push_back({m, c});
        }
        const auto pick = best(cands, [&](Pcs m, const TypeRoot &c) {
            return std::vector<int>{kMiss(m),
                                    count(m & ~piano), // exact fit first, then the fewest missing
                                    isCanonical(canon, out.key, m, c) ? 0 : 1,
                                    c.type == kMin7 ? 1 : 0, // dom7 before min7
                                    degreeOf(c)};
        });
        if (pick) {
            out.chord = *pick;
            out.step = n < 3 ? 1 : 2;
            return out;
        }
    }
    // step 3: a canonical chord of the key one note away (one extra, one missing, one swapped)
    {
        std::vector<std::pair<Pcs, TypeRoot>> cands;
        std::vector<int> ranks;
        for (const auto &e: canon) {
            const Pcs m = chordMask(e.chord.type, e.chord.root);
            const int extra = count(piano & ~m);
            const int missing = count(m & ~piano);
            if (extra <= 1 && missing <= 1 && extra + missing > 0)
                cands.push_back({m, e.chord});
        }
        const auto pick = best(cands, [&](Pcs m, const TypeRoot &c) {
            return std::vector<int>{kMiss(m), count(piano & ~m) + count(m & ~piano), degreeOf(c)};
        });
        if (pick) {
            out.chord = *pick;
            out.step = 3;
            return out;
        }
    }
    // step 4: a major / minor chord on the lowest piano note (k: 1 the note, 2 with its third, 4 the seventh chord)
    const int b = ((lowestPitch % 12) + 12) % 12;
    const bool minor = has(piano, b + 3);
    out.step = 4;
    switch (k) {
        case 1: out.chord = {0, b}; break;
        case 2: out.chord = {minor ? 3 : 4, b}; break;
        case 4: out.chord = {minor ? kMin7 : kDom7, b}; break;
        default: out.chord = {minor ? 8 : 7, b}; break;
    }
    return out;
}

} // namespace ChordTheory
