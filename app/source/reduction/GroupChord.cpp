#include "VirtualOrch/reduction/GroupChord.h"

#include "VirtualOrch/MusicToken.h"

#include <JuceHeader.h>

#include <algorithm>
#include <random>

namespace GroupChord {
namespace {

const char *kNames[] = {"C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"};

struct Table {
    std::vector<Shape> shapes;                         // by offset
    std::map<std::pair<std::vector<int32_t>, int32_t>, int32_t> byKey; // (intervals, n) -> offset
    std::array<std::vector<int32_t>, 5> byPcs;         // pitch classes -> offsets
};

auto inversions(const std::vector<std::pair<std::string, std::vector<int32_t>>> &templates)
    -> std::vector<std::pair<std::string, std::vector<int32_t>>> {
    static const char *inv[] = {"root", "1st", "2nd", "3rd"};
    std::vector<std::pair<std::string, std::vector<int32_t>>> out;
    for (const auto &[name, t]: templates)
        for (size_t i = 0; i < t.size(); ++i) {
            std::vector<int32_t> sh;
            for (const int32_t x: t)
                sh.push_back(((x - t[i]) % 12 + 12) % 12);
            std::sort(sh.begin(), sh.end());
            if (std::any_of(out.begin(), out.end(), [&](const auto &o) { return o.second == sh; }))
                continue; // aug / dim7: one shape each
            out.push_back({name == "aug" || name == "dim7" ? name : name + " " + inv[i], sh});
        }
    return out;
}

// The order of overnight/pkg/groups4.py (SHAPE_ROWS): monad, dyads +1..+11, triad inversions, seventh inversions.
auto build() -> Table {
    Table t;
    struct Row {
        Kind kind;
        std::string name;
        std::vector<int32_t> iv;
    };
    std::vector<Row> rows{{Kind::Monad, "unison/octave", {0}}};
    for (int32_t i = 1; i < 12; ++i)
        rows.push_back({Kind::Dyad, "+" + std::to_string(i), {0, i}});
    for (const auto &[n, sh]: inversions({{"maj", {0, 4, 7}}, {"min", {0, 3, 7}}, {"dim", {0, 3, 6}}, {"aug", {0, 4, 8}}}))
        rows.push_back({Kind::Triad, n, sh});
    for (const auto &[n, sh]: inversions({{"maj7", {0, 4, 7, 11}}, {"7", {0, 4, 7, 10}}, {"m7", {0, 3, 7, 10}},
                                          {"m7b5", {0, 3, 6, 10}}, {"mMaj7", {0, 3, 7, 11}}, {"dim7", {0, 3, 6, 9}}}))
        rows.push_back({Kind::Seventh, n, sh});
    for (const auto &r: rows) {
        const auto [lo, hi] = r.kind == Kind::Monad ? std::pair{2, 5}
                              : r.kind == Kind::Dyad ? std::pair{2, 6}
                              : r.kind == Kind::Triad ? std::pair{3, 10} : std::pair{4, 12};
        for (int32_t n = lo; n <= hi; ++n) {
            const auto off = static_cast<int32_t>(t.shapes.size());
            t.shapes.push_back({r.iv, r.kind, n, r.name});
            t.byKey[{r.iv, n}] = off;
            t.byPcs[r.iv.size()].push_back(off);
        }
    }
    return t;
}

auto table() -> const Table & {
    static const Table t = build();
    return t;
}

std::mutex bankLock;
std::shared_ptr<const VoicingBankV4> shared;

} // namespace

auto shape(int32_t offset) -> const Shape & {
    const auto &s = table().shapes;
    return s[static_cast<size_t>(std::clamp<int32_t>(offset, 0, static_cast<int32_t>(s.size()) - 1))];
}

auto offsetOf(const std::vector<int32_t> &intervals, int32_t n) -> int32_t {
    const auto &k = table().byKey;
    const auto it = k.find({intervals, n});
    return it == k.end() ? -1 : it->second;
}

auto offsetsWithPitchClasses(int32_t k) -> const std::vector<int32_t> & {
    return table().byPcs[static_cast<size_t>(std::clamp<int32_t>(k, 0, 4))];
}

auto pitchClassMask(int32_t bassPitch, int32_t offset) -> uint16_t {
    uint16_t m = 0;
    for (const int32_t x: shape(offset).intervals)
        m |= static_cast<uint16_t>(1u << (((bassPitch + x) % 12 + 12) % 12));
    return m;
}

auto label(int32_t bassPitch, int32_t offset) -> std::string {
    const auto &s = shape(offset);
    return std::string(kNames[((bassPitch % 12) + 12) % 12]) + std::to_string(bassPitch / 12 - 1) + " " + s.name + " x"
           + std::to_string(s.n);
}

auto VoicingBankV4::loadJson(const std::string &text, std::string &error) -> bool {
    const auto parsed = juce::JSON::parse(juce::String(text));
    const auto *root = parsed.getDynamicObject();
    const auto *entries = root != nullptr ? root->getProperty("bank").getDynamicObject() : nullptr;
    if (entries == nullptr) {
        error = "voicing bank v4: no \"bank\" object";
        return false;
    }
    decltype(bank) next;
    for (const auto &kv: entries->getProperties()) {
        const int32_t off = kv.name.toString().getIntValue();
        const auto *list = kv.value.getArray();
        if (list == nullptr || off < 0 || off >= kShapes)
            continue;
        for (const auto &e: *list) {
            const auto *pair = e.getArray();
            if (pair == nullptr || pair->size() < 2 || pair->getReference(0).getArray() == nullptr)
                continue;
            std::vector<int32_t> iv;
            for (const auto &x: *pair->getReference(0).getArray())
                iv.push_back(static_cast<int32_t>(x));
            if (! iv.empty())
                next[off].push_back({std::move(iv), static_cast<int64_t>(pair->getReference(1))});
        }
    }
    if (next.empty()) {
        error = "voicing bank v4: no voicings";
        return false;
    }
    bank = std::move(next);
    return true;
}

auto VoicingBankV4::loadFile(const std::string &path, std::string &error) -> bool {
    const juce::File f(path);
    if (! f.existsAsFile()) {
        error = "voicing bank v4 not found: " + path;
        return false;
    }
    return loadJson(f.loadFileAsString().toStdString(), error);
}

auto VoicingBankV4::decode(int32_t bassPitch, int32_t offset, uint64_t seed, int32_t top) const -> std::vector<int32_t> {
    const auto &s = shape(offset);
    std::vector<int32_t> ps;
    const auto it = bank.find(offset);
    if (it != bank.end() && ! it->second.empty()) {
        int64_t total = 0;
        for (const auto &[iv, c]: it->second)
            total += std::max<int64_t>(1, c);
        std::mt19937_64 rng(seed * 0x9E3779B97F4A7C15ull + static_cast<uint64_t>(offset));
        int64_t u = static_cast<int64_t>(rng() % static_cast<uint64_t>(std::max<int64_t>(1, total)));
        const auto *pick = &it->second.front().first;
        for (const auto &[iv, c]: it->second) {
            u -= std::max<int64_t>(1, c);
            if (u < 0) {
                pick = &iv;
                break;
            }
        }
        for (const int32_t x: *pick)
            ps.push_back(bassPitch + x);
    } else { // close position upward from the bass, cycling the shape's pitch classes
        ps.push_back(bassPitch);
        int32_t octave = 0;
        size_t i = 1;
        while (static_cast<int32_t>(ps.size()) < s.n && octave < 120) {
            if (i >= s.intervals.size()) {
                i = 0;
                octave += 12;
            }
            const int32_t p = bassPitch + s.intervals[i] + octave;
            if (p > ps.back())
                ps.push_back(p);
            ++i;
        }
    }
    std::vector<int32_t> out;
    for (const int32_t p: ps)
        if (p >= 0 && (p <= top || p == bassPitch) && std::find(out.begin(), out.end(), p) == out.end())
            out.push_back(p);
    std::sort(out.begin(), out.end());
    return out;
}

auto setSharedBank(std::shared_ptr<const VoicingBankV4> bank) -> void {
    const std::lock_guard<std::mutex> g(bankLock);
    shared = std::move(bank);
}

auto sharedBank() -> std::shared_ptr<const VoicingBankV4> {
    const std::lock_guard<std::mutex> g(bankLock);
    return shared;
}

auto expand(const Token &token) -> std::vector<Token> {
    if (token.chordShape < 0)
        return {token};
    const int32_t bass = token.getPitch();
    static const VoicingBankV4 none;
    const auto bank = sharedBank();
    const auto pitches = (bank ? *bank : none).decode(bass, token.chordShape,
                                                      static_cast<uint64_t>(token.noteId < 0 ? 0 : token.noteId));
    std::vector<Token> out;
    for (const int32_t p: pitches) {
        Token t = token;
        t.note = token.note - bass + p;
        t.chordShape = -1;
        out.push_back(t);
    }
    if (out.empty()) {
        Token t = token;
        t.chordShape = -1;
        out.push_back(t);
    }
    return out;
}

auto expandAll(const std::vector<Token> &tokens) -> std::vector<Token> {
    std::vector<Token> out;
    out.reserve(tokens.size());
    for (const auto &t: tokens) {
        if (t.chordShape < 0) {
            out.push_back(t);
            continue;
        }
        const auto e = expand(t);
        out.insert(out.end(), e.begin(), e.end());
    }
    return out;
}

} // namespace GroupChord
