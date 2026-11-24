#include "VirtualOrch/reduction/VoicingBank.h"

#include <JuceHeader.h>

#include <algorithm>
#include <climits>
#include <cstdlib>

auto VoicingBank::typeIndex(const std::string &name) -> int {
    for (int i = 0; i < kTypes; ++i)
        if (name == kTypeNames[static_cast<size_t>(i)])
            return i;
    return -1;
}

auto VoicingBank::intervals(int type) -> std::vector<int> {
    switch (type) {
        case 0: return {0};
        case 1: case 2: case 3: case 4: case 5: case 6: return {0, type}; // icN
        case 7: return {0, 4, 7};      // maj
        case 8: return {0, 3, 7};      // min
        case 9: return {0, 3, 6};      // dim
        case 10: return {0, 4, 8};     // aug
        case 11: return {0, 4, 7, 10}; // dom7
        case 12: return {0, 3, 7, 10}; // min7
        case 13: return {0, 4, 7, 11}; // maj7
        case 14: return {0, 3, 7, 11}; // minmaj7
        case 15: return {0, 3, 6, 10}; // hdim7
        case 16: return {0, 3, 6, 9};  // dim7
        case 17: return {0, 4, 8, 11}; // augmaj7
        default: return {0};
    }
}

auto VoicingBank::pitchClassCount(int type) -> int { return static_cast<int>(intervals(type).size()); }

auto VoicingBank::key(int type, int n, int bassIv) -> std::string {
    const char *name = type >= 0 && type < kTypes ? kTypeNames[static_cast<size_t>(type)] : "?";
    return std::string(name) + "|" + std::to_string(n) + "|" + std::to_string(bassIv);
}

auto VoicingBank::add(int type, int n, int bassIv, std::vector<std::vector<int>> voicings) -> void {
    bank[key(type, n, bassIv)] = std::move(voicings);
}

auto VoicingBank::loadJson(const std::string &text, std::string &error) -> bool {
    const auto parsed = juce::JSON::parse(juce::String(text));
    const auto *root = parsed.getDynamicObject();
    if (root == nullptr) {
        error = "voicing bank: not a JSON object";
        return false;
    }
    const auto *entries = root->getProperty("bank").getDynamicObject();
    if (entries == nullptr) {
        error = "voicing bank: no \"bank\" object";
        return false;
    }
    std::map<std::string, std::vector<std::vector<int>>> next;
    for (const auto &kv: entries->getProperties()) {
        const auto *list = kv.value.getArray();
        if (list == nullptr)
            continue;
        std::vector<std::vector<int>> voicings;
        for (const auto &v: *list) {
            const auto *offsets = v.getArray();
            if (offsets == nullptr || offsets->isEmpty())
                continue;
            std::vector<int> o;
            for (const auto &x: *offsets)
                o.push_back(static_cast<int>(x));
            voicings.push_back(std::move(o));
        }
        if (! voicings.empty())
            next[kv.name.toString().toStdString()] = std::move(voicings);
    }
    if (next.empty()) {
        error = "voicing bank: \"bank\" has no voicings";
        return false;
    }
    bank = std::move(next);
    return true;
}

auto VoicingBank::decode(int type, int rootField, int n, int32_t bassPitch, std::mt19937_64 &rng) const -> Decoded {
    Decoded out;
    n = std::clamp(n, 1, 10);
    type = std::clamp(type, 0, kTypes - 1);
    const int bassIv = ((bassPitch - rootField) % 12 + 12) % 12;
    const std::vector<std::vector<int>> *found = nullptr;
    if (const auto it = bank.find(key(type, n, bassIv)); it != bank.end()) {
        found = &it->second;
    } else {
        // the same type and bass position with the nearest n (ties: the smaller n)
        int bestDist = INT_MAX;
        for (int m = 1; m <= 10; ++m)
            if (const auto jt = bank.find(key(type, m, bassIv)); jt != bank.end() && std::abs(m - n) < bestDist) {
                bestDist = std::abs(m - n);
                found = &jt->second;
            }
        out.source = Decoded::Source::NearestN;
    }
    if (found != nullptr && ! found->empty()) {
        std::uniform_int_distribution<size_t> pickOne(0, found->size() - 1);
        const auto &voicing = (*found)[pickOne(rng)];
        for (const int o: voicing)
            out.pitches.push_back(bassPitch + o);
    } else {
        // the chord tones stacked upward from the bass, n notes
        out.source = Decoded::Source::Stacked;
        std::vector<int> pcs;
        for (const int iv: intervals(type))
            pcs.push_back((trueRootPc(rootField) + iv) % 12);
        out.pitches.push_back(bassPitch);
        int32_t p = bassPitch;
        while (static_cast<int>(out.pitches.size()) < n) {
            ++p;
            while (std::find(pcs.begin(), pcs.end(), ((p % 12) + 12) % 12) == pcs.end())
                ++p;
            out.pitches.push_back(p);
        }
    }
    std::sort(out.pitches.begin(), out.pitches.end());
    out.pitches.erase(std::unique(out.pitches.begin(), out.pitches.end()), out.pitches.end());
    out.pitches.erase(std::remove_if(out.pitches.begin(), out.pitches.end(),
                                     [&](int32_t p) { return p != bassPitch && (p < 21 || p > 108); }),
                      out.pitches.end());
    return out;
}
