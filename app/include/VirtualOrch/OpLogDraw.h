#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "VirtualOrch/OpLog.h"

/**
 * One sampled choice for the operation log (RT_DRAW fields, OT_DRAW steps): the chosen token, its
 * probability under the distribution the sampler drew from (`p`: masks, biases, temperature, top-p),
 * the model's own probability (`p_model`: softmax of the raw logits at temperature 1), and the ten
 * most probable alternatives by `p`. Only computed while OpLog::on().
 */
namespace OpLogDraw {

struct Choice {
    int32_t token = 0;
    float p = 0.0f;
    float pModel = 0.0f;
};

struct Draw {
    std::string name;
    int32_t chosen = 0;
    float p = 0.0f;
    float pModel = 0.0f;
    std::vector<Choice> top;
    bool forced = false;

    [[nodiscard]] auto json() const -> std::string {
        std::string tops = "[";
        for (size_t i = 0; i < top.size(); ++i) {
            if (i)
                tops += ',';
            tops += "[" + std::to_string(top[i].token) + "," + OpLog::valueJson(static_cast<double>(top[i].p)) + ","
                    + OpLog::valueJson(static_cast<double>(top[i].pModel)) + "]";
        }
        tops += "]";
        OpLog::Obj o;
        o.add("field", name).add("chosen", static_cast<int64_t>(chosen)).add("p", static_cast<double>(p))
            .add("p_model", static_cast<double>(pModel)).add("top", OpLog::Raw{tops});
        if (forced)
            o.add("forced", true);
        return o.str();
    }
};

/** Softmax of `raw` (finite entries) at temperature 1; -inf stays 0. */
[[nodiscard]] inline auto softmax(const float *raw, size_t n) -> std::vector<float> {
    std::vector<float> out(n, 0.0f);
    float mx = -INFINITY;
    for (size_t i = 0; i < n; ++i)
        if (std::isfinite(raw[i]))
            mx = std::max(mx, raw[i]);
    if (! std::isfinite(mx))
        return out;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
        if (std::isfinite(raw[i])) {
            out[i] = std::exp(raw[i] - mx);
            sum += out[i];
        }
    if (sum > 0.0)
        for (auto &v: out)
            v = static_cast<float>(v / sum);
    return out;
}

/**
 * The distribution sampleTopP(scores, topP, temperature) draws from (MusicToken.h): nucleus at
 * temperature 1 over the finite scores, then softmax(score / temperature) on the nucleus;
 * temperature <= 0 puts everything on the best score.
 */
[[nodiscard]] inline auto samplingDistribution(const std::vector<float> &scores, float topP, float temperature)
    -> std::vector<float> {
    std::vector<float> out(scores.size(), 0.0f);
    std::vector<int32_t> finite;
    for (size_t i = 0; i < scores.size(); ++i)
        if (std::isfinite(scores[i]))
            finite.push_back(static_cast<int32_t>(i));
    if (finite.empty())
        return out;
    std::sort(finite.begin(), finite.end(), [&](int32_t a, int32_t b) {
        return scores[static_cast<size_t>(a)] > scores[static_cast<size_t>(b)];
    });
    const float rowmax = scores[static_cast<size_t>(finite.front())];
    size_t keep = finite.size();
    if (topP < 1.0f) {
        double total = 0.0;
        for (const auto i: finite)
            total += std::exp(scores[static_cast<size_t>(i)] - rowmax);
        double cum = 0.0;
        for (size_t k = 0; k < finite.size(); ++k) {
            cum += std::exp(scores[static_cast<size_t>(finite[k])] - rowmax) / total;
            if (cum >= topP) {
                keep = k + 1;
                break;
            }
        }
    }
    if (temperature <= 0.0f) {
        out[static_cast<size_t>(finite.front())] = 1.0f;
        return out;
    }
    double sum = 0.0;
    for (size_t k = 0; k < keep; ++k)
        sum += std::exp((scores[static_cast<size_t>(finite[k])] - rowmax) / temperature);
    for (size_t k = 0; k < keep; ++k) {
        const auto i = static_cast<size_t>(finite[k]);
        out[i] = static_cast<float>(std::exp((scores[i] - rowmax) / temperature) / sum);
    }
    return out;
}

/**
 * Describe a draw. `raw` = the model's logits over the whole row (size n), `scores` = what the
 * sampler sampled from, for tokens offset .. offset + scores.size() - 1 of the row.
 */
[[nodiscard]] inline auto describe(const std::string &name, const float *raw, size_t n,
                                   const std::vector<float> &scores, int32_t offset, float topP,
                                   float temperature, int32_t chosen, bool forced, size_t topK = 10) -> Draw {
    Draw d;
    d.name = name;
    d.chosen = chosen;
    d.forced = forced;
    const auto pm = softmax(raw, n);
    const auto ps = samplingDistribution(scores, topP, temperature);
    const auto pOf = [&](int32_t token) -> float {
        const int64_t i = static_cast<int64_t>(token) - offset;
        return i >= 0 && i < static_cast<int64_t>(ps.size()) ? ps[static_cast<size_t>(i)] : 0.0f;
    };
    const auto pmOf = [&](int32_t token) -> float {
        return token >= 0 && static_cast<size_t>(token) < pm.size() ? pm[static_cast<size_t>(token)] : 0.0f;
    };
    d.p = pOf(chosen);
    d.pModel = pmOf(chosen);
    std::vector<int32_t> idx;
    for (size_t i = 0; i < ps.size(); ++i)
        if (ps[i] > 0.0f)
            idx.push_back(static_cast<int32_t>(i));
    const size_t k = std::min(topK, idx.size());
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
                      [&](int32_t a, int32_t b) { return ps[static_cast<size_t>(a)] > ps[static_cast<size_t>(b)]; });
    for (size_t j = 0; j < k; ++j) {
        const int32_t token = idx[j] + offset;
        d.top.push_back({token, pOf(token), pmOf(token)});
    }
    return d;
}

} // namespace OpLogDraw
