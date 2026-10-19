#include "VirtualOrch/reduction/StreamingKvCache.h"

#include "VirtualOrch/OrtEnv.h"

#include <algorithm>
#include <cstring>

namespace StreamingKv {

namespace {

/** Copy row ranges [0,head) and [from, rows) of every (layer, head) block into a new array. */
auto sliceRows(const std::vector<float> &src, int blocks, int rows, int headDim, int head,
               int tailFrom) -> std::vector<float> {
    const int tail = rows - tailFrom;
    const int newRows = head + tail;
    std::vector<float> out(static_cast<size_t>(blocks) * static_cast<size_t>(newRows)
                           * static_cast<size_t>(headDim));
    for (int b = 0; b < blocks; ++b) {
        const float *from = src.data() + static_cast<size_t>(b) * static_cast<size_t>(rows)
                                             * static_cast<size_t>(headDim);
        float *to = out.data() + static_cast<size_t>(b) * static_cast<size_t>(newRows)
                                     * static_cast<size_t>(headDim);
        std::memcpy(to, from, static_cast<size_t>(head) * static_cast<size_t>(headDim) * sizeof(float));
        std::memcpy(to + static_cast<size_t>(head) * static_cast<size_t>(headDim),
                    from + static_cast<size_t>(tailFrom) * static_cast<size_t>(headDim),
                    static_cast<size_t>(tail) * static_cast<size_t>(headDim) * sizeof(float));
    }
    return out;
}

auto keepHeadAndTail(State &state, const Dims &dims, int head, int tailFrom) -> void {
    const int blocks = dims.layers * dims.heads;
    state.key = sliceRows(state.key, blocks, state.rows, dims.headDim, head, tailFrom);
    state.value = sliceRows(state.value, blocks, state.rows, dims.headDim, head, tailFrom);
    state.rows = head + (state.rows - tailFrom);
}

} // namespace

auto cutRows(State &state, const Dims &dims, int keepRows) -> void {
    keepRows = std::clamp(keepRows, 0, state.rows);
    if (keepRows == state.rows)
        return;
    keepHeadAndTail(state, dims, keepRows, state.rows);
}

auto evictForIncoming(State &state, const Dims &dims, int incoming) -> void {
    if (state.rows + incoming <= dims.maxRows)
        return;
    const int keepRecent = std::max(0, dims.maxRows - dims.sinkRows - incoming);
    const int head = std::min(dims.sinkRows, state.rows);
    const int tailFrom = std::max(head, state.rows - keepRecent);
    keepHeadAndTail(state, dims, head, tailFrom);
}

CommittedScratchCache::CommittedScratchCache(Backend &backendIn, int chunkTokens)
    : backend(backendIn), dims(backendIn.dims()), chunk(std::max(1, chunkTokens)) {}

auto CommittedScratchCache::reset() -> void {
    c = {};
    cFed.clear();
    cLogits.clear();
    ++cVersion;
    w = {};
    wFed.clear();
    wLogits.clear();
    wValid = false;
    failed = false;
}

auto CommittedScratchCache::feed(State &cache, std::vector<float> &logits,
                                 const std::vector<int32_t> &tokens, size_t from, size_t to)
    -> bool {
    for (size_t pos = from; pos < to;) {
        const size_t n = std::min(static_cast<size_t>(chunk), to - pos);
        evictForIncoming(cache, dims, static_cast<int>(n));
        std::vector<int64_t> ids(tokens.begin() + static_cast<std::ptrdiff_t>(pos),
                                 tokens.begin() + static_cast<std::ptrdiff_t>(pos + n));
        State present;
        std::vector<float> last;
        ++calls;
        if (! backend.run(cache, ids, present, last))
            return false;
        fedTotal += n;
        cache = std::move(present);
        logits = std::move(last);
        pos += n;
    }
    return true;
}

auto CommittedScratchCache::logitsAfter(const std::vector<int32_t> &desired, size_t committedLen)
    -> const std::vector<float> & {
    failed = false;
    if (desired.empty())
        return empty;
    committedLen = std::min(committedLen, desired.size());

    // 1. The committed prefix must still be what C was computed from.
    const bool prefixOk =
        cFed.size() <= committedLen
        && std::equal(cFed.begin(), cFed.end(), desired.begin());
    if (! prefixOk)
        reset();

    // 2. Advance C over the newly committed tokens.
    if (cFed.size() < committedLen) {
        if (! feed(c, cLogits, desired, cFed.size(), committedLen)) {
            reset();
            failed = true;
            return empty;
        }
        cFed.insert(cFed.end(), desired.begin() + static_cast<std::ptrdiff_t>(cFed.size()),
                    desired.begin() + static_cast<std::ptrdiff_t>(committedLen));
        ++cVersion;
        wValid = false;
    }

    // 3. The scratch tail goes on top of C, into W.
    const size_t scratchLen = desired.size() - committedLen;
    if (scratchLen == 0)
        return cLogits;

    const auto scratchBegin = desired.begin() + static_cast<std::ptrdiff_t>(committedLen);
    const bool reuse = wValid && wVersion == cVersion && wFed.size() <= scratchLen
                       && std::equal(wFed.begin(), wFed.end(), scratchBegin);
    if (! reuse) {
        w = c;
        wFed.clear();
        wLogits = cLogits;
        wValid = true;
        wVersion = cVersion;
    }
    if (wFed.size() < scratchLen) {
        const size_t from = committedLen + wFed.size();
        if (! feed(w, wLogits, desired, from, desired.size())) {
            failed = true;
            wValid = false;
            return empty;
        }
        wFed.assign(scratchBegin, desired.end());
    }
    return wLogits;
}

// ---- ONNX backend ----

auto OnnxBackend::init(const char *modelPath, std::string &error) -> bool {
    session.reset();
    const auto tryCreate = [&](Ort::SessionOptions &opts) {
        session = std::make_unique<Ort::Session>(sharedOrtEnv(), modelPath, opts);
    };
#ifdef __linux__
    try {
        Ort::SessionOptions cudaOptions;
        OrtCUDAProviderOptions cudaProvider{};
        cudaOptions.AppendExecutionProvider_CUDA(cudaProvider);
        tryCreate(cudaOptions);
    } catch (const Ort::Exception &) {
        session.reset();
    }
#endif
    if (session == nullptr) {
        try {
            Ort::SessionOptions cpuOptions;
            tryCreate(cpuOptions);
        } catch (const Ort::Exception &ex) {
            error = ex.what();
            session.reset();
            return false;
        }
    }

    inputNames.clear();
    outputNames.clear();
    const Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session->GetInputCount(); ++i)
        inputNames.emplace_back(session->GetInputNameAllocated(i, allocator).get());
    for (size_t i = 0; i < session->GetOutputCount(); ++i)
        outputNames.emplace_back(session->GetOutputNameAllocated(i, allocator).get());
    const auto has = [](const std::vector<std::string> &names, const char *name) {
        return std::find(names.begin(), names.end(), name) != names.end();
    };
    if (! has(inputNames, "input_ids") || ! has(inputNames, "past_key")
        || ! has(inputNames, "past_value") || ! has(outputNames, "logits")
        || ! has(outputNames, "present_key") || ! has(outputNames, "present_value")) {
        error = "not a KV export (needs input_ids, past_key, past_value -> logits, present_*)";
        session.reset();
        return false;
    }

    hasPositionIds = has(inputNames, "position_ids");

    const auto pastIndex = static_cast<size_t>(
        std::find(inputNames.begin(), inputNames.end(), "past_key") - inputNames.begin());
    const auto pastShape = session->GetInputTypeInfo(pastIndex).GetTensorTypeAndShapeInfo().GetShape();
    if (pastShape.size() == 5 && pastShape[0] > 0 && pastShape[2] > 0 && pastShape[4] > 0) {
        dimsValue.layers = static_cast<int>(pastShape[0]);
        dimsValue.heads = static_cast<int>(pastShape[2]);
        dimsValue.headDim = static_cast<int>(pastShape[4]);
    }
    const auto logitsIndex = static_cast<size_t>(
        std::find(outputNames.begin(), outputNames.end(), "logits") - outputNames.begin());
    const auto logitShape =
        session->GetOutputTypeInfo(logitsIndex).GetTensorTypeAndShapeInfo().GetShape();
    vocab = (! logitShape.empty() && logitShape.back() > 0) ? static_cast<size_t>(logitShape.back()) : 0;
    return true;
}

auto OnnxBackend::run(const State &past, const std::vector<int64_t> &ids, State &present,
                      std::vector<float> &lastLogits) -> bool {
    if (session == nullptr || ids.empty())
        return false;
    try {
        std::vector<int64_t> idsCopy = ids;
        const std::array<int64_t, 2> idsShape{1, static_cast<int64_t>(idsCopy.size())};
        std::vector<int64_t> positions(idsCopy.size());
        for (size_t i = 0; i < positions.size(); ++i)
            positions[i] = static_cast<int64_t>(past.rows) + static_cast<int64_t>(i);
        const std::array<int64_t, 5> pastShape{dimsValue.layers, 1, dimsValue.heads, past.rows,
                                               dimsValue.headDim};
        static float emptyData = 0.0f;
        auto *keyPtr = const_cast<float *>(past.key.empty() ? &emptyData : past.key.data());
        auto *valuePtr = const_cast<float *>(past.value.empty() ? &emptyData : past.value.data());

        std::vector<Ort::Value> inputs;
        std::vector<const char *> inputNamePtrs;
        for (const auto &name: inputNames) {
            inputNamePtrs.push_back(name.c_str());
            if (name == "input_ids")
                inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                    memoryInfo, idsCopy.data(), idsCopy.size(), idsShape.data(), idsShape.size()));
            else if (name == "position_ids")
                inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                    memoryInfo, positions.data(), positions.size(), idsShape.data(), idsShape.size()));
            else if (name == "past_key")
                inputs.push_back(Ort::Value::CreateTensor<float>(
                    memoryInfo, keyPtr, past.key.size(), pastShape.data(), pastShape.size()));
            else if (name == "past_value")
                inputs.push_back(Ort::Value::CreateTensor<float>(
                    memoryInfo, valuePtr, past.value.size(), pastShape.data(), pastShape.size()));
            else {
                lastError = "unexpected input " + name;
                return false;
            }
        }
        std::vector<const char *> outputNamePtrs;
        for (const auto &name: outputNames)
            outputNamePtrs.push_back(name.c_str());

        auto outputs = session->Run(Ort::RunOptions{nullptr}, inputNamePtrs.data(), inputs.data(),
                                    inputs.size(), outputNamePtrs.data(), outputNamePtrs.size());
        for (size_t i = 0; i < outputNames.size(); ++i) {
            const auto &name = outputNames[i];
            const auto info = outputs[i].GetTensorTypeAndShapeInfo();
            const auto shape = info.GetShape();
            const float *data = outputs[i].GetTensorData<float>();
            if (name == "logits") {
                if (shape.size() != 3 || shape[2] <= 0 || shape[1] <= 0)
                    return false;
                const auto v = static_cast<size_t>(shape[2]);
                const float *row = data + static_cast<size_t>(shape[1] - 1) * v;
                lastLogits.assign(row, row + v);
            } else if (name == "present_key") {
                if (shape.size() != 5)
                    return false;
                present.rows = static_cast<int>(shape[3]);
                present.key.assign(data, data + info.GetElementCount());
            } else if (name == "present_value") {
                present.value.assign(data, data + info.GetElementCount());
            }
        }
        return true;
    } catch (const Ort::Exception &ex) {
        lastError = ex.what();
        return false;
    }
}

} // namespace StreamingKv
