#include "VirtualOrch/reduction/StreamingKvCache.h"

#include "VirtualOrch/OrtEnv.h"

#include <algorithm>
#include <array>
#include <cstring>

#ifdef __linux__
#include <dlfcn.h>
#endif

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
    if (state.device != nullptr) {
        // null on failure: the next run() sees rows without data and fails
        state.device = state.device->keepRows(head, tailFrom);
        state.rows = head + (state.rows - tailFrom);
        return;
    }
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

/** What the device states need: the session (their memory comes from its CUDA arena) and cudart. */
struct OnnxBackend::CudaContext {
    using Memcpy2D = int (*)(void *, size_t, const void *, size_t, size_t, size_t, int);
    static constexpr int kDeviceToDevice = 3; // cudaMemcpyDeviceToDevice

    CudaContext(std::shared_ptr<Ort::Session> sessionIn, Memcpy2D memcpy2DIn, const Dims &dimsIn)
        : session(std::move(sessionIn)), allocator(*session, memory), memcpy2D(memcpy2DIn),
          dims(dimsIn) {}

    std::shared_ptr<Ort::Session> session;
    Ort::MemoryInfo memory{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};
    Ort::Allocator allocator;
    Memcpy2D memcpy2D;
    Dims dims;
};

namespace {

/** present_key / present_value of one run, left in GPU memory. */
class CudaKv final : public DeviceKv {
public:
    CudaKv(std::shared_ptr<OnnxBackend::CudaContext> contextIn, Ort::Value keyIn,
           Ort::Value valueIn, int rowsIn)
        : context(std::move(contextIn)), key(std::move(keyIn)), value(std::move(valueIn)),
          rows(rowsIn) {}

    [[nodiscard]] auto keepRows(int head, int tailFrom) const
        -> std::shared_ptr<const DeviceKv> override {
        const auto &d = context->dims;
        const int tail = rows - tailFrom;
        const int newRows = head + tail;
        const std::array<int64_t, 5> shape{d.layers, 1, d.heads, newRows, d.headDim};
        const auto blocks = static_cast<size_t>(d.layers) * static_cast<size_t>(d.heads);
        const size_t rowBytes = static_cast<size_t>(d.headDim) * sizeof(float);
        try {
            auto newKey = Ort::Value::CreateTensor<float>(context->allocator, shape.data(), shape.size());
            auto newValue = Ort::Value::CreateTensor<float>(context->allocator, shape.data(), shape.size());
            // one strided copy per range: `blocks` rows of `rows` (source) / `newRows` (target)
            const auto copy = [&](const Ort::Value &src, Ort::Value &dst) {
                const auto *from = static_cast<const char *>(src.GetTensorRawData());
                auto *to = static_cast<char *>(dst.GetTensorMutableRawData());
                const size_t srcPitch = static_cast<size_t>(rows) * rowBytes;
                const size_t dstPitch = static_cast<size_t>(newRows) * rowBytes;
                bool ok = true;
                if (head > 0)
                    ok = ok && context->memcpy2D(to, dstPitch, from, srcPitch,
                                                 static_cast<size_t>(head) * rowBytes, blocks,
                                                 OnnxBackend::CudaContext::kDeviceToDevice) == 0;
                if (tail > 0)
                    ok = ok && context->memcpy2D(to + static_cast<size_t>(head) * rowBytes, dstPitch,
                                                 from + static_cast<size_t>(tailFrom) * rowBytes, srcPitch,
                                                 static_cast<size_t>(tail) * rowBytes, blocks,
                                                 OnnxBackend::CudaContext::kDeviceToDevice) == 0;
                return ok;
            };
            if (! copy(key, newKey) || ! copy(value, newValue))
                return nullptr;
            return std::make_shared<CudaKv>(context, std::move(newKey), std::move(newValue), newRows);
        } catch (const Ort::Exception &) {
            return nullptr;
        }
    }

    // context first: the tensors are freed before the session they came from
    std::shared_ptr<OnnxBackend::CudaContext> context;
    Ort::Value key;
    Ort::Value value;
    int rows;
};

/** cudaMemcpy2D from the cudart the CUDA provider already loaded (no link-time CUDA dependency). */
auto findMemcpy2D() -> OnnxBackend::CudaContext::Memcpy2D {
#ifdef __linux__
    for (const char *name: {"libcudart.so.12", "libcudart.so"}) {
        void *lib = dlopen(name, RTLD_NOW | RTLD_NOLOAD);
        if (lib == nullptr)
            lib = dlopen(name, RTLD_NOW);
        if (lib == nullptr)
            continue;
        if (void *fn = dlsym(lib, "cudaMemcpy2D"); fn != nullptr)
            return reinterpret_cast<OnnxBackend::CudaContext::Memcpy2D>(fn);
    }
#endif
    return nullptr;
}

/** Rows with no data behind them (a failed device slice, or a foreign state). */
auto missingData(const State &past, const Dims &d) -> bool {
    const auto expected = static_cast<size_t>(d.layers) * static_cast<size_t>(d.heads)
                          * static_cast<size_t>(past.rows) * static_cast<size_t>(d.headDim);
    return past.device == nullptr && (past.key.size() != expected || past.value.size() != expected);
}

} // namespace

auto OnnxBackend::init(const char *modelPath, std::string &error) -> bool {
    session.reset();
    cuda.reset();
    bool onCuda = false;
    const auto tryCreate = [&](Ort::SessionOptions &opts) {
        session = std::make_shared<Ort::Session>(sharedOrtEnv(), modelPath, opts);
    };
#ifdef __linux__
    try {
        Ort::SessionOptions cudaOptions;
        OrtCUDAProviderOptions cudaProvider{};
        cudaOptions.AppendExecutionProvider_CUDA(cudaProvider);
        tryCreate(cudaOptions);
        onCuda = true;
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

    if (onCuda) {
        if (const auto memcpy2D = findMemcpy2D(); memcpy2D != nullptr) {
            try {
                cuda = std::make_shared<CudaContext>(session, memcpy2D, dimsValue);
            } catch (const Ort::Exception &) {
                cuda.reset(); // host KV path on the CUDA session
            }
        }
    }
    return true;
}

auto OnnxBackend::run(const State &past, const std::vector<int64_t> &ids, State &present,
                      std::vector<float> &lastLogits) -> bool {
    if (session == nullptr || ids.empty())
        return false;
    if (missingData(past, dimsValue)) {
        lastError = "past KV has rows but no data";
        return false;
    }
    if (past.device != nullptr && dynamic_cast<const CudaKv *>(past.device.get()) == nullptr) {
        lastError = "past KV belongs to another backend";
        return false;
    }
    return cuda != nullptr ? runOnDevice(past, ids, present, lastLogits)
                           : runOnHost(past, ids, present, lastLogits);
}

/**
 * The KV state stays on the GPU: past_* are bound from the previous run's device tensors and
 * present_* are left where ORT computes them; only the ids go in and the logits come out.
 */
auto OnnxBackend::runOnDevice(const State &past, const std::vector<int64_t> &ids, State &present,
                              std::vector<float> &lastLogits) -> bool {
    try {
        std::vector<int64_t> idsCopy = ids;
        const std::array<int64_t, 2> idsShape{1, static_cast<int64_t>(idsCopy.size())};
        std::vector<int64_t> positions(idsCopy.size());
        for (size_t i = 0; i < positions.size(); ++i)
            positions[i] = static_cast<int64_t>(past.rows) + static_cast<int64_t>(i);
        const std::array<int64_t, 5> pastShape{dimsValue.layers, 1, dimsValue.heads, past.rows,
                                               dimsValue.headDim};
        static float emptyData = 0.0f;
        const auto *pastDevice = static_cast<const CudaKv *>(past.device.get());

        // host inputs (ids, positions, and a host or empty past) are copied in by ORT
        std::vector<Ort::Value> hostInputs;
        Ort::IoBinding binding(*session);
        for (const auto &name: inputNames) {
            if (name == "input_ids" || name == "position_ids") {
                auto &data = name == "input_ids" ? idsCopy : positions;
                hostInputs.push_back(Ort::Value::CreateTensor<int64_t>(
                    memoryInfo, data.data(), data.size(), idsShape.data(), idsShape.size()));
                binding.BindInput(name.c_str(), hostInputs.back());
            } else if (name == "past_key" || name == "past_value") {
                const bool isKey = name == "past_key";
                if (pastDevice != nullptr) {
                    binding.BindInput(name.c_str(), isKey ? pastDevice->key : pastDevice->value);
                } else {
                    const auto &host = isKey ? past.key : past.value;
                    auto *ptr = const_cast<float *>(host.empty() ? &emptyData : host.data());
                    hostInputs.push_back(Ort::Value::CreateTensor<float>(
                        memoryInfo, ptr, host.size(), pastShape.data(), pastShape.size()));
                    binding.BindInput(name.c_str(), hostInputs.back());
                }
            } else {
                lastError = "unexpected input " + name;
                return false;
            }
        }
        const std::array<const char *, 3> bound{"logits", "present_key", "present_value"};
        binding.BindOutput(bound[0], memoryInfo);
        binding.BindOutput(bound[1], cuda->memory);
        binding.BindOutput(bound[2], cuda->memory);

        binding.SynchronizeInputs();
        session->Run(Ort::RunOptions{nullptr}, binding);
        binding.SynchronizeOutputs();
        auto outputs = binding.GetOutputValues();
        if (outputs.size() != bound.size())
            return false;

        const auto logitShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        if (logitShape.size() != 3 || logitShape[2] <= 0 || logitShape[1] <= 0)
            return false;
        const auto v = static_cast<size_t>(logitShape[2]);
        const float *row = outputs[0].GetTensorData<float>() + static_cast<size_t>(logitShape[1] - 1) * v;
        lastLogits.assign(row, row + v);

        const auto kvShape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        if (kvShape.size() != 5)
            return false;
        present.key.clear();
        present.value.clear();
        present.rows = static_cast<int>(kvShape[3]);
        present.device = std::make_shared<CudaKv>(cuda, std::move(outputs[1]), std::move(outputs[2]),
                                                  present.rows);
        return true;
    } catch (const Ort::Exception &ex) {
        lastError = ex.what();
        return false;
    }
}

auto OnnxBackend::runOnHost(const State &past, const std::vector<int64_t> &ids, State &present,
                            std::vector<float> &lastLogits) -> bool {
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
                present.device.reset();
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
