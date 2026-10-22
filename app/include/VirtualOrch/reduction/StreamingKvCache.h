#pragma once

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/**
 * StreamingLLM KV cache for the RoPE dense models (da_rope_td_T7, maestro_rope_td_T3 exports):
 * input_ids [1, n] + past_key / past_value [layers, 1, heads, rows, dim] ->
 * logits [1, n, vocab] + present_key / present_value. The cache keeps 4 sink rows plus the
 * most recent rows, at most 512 in all; RoPE positions are the slots in the cache, so the app
 * evicts BEFORE feeding (the graph would otherwise rotate new tokens at positions > 512,
 * which the model never saw).
 *
 * CommittedScratchCache adds the "committed + scratch" paradigm: only tokens that can no
 * longer change are written into the committed cache C. Tokens after them (the scratch tail,
 * e.g. a held piano note's control and everything generated after it) are fed on top of C into
 * a working cache W that is thrown away when the tail changes.
 */
namespace StreamingKv {

struct Dims {
    int layers = 24;
    int heads = 16;
    int headDim = 64;
    int maxRows = 512;
    int sinkRows = 4;
};

/**
 * KV rows a backend keeps in its own memory (the CUDA OnnxBackend: GPU tensors), so they are not
 * copied to the host and back on every call. Immutable: slicing makes a new one, so copies of a
 * State (the scratch cache W = C) can share it.
 */
class DeviceKv {
public:
    virtual ~DeviceKv() = default;
    /** Rows [0, head) and [tailFrom, rows) of every (layer, head) block; nullptr on failure. */
    [[nodiscard]] virtual auto keepRows(int head, int tailFrom) const
        -> std::shared_ptr<const DeviceKv> = 0;
};

/**
 * keys / values laid out [layer][head][row][dim] (batch 1), either in `key` / `value` or, when
 * `device` is set, in the backend's memory (`key` / `value` are then empty).
 */
struct State {
    std::vector<float> key;
    std::vector<float> value;
    int rows = 0;
    std::shared_ptr<const DeviceKv> device;
};

/** Keep only the first `keepRows` rows. */
auto cutRows(State &state, const Dims &dims, int keepRows) -> void;

/** Make room for `incoming` rows: keep the sink rows and the newest rows. */
auto evictForIncoming(State &state, const Dims &dims, int incoming) -> void;

class Backend {
public:
    virtual ~Backend() = default;
    [[nodiscard]] virtual auto dims() const -> Dims = 0;
    /** Run `ids` on top of `past`; fills `present` and the logits of the LAST id. */
    [[nodiscard]] virtual auto run(const State &past,
                                   const std::vector<int64_t> &ids,
                                   State &present,
                                   std::vector<float> &lastLogits) -> bool = 0;
};

class CommittedScratchCache {
public:
    explicit CommittedScratchCache(Backend &backend, int chunkTokens = 64);

    /**
     * Logits for the token after `desired`. desired[0, committedLen) is final; the rest is the
     * scratch tail. Empty result: nothing to feed, or the backend failed.
     */
    [[nodiscard]] auto logitsAfter(const std::vector<int32_t> &desired, size_t committedLen)
        -> const std::vector<float> &;

    auto reset() -> void;

    [[nodiscard]] auto committedFed() const -> size_t { return cFed.size(); }
    [[nodiscard]] auto scratchFed() const -> size_t { return wValid ? wFed.size() : 0; }
    [[nodiscard]] auto committedRows() const -> int { return c.rows; }
    /** Tokens the committed cache has evicted so far (rows fed minus rows kept). */
    [[nodiscard]] auto evictedTokens() const -> size_t {
        return cFed.size() - static_cast<size_t>(c.rows);
    }
    /** Sink rows that are never evicted (the first tokens of the stream). */
    [[nodiscard]] auto sinkRows() const -> size_t { return static_cast<size_t>(dims.sinkRows); }
    [[nodiscard]] auto backendCalls() const -> int { return calls; }
    [[nodiscard]] auto tokensFed() const -> size_t { return fedTotal; }
    [[nodiscard]] auto lastFailed() const -> bool { return failed; }

private:
    auto feed(State &cache, std::vector<float> &logits, const std::vector<int32_t> &tokens,
              size_t from, size_t to) -> bool;

    Backend &backend;
    Dims dims;
    int chunk;

    State c;
    std::vector<int32_t> cFed;
    std::vector<float> cLogits;
    uint64_t cVersion = 0;

    State w;
    std::vector<int32_t> wFed; // scratch tokens W was computed from (on top of cFed at wVersion)
    std::vector<float> wLogits;
    bool wValid = false;
    uint64_t wVersion = 0;

    std::vector<float> empty;
    int calls = 0;
    size_t fedTotal = 0;
    bool failed = false;
};

/** ONNX Runtime backend for the KV exports. */
class OnnxBackend : public Backend {
public:
    /** Returns false (and fills `error`) when the model cannot be loaded. */
    auto init(const char *modelPath, std::string &error) -> bool;

    [[nodiscard]] auto isLoaded() const -> bool { return session != nullptr; }
    /** True when the cache stays in GPU memory between calls (CUDA session + libcudart). */
    [[nodiscard]] auto keepsKvOnDevice() const -> bool { return cuda != nullptr; }
    /** True for the learned-position export (T8): it takes explicit position_ids. */
    [[nodiscard]] auto usesPositionIds() const -> bool { return hasPositionIds; }
    /**
     * Learned positions cannot be re-indexed: rows are never evicted from the cache (the caller
     * rebuilds it instead), so the cache may hold up to `maxRows` before the graph is fed.
     */
    auto setRowLimit(int maxRows) -> void { dimsValue.maxRows = maxRows; }
    [[nodiscard]] auto vocabSize() const -> size_t { return vocab; }
    [[nodiscard]] auto dims() const -> Dims override { return dimsValue; }
    [[nodiscard]] auto run(const State &past, const std::vector<int64_t> &ids, State &present,
                           std::vector<float> &lastLogits) -> bool override;

    std::string lastError;

    struct CudaContext;

private:
    auto runOnHost(const State &past, const std::vector<int64_t> &ids, State &present,
                   std::vector<float> &lastLogits) -> bool;
    auto runOnDevice(const State &past, const std::vector<int64_t> &ids, State &present,
                     std::vector<float> &lastLogits) -> bool;

    // Shared with the device states, which must not outlive the session that allocated them.
    std::shared_ptr<Ort::Session> session;
    std::shared_ptr<CudaContext> cuda;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Dims dimsValue;
    size_t vocab = 0;
    bool hasPositionIds = false;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
};

} // namespace StreamingKv
