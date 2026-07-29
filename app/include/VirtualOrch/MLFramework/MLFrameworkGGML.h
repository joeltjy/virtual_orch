#pragma once
#ifdef ENABLE_GGML

#include <fstream>
#include <string>
#include <unordered_set>
#include <JuceHeader.h>

#include "VirtualOrch/MLFramework/MLFramework.h"

#include "ggml.h"
#include "ggml-alloc.h"

#ifdef ENABLE_GGML_BLAS
#include "ggml-blas.h"
#endif

#ifdef ENABLE_GGML_CPU
#include "ggml-cpu.h"
#endif

#ifdef ENABLE_GGML_CUDA
#include "ggml-cuda.h"
#endif

#ifdef ENABLE_GGML_METAL
#include "ggml-metal.h"
#endif

#ifdef ENABLE_GGML_SYCL
#include "ggml-sycl.h"
#endif

#ifdef ENABLE_GGML_VULKAN
#include "ggml-vulkan.h"
#endif

#ifdef ENABLE_GGML_ZENDNN
#include "ggml-zendnn.h"
#endif

#define MAX_NODES 4096

// Default hparams
struct ggml_hparams {
    int32_t n_vocab = 55028;
    int32_t n_ctx = 1024;
    int32_t n_embd = 768;
    int32_t n_head = 12;
    int32_t n_layer = 12;
    int32_t ftype = 1;
    float eps = 1e-5f;
};

struct ggml_layer {
    // normalization
    struct ggml_tensor *ln_1_g;
    struct ggml_tensor *ln_1_b;

    struct ggml_tensor *ln_2_g;
    struct ggml_tensor *ln_2_b;

    // attention
    struct ggml_tensor *c_attn_attn_w;
    struct ggml_tensor *c_attn_attn_b;

    struct ggml_tensor *c_attn_proj_w;
    struct ggml_tensor *c_attn_proj_b;

    // mlp
    struct ggml_tensor *c_mlp_fc_w;
    struct ggml_tensor *c_mlp_fc_b;

    struct ggml_tensor *c_mlp_proj_w;
    struct ggml_tensor *c_mlp_proj_b;
};

struct ggml_model {
    ggml_hparams hparams;

    // normalization
    struct ggml_tensor *ln_f_g;
    struct ggml_tensor *ln_f_b;

    struct ggml_tensor *wte; // position embedding
    struct ggml_tensor *wpe; // token embedding
    struct ggml_tensor *lm_head; // language model head

    std::vector<ggml_layer> layers;

    struct ggml_context *ctx_w;
    struct ggml_context *ctx_kv;

    ggml_backend_t backend = NULL;

    ggml_backend_buffer_t buffer_w;

    std::map<std::string, struct ggml_tensor *> tensors;
};

struct ggml_vocab {
    using id = int32_t;
    using token = std::string;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;
    std::vector<std::string> special_tokens;

    void add_special_token(const std::string &token);
};

class MLFrameworkGGML : public MLFramework {
public:
    MLFrameworkGGML();

    ~MLFrameworkGGML() override;

    std::vector<float> runModelAndGetLogits(std::vector<int32_t> &tokens, bool returnAllLogits = false) override;

    auto getBuiltAccelerators() const -> std::unordered_set<MLFramework::ACCELERATOR> override;

    auto getAvailableAccelerators() -> std::unordered_set<MLFramework::ACCELERATOR> override;

    void init(const std::string &modelPath, const ModelType &newModelType,
              MLFramework::ACCELERATOR accelerator) override;

private:
    void loadAccelerators();

    auto tryAcceleratorVulkan() const -> bool;

    auto tryAcceleratorSYCL() const -> bool;

    auto tryAcceleratorMetal() const -> bool;

    auto tryAcceleratorCUDA() const -> bool;

    auto tryAcceleratorCPU() const -> bool;

    bool ggml_load(const std::string &fname);

    // evaluate the transformer
    //
    //   - allocr:    ggml_gallocr to use to allocate the compute buffer
    //   - embd_inp:  the embeddings of the tokens in the context
    //   - embd_w:    the predicted logits for the next token
    //
    bool ggml_eval(
        const std::vector<ggml_vocab::id> &embd_inp,
        std::vector<float> &embd_w);

    // build the computation graph
    struct ggml_cgraph *ggml_graph(
        const int n_tokens);

    ggml_vocab vocab;

    ggml_model model;

    ggml_gallocr_t allocr = nullptr;

    int32_t n_ctx = 1024;

    bool accelerators_loaded = false;
    bool initialized = false;
};

#endif
