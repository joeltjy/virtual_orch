#ifdef ENABLE_GGML

#include <string>
#include <thread>
#include <unordered_set>

#include "VirtualOrch/MLFramework/MLFrameworkGGML.h"

MLFrameworkGGML::MLFrameworkGGML() : MLFramework() {
}

MLFrameworkGGML::~MLFrameworkGGML() {
    if (initialized) {
        ggml_free(model.ctx_w);

        ggml_gallocr_free(allocr);
        ggml_backend_buffer_free(model.buffer_w);
        ggml_backend_free(model.backend);
    }
}

auto MLFrameworkGGML::getBuiltAccelerators() const -> std::unordered_set<MLFramework::ACCELERATOR> {
    return std::unordered_set<MLFramework::ACCELERATOR>{
#if defined ENABLE_GGML_VULKAN
        MLFramework::ACCELERATOR::GGML_VULKAN,
#endif
#if defined ENABLE_GGML_SYCL
        MLFramework::ACCELERATOR::GGML_SYCL,
#endif
#if defined ENABLE_GGML_METAL
        MLFramework::ACCELERATOR::GGML_METAL,
#endif
#if defined ENABLE_GGML_CUDA
        MLFramework::ACCELERATOR::GGML_CUDA,
#endif
#if defined ENABLE_GGML_CPU
        MLFramework::ACCELERATOR::GGML_CPU,
#endif
    };
}

auto MLFrameworkGGML::getAvailableAccelerators() -> std::unordered_set<MLFramework::ACCELERATOR> {
    std::unordered_set<MLFramework::ACCELERATOR> availableAccels;
    loadAccelerators();
    for (const auto &accel: getBuiltAccelerators()) {
        switch (accel) {
            case MLFramework::ACCELERATOR::GGML_VULKAN:
                if (tryAcceleratorVulkan()) {
                    availableAccels.insert(MLFramework::ACCELERATOR::GGML_VULKAN);
                }
                break;
            case MLFramework::ACCELERATOR::GGML_SYCL:
                if (tryAcceleratorSYCL()) {
                    availableAccels.insert(MLFramework::ACCELERATOR::GGML_SYCL);
                }
                break;
            case MLFramework::ACCELERATOR::GGML_METAL:
                if (tryAcceleratorMetal()) {
                    availableAccels.insert(MLFramework::ACCELERATOR::GGML_METAL);
                }
                break;
            case MLFramework::ACCELERATOR::GGML_CUDA:
                if (tryAcceleratorCUDA()) {
                    availableAccels.insert(MLFramework::ACCELERATOR::GGML_CUDA);
                }
                break;
            case MLFramework::ACCELERATOR::GGML_CPU:
                if (tryAcceleratorCPU()) {
                    availableAccels.insert(MLFramework::ACCELERATOR::GGML_CPU);
                }
                break;
            default:
                throw std::runtime_error("Unexpected accelerator when testing available accelerators for GGML.");
                break;
        }
    }
#ifdef DEBUG
    for (const auto &name: getBuiltAccelerators()) {
        std::string accelName = MLFramework::ACCELERATOR_FRIENDLY_NAMES.at(name);
        DBG("GGML built provider: " + juce::String(accelName));
    }
    for (const auto name: availableAccels) {
        std::string accelName = MLFramework::ACCELERATOR_FRIENDLY_NAMES.at(name);
        DBG("GGML usable provider: " + juce::String(accelName));
    }
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t ggml_reg = ggml_backend_reg_get(i);
        if (ggml_reg) {
            DBG("GGML registered backend: " + juce::String(ggml_backend_reg_name(ggml_reg)));
        }
    }
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t ggml_dev = ggml_backend_dev_get(i);
        if (ggml_dev) {
            ggml_backend_reg_t ggml_reg = ggml_backend_dev_backend_reg(ggml_dev);
            DBG("GGML registered device: " + juce::String(ggml_backend_dev_name(ggml_dev)) + " [desc: " + juce::String(
                    ggml_backend_dev_description(ggml_dev)) + ", reg: " + juce::String(ggml_backend_reg_name(ggml_reg))
                +
                "]");
        }
    }
#endif
    return availableAccels;
}

void MLFrameworkGGML::loadAccelerators() {
    if (!accelerators_loaded) {
        const char *backend_dir_path = std::getenv("GGML_BACKEND_DIR_PATH");
        if (backend_dir_path) {
            ggml_backend_load_all_from_path(backend_dir_path);
        } else {
            ggml_backend_load_all();
        }
        accelerators_loaded = true;
    }
}

auto MLFrameworkGGML::tryAcceleratorVulkan() const -> bool {
    bool result = false;
#ifdef ENABLE_GGML_VULKAN
    ggml_backend_t backend = ggml_backend_init_by_name("Vulkan0", NULL);
    if (backend) {
        result = true;
        ggml_backend_free(backend);
    }
#endif
    return result;
}

auto MLFrameworkGGML::tryAcceleratorSYCL() const -> bool {
    bool result = false;
#ifdef ENABLE_GGML_SYCL
    ggml_backend_t backend = ggml_backend_init_by_name("SYCL0", NULL);
    if (backend) {
        result = true;
        ggml_backend_free(backend);
    }
#endif
    return result;
}

auto MLFrameworkGGML::tryAcceleratorMetal() const -> bool {
    bool result = false;
#ifdef ENABLE_GGML_METAL
    ggml_backend_t backend = ggml_backend_init_by_name("Metal", NULL);
    if (backend) {
        result = true;
        ggml_backend_free(backend);
    }
#endif
    return result;
}

auto MLFrameworkGGML::tryAcceleratorCUDA() const -> bool {
    bool result = false;
#ifdef ENABLE_GGML_CUDA
    ggml_backend_t backend = ggml_backend_init_by_name("CUDA0", 0);
    if (backend) {
        result = true;
        ggml_backend_free(backend);
    }
#endif
    return result;
}

auto MLFrameworkGGML::tryAcceleratorCPU() const -> bool {
    bool result = false;
#ifdef ENABLE_GGML_CPU
    ggml_backend_t backend = ggml_backend_init_by_name("CPU", NULL);
    if (backend) {
        result = true;
        ggml_backend_free(backend);
    }
#endif
    return result;
}

void MLFrameworkGGML::init(const std::string &modelPath, const ModelType &newModelType,
                           MLFramework::ACCELERATOR accelerator) {
    initialized = true;

    loadAccelerators();

    switch (accelerator) {
#ifdef ENABLE_GGML_CPU
        case MLFramework::ACCELERATOR::GGML_CPU:
            fprintf(stderr, "%s: using CPU backend\n", __func__);;
            model.backend = ggml_backend_init_by_name("CPU", NULL);
            if (!model.backend) {
                fprintf(stderr, "%s: ggml_backend_init_by_name() failed\n", __func__);
                throw std::runtime_error("Failed to initialize GGML CPU accelerator for model");
            }
            break;
#endif
#ifdef ENABLE_GGML_CUDA
        case MLFramework::ACCELERATOR::GGML_CUDA:
            fprintf(stderr, "%s: using CUDA backend\n", __func__);
            model.backend = ggml_backend_init_by_name("CUDA0", 0);
            if (!model.backend) {
                fprintf(stderr, "%s: ggml_backend_init_by_name() failed\n", __func__);
                throw std::runtime_error("Failed to initialize GGML NVIDIA CUDA accelerator for model");
            }
            break;
#endif
#ifdef ENABLE_GGML_METAL
        case MLFramework::ACCELERATOR::GGML_METAL:
            fprintf(stderr, "%s: using Metal backend\n", __func__);
            model.backend = ggml_backend_init_by_name("Metal", NULL);
            if (!model.backend) {
                fprintf(stderr, "%s: ggml_backend_init_by_name() failed\n", __func__);
                throw std::runtime_error("Failed to initialize GGML Apple Metal accelerator for model");
            }
            break;
#endif
#ifdef ENABLE_GGML_SYCL
        case MLFramework::ACCELERATOR::GGML_SYCL:
            fprintf(stderr, "%s: using Vulkan backend\n", __func__);
            model.backend = ggml_backend_init_by_name("SYCL0", NULL);
            if (!model.backend) {
                fprintf(stderr, "%s: ggml_backend_init_by_name() failed\n", __func__);
                throw std::runtime_error("Failed to initialize GGML SYCL accelerator for model");
            }
            break;
#endif
#ifdef ENABLE_GGML_VULKAN
        case MLFramework::ACCELERATOR::GGML_VULKAN:
            fprintf(stderr, "%s: using Vulkan backend\n", __func__);
            model.backend = ggml_backend_init_by_name("Vulkan0", NULL);
            if (!model.backend) {
                fprintf(stderr, "%s: ggml_backend_init_by_name() failed\n", __func__);
                throw std::runtime_error("Failed to initialize GGML Vulkan accelerator for model");
            }
            break;
#endif
        default:
            throw std::runtime_error("Unsupported accelerator for GGML.");
            break;
    }

    // Set backend options
#ifdef ENABLE_GGML_CPU
    size_t available_threads = std::thread::hardware_concurrency();
    size_t n_threads = 1;
    if (available_threads > MLFramework::IDEAL_MIN_RESERVED_THREADS_NON_FRAMEWORK) {
        n_threads = available_threads - MLFramework::IDEAL_MIN_RESERVED_THREADS_NON_FRAMEWORK;
    } else if (available_threads > MLFramework::UNIDEAL_MIN_RESERVED_THREADS_FRAMEWORK) {
        n_threads = available_threads - MLFramework::UNIDEAL_MIN_RESERVED_THREADS_FRAMEWORK;
    }
    DBG("GGML CPU backend: setting number of threads to " + juce::String(n_threads) + " (available: " +
        juce::String(available_threads) + ")");

    ggml_backend_dev_t dev = ggml_backend_get_device(model.backend);
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(model.backend, static_cast<int>(n_threads));
            }
        }
    }
#endif

    fprintf(stderr, "%s: initialized GGML backend '%s'\n", __func__, ggml_backend_name(model.backend));

    // Load the model
    if (!ggml_load(modelPath)) {
        throw std::runtime_error("Failed to load model");
    }

    // allocate the compute buffer
    {
        // create a graph allocator with the backend's default buffer type
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        // create the worst case graph for memory usage estimation
        int n_tokens = 120 + 3 + 1;
        int n_past = model.hparams.n_ctx - 1;
        struct ggml_cgraph *gf = ggml_graph(n_tokens);

        // pre-allocate the compute buffer for the worst case (optional)
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);
        fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size / 1024.0 / 1024.0);
    }
}

std::vector<float> MLFrameworkGGML::runModelAndGetLogits(std::vector<int32_t> &tokens, bool returnAllLogits) {
    std::vector<float> logits;
    // This GGML implementations performs inference by batch, so
    // we need to pass notes 1 by 1, since it already has the context
    // stored in memory (memory_k and memory_v).
    // Here we pass the last token in the tokens vector.
    // std::vector<int32_t> newTokens{tokens.back()};
    // TODO: I reverted back to n_past=0. Does that work?
    int n_past = 0;
    ggml_eval(tokens, logits);
    return logits;
}

bool MLFrameworkGGML::ggml_load(const std::string &fname) {
    printf("%s: loading model from '%s'\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    // load hparams
    {
        auto &hparams = model.hparams;

        fin.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        fin.read((char *) &hparams.n_ctx, sizeof(hparams.n_ctx));
        fin.read((char *) &hparams.n_embd, sizeof(hparams.n_embd));
        fin.read((char *) &hparams.n_head, sizeof(hparams.n_head));
        fin.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *) &hparams.ftype, sizeof(hparams.ftype));

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: ftype   = %d\n", __func__, hparams.ftype);
        printf("%s: qntvr   = %d\n", __func__, qntvr);

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        fin.read((char *) &n_vocab, sizeof(n_vocab));

        if (n_vocab != model.hparams.n_vocab) {
            fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
                    __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
            return false;
        }

        std::string word;
        std::vector<char> buf(128);

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            fin.read((char *) &len, sizeof(len));

            buf.resize(len);
            fin.read((char *) buf.data(), len);
            word.assign(buf.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype) (model.hparams.ftype));
    if (wtype == GGML_TYPE_COUNT) {
        fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n",
                __func__, fname.c_str(), model.hparams.ftype);
        return false;
    }

    auto &ctx = model.ctx_w;

    // create the ggml context
    {
        size_t n_tensors = 2 + 6 + 12 * model.hparams.n_layer;
        struct ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };

        ctx = ggml_init(params);
        if (!ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // create the tensors for the model
    {
        const auto &hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.ln_f_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.ln_f_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        model.wte = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);
        model.wpe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ctx);
        model.lm_head = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);

        // map by name
        model.tensors["model/ln_f/g"] = model.ln_f_g;
        model.tensors["model/ln_f/b"] = model.ln_f_b;

        model.tensors["model/wte"] = model.wte;
        model.tensors["model/wpe"] = model.wpe;
        model.tensors["model/lm_head"] = model.lm_head;

        for (int i = 0; i < n_layer; ++i) {
            auto &layer = model.layers[i];

            layer.ln_1_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.ln_2_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.c_attn_attn_w = ggml_new_tensor_2d(ctx, wtype, n_embd, 3 * n_embd);
            layer.c_attn_attn_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * n_embd);

            layer.c_attn_proj_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.c_attn_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.c_mlp_fc_w = ggml_new_tensor_2d(ctx, wtype, n_embd, 4 * n_embd);
            layer.c_mlp_fc_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * n_embd);

            layer.c_mlp_proj_w = ggml_new_tensor_2d(ctx, wtype, 4 * n_embd, n_embd);
            layer.c_mlp_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            // map by name
            model.tensors["model/h" + std::to_string(i) + "/ln_1/g"] = layer.ln_1_g;
            model.tensors["model/h" + std::to_string(i) + "/ln_1/b"] = layer.ln_1_b;

            model.tensors["model/h" + std::to_string(i) + "/ln_2/g"] = layer.ln_2_g;
            model.tensors["model/h" + std::to_string(i) + "/ln_2/b"] = layer.ln_2_b;

            model.tensors["model/h" + std::to_string(i) + "/attn/c_attn/w"] = layer.c_attn_attn_w;
            model.tensors["model/h" + std::to_string(i) + "/attn/c_attn/b"] = layer.c_attn_attn_b;

            model.tensors["model/h" + std::to_string(i) + "/attn/c_proj/w"] = layer.c_attn_proj_w;
            model.tensors["model/h" + std::to_string(i) + "/attn/c_proj/b"] = layer.c_attn_proj_b;

            model.tensors["model/h" + std::to_string(i) + "/mlp/c_fc/w"] = layer.c_mlp_fc_w;
            model.tensors["model/h" + std::to_string(i) + "/mlp/c_fc/b"] = layer.c_mlp_fc_b;

            model.tensors["model/h" + std::to_string(i) + "/mlp/c_proj/w"] = layer.c_mlp_proj_w;
            model.tensors["model/h" + std::to_string(i) + "/mlp/c_proj/b"] = layer.c_mlp_proj_b;
        }
    }

    // allocate the model tensors in a backend buffer
    model.buffer_w = ggml_backend_alloc_ctx_tensors(ctx, model.backend);

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %6.2f MB\n", __func__,
           ggml_backend_buffer_get_size(model.buffer_w) / (1024.0 * 1024.0));

    // override the default training context with the user-provided
    model.hparams.n_ctx = n_ctx;

    // key + value memory
    {
        auto *ctx = model.ctx_kv;

        // create the ggml context
        {
            size_t n_tensors = 2;
            struct ggml_init_params params = {
                /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors,
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ctx = ggml_init(params);
            if (!ctx) {
                fprintf(stderr, "%s: ggml_init() failed\n", __func__);
                return false;
            }
        }
    }

    // load weights
    {
        size_t total_size = 0;

        bool has_lm_head = false;

        std::vector<char> read_buf;

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ttype), sizeof(ttype));

            if (fin.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = {1, 1};
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.c_str());
                return false;
            }

            auto tensor = model.tensors[name];
            ggml_set_name(tensor, name.c_str());
            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.c_str());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                        __func__, name.c_str(), (int) tensor->ne[0], (int) tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            // for debugging
            if (0) {
                printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n", name.c_str(), ne[0], ne[1],
                       ggml_type_name(ggml_type(ttype)), ggml_nbytes(tensor) / 1024.0 / 1024.0, ggml_nbytes(tensor));
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.c_str(), ggml_nbytes(tensor), nelements * bpe);
                return false;
            }

            if (ggml_backend_buffer_is_host(model.buffer_w)) {
                // for some backends such as CPU and Metal, the tensor data is in system memory and we can read directly into it
                fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(ggml_nbytes(tensor));
                fin.read(read_buf.data(), ggml_nbytes(tensor));
                ggml_backend_tensor_set(tensor, read_buf.data(), 0, ggml_nbytes(tensor));
            }

            // GPT-2 models share the WTE tensor as the LM head
            if (name == "model/wte" && has_lm_head == false) {
                //ggml_backend_tensor_copy(tensor, model.lm_head);
                model.lm_head = tensor;
            }

            if (name == "model/lm_head") {
                has_lm_head = true;
            }

            total_size += ggml_nbytes(tensor);
        }

        printf("%s: model size  = %8.2f MB\n", __func__, total_size / 1024.0 / 1024.0);
    }

    fin.close();

    return true;
}

// build the computation graph
struct ggml_cgraph *MLFrameworkGGML::ggml_graph(
    const int n_tokens) {
    const int N = n_tokens;

    const auto &hparams = model.hparams;

    const int n_embd = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx = hparams.n_ctx;
    const int n_head = hparams.n_head;

    // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
    static size_t buf_size = ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead_custom(MAX_NODES, false);
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };

    struct ggml_context *ctx = ggml_init(params);

    struct ggml_cgraph *gf = ggml_new_graph_custom(ctx, MAX_NODES, false);

    struct ggml_tensor *embd = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    // at this point, the tensor data is not allocated yet and cannot be set
    // we will find the tensor after the graph is allocated by its name, and set the data then
    ggml_set_name(embd, "embd");
    // setting a tensor as an input will ensure that it is allocated at the beginning of the graph
    // this is important to ensure that the input tensors are not overwritten before they are used
    ggml_set_input(embd);

    struct ggml_tensor *position = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(position, "position");
    ggml_set_input(position);

    // wte + wpe
    struct ggml_tensor *inpL =
            ggml_add(ctx,
                     ggml_get_rows(ctx, model.wte, embd),
                     ggml_get_rows(ctx, model.wpe, position));

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor *KQ_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N, N, 1);
    ggml_set_name(KQ_mask, "KQ_mask");
    ggml_set_input(KQ_mask);

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor *cur;

        // norm
        {
            // [ 768, N]
            cur = ggml_norm(ctx, inpL, hparams.eps);

            // cur = ln_1_g*cur + ln_1_b
            // [ 768, N]
            cur = ggml_add(ctx,
                           ggml_mul(ctx,
                                    cur,
                                    model.layers[il].ln_1_g),
                           model.layers[il].ln_1_b);
        }

        // attn
        // [2304, 768] - model.layers[il].c_attn_attn_w
        // [2304,   1] - model.layers[il].c_attn_attn_b
        // [ 768,   N] - cur (in)
        // [2304,   N] - cur (out)
        //
        // cur = attn_w*cur + attn_b
        // [2304, N]
        {
            cur = ggml_mul_mat(ctx,
                               model.layers[il].c_attn_attn_w,
                               cur);

            cur = ggml_add(ctx,
                           cur,
                           model.layers[il].c_attn_attn_b);
        }

        // self-attention
        {
            struct ggml_tensor *Qcur = ggml_view_2d(ctx, cur, n_embd, N, cur->nb[1], 0 * sizeof(float) * n_embd);
            struct ggml_tensor *Kcur = ggml_view_2d(ctx, cur, n_embd, N, cur->nb[1], 1 * sizeof(float) * n_embd);
            struct ggml_tensor *Vcur = ggml_view_2d(ctx, cur, n_embd, N, cur->nb[1], 2 * sizeof(float) * n_embd);

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            // [64, N, 12]
            struct ggml_tensor *Q =
                    ggml_permute(ctx,
                                 ggml_cont_3d(ctx, Qcur, n_embd / n_head, n_head, N),
                                 0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            // [64, N, 12]
            struct ggml_tensor *K =
                    ggml_permute(ctx,
                                 ggml_cont_3d(ctx, Kcur, n_embd / n_head, n_head, N),
                                 0, 2, 1, 3);

            // GG: flash attention
            //struct ggml_tensor * V =
            //    ggml_cpy(ctx0,
            //            ggml_permute(ctx0,
            //                ggml_reshape_3d(ctx0,
            //                    ggml_view_1d(ctx0, model.memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_v)*n_embd),
            //                    n_embd/n_head, n_head, n_past + N),
            //                1, 2, 0, 3),
            //            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_past + N, n_embd/n_head, n_head));

            //struct ggml_tensor * KQV = ggml_flash_attn(ctx0, Q, K, V, true);

            // K * Q
            // [N, N, 12]
            struct ggml_tensor *KQ = ggml_mul_mat(ctx, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            // [N, N, 12]
            struct ggml_tensor *KQ_scaled =
                    ggml_scale(ctx,
                               KQ,
                               1.0f / (float(il + 1) * sqrtf(float(n_embd) / n_head)));

            // KQ_masked = mask_past(KQ_scaled)
            // [N, N, 12]
            struct ggml_tensor *KQ_masked = ggml_add(ctx, KQ_scaled, KQ_mask);

            // KQ = soft_max(KQ_masked)
            // [N, N, 12]
            struct ggml_tensor *KQ_soft_max = ggml_soft_max(ctx, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, N).permute(1, 2, 0, 3).contiguous()
            // [N, 64, 12]
            struct ggml_tensor *V_trans =
                    ggml_cont_3d(ctx,
                                 ggml_permute(ctx,
                                              ggml_cont_3d(ctx, Vcur, n_embd / n_head, n_head, N),
                                              1, 2, 0, 3),
                                 N, n_embd / n_head, n_head);

            // KQV = transpose(V) * KQ_soft_max
            // [64, N, 12]
            struct ggml_tensor *KQV = ggml_mul_mat(ctx, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            // [64, 12, N]
            struct ggml_tensor *KQV_merged = ggml_permute(ctx, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            // [768, N]
            cur = ggml_cont_2d(ctx, KQV_merged, n_embd, N);
        }

        // projection
        // [ 768, 768] - model.layers[il].c_attn_proj_w
        // [ 768,   1] - model.layers[il].c_attn_proj_b
        // [ 768,   N] - cur (in)
        // [ 768,   N] - cur (out)
        //
        // cur = proj_w*cur + proj_b
        // [768, N]
        {
            cur = ggml_mul_mat(ctx,
                               model.layers[il].c_attn_proj_w,
                               cur);

            cur = ggml_add(ctx,
                           cur,
                           model.layers[il].c_attn_proj_b);
        }

        // add the input
        cur = ggml_add(ctx, cur, inpL);

        struct ggml_tensor *inpFF = cur;

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx, inpFF, hparams.eps);

                // cur = ln_2_g*cur + ln_2_b
                // [ 768, N]
                cur = ggml_add(ctx,
                               ggml_mul(ctx,
                                        cur,
                                        model.layers[il].ln_2_g),
                               model.layers[il].ln_2_b);
            }

            // fully connected
            // [3072, 768] - model.layers[il].c_mlp_fc_w
            // [3072,   1] - model.layers[il].c_mlp_fc_b
            // [ 768,   N] - cur (in)
            // [3072,   N] - cur (out)
            //
            // cur = fc_w*cur + fc_b
            // [3072, N]
            cur = ggml_mul_mat(ctx,
                               model.layers[il].c_mlp_fc_w,
                               cur);

            cur = ggml_add(ctx,
                           cur,
                           model.layers[il].c_mlp_fc_b);

            // GELU activation
            // [3072, N]
            cur = ggml_gelu(ctx, cur);

            // projection
            // [ 768, 3072] - model.layers[il].c_mlp_proj_w
            // [ 768,    1] - model.layers[il].c_mlp_proj_b
            // [3072,    N] - cur (in)
            // [ 768,    N] - cur (out)
            //
            // cur = proj_w*cur + proj_b
            // [768, N]
            cur = ggml_mul_mat(ctx,
                               model.layers[il].c_mlp_proj_w,
                               cur);

            cur = ggml_add(ctx,
                           cur,
                           model.layers[il].c_mlp_proj_b);
        }

        // input for next layer
        inpL = ggml_add(ctx, cur, inpFF);
    }

    // norm
    {
        // [ 768, N]
        inpL = ggml_norm(ctx, inpL, hparams.eps);

        // inpL = ln_f_g*inpL + ln_f_b
        // [ 768, N]
        inpL = ggml_add(ctx,
                        ggml_mul(ctx,
                                 inpL,
                                 model.ln_f_g),
                        model.ln_f_b);
    }

    // inpL = WTE * inpL
    // [ 768, 50257] - model.lm_head
    // [ 768, N]     - inpL
    inpL = ggml_mul_mat(ctx, model.lm_head, inpL);
    ggml_set_name(inpL, "logits");
    // setting a tensor as the output will ensure that it is not overwritten by subsequent operations
    ggml_set_output(inpL);

    // logits -> probs
    //inpL = ggml_soft_max(ctx0, inpL);

    ggml_build_forward_expand(gf, inpL);

    ggml_free(ctx);

    return gf;
}

// evaluate the transformer
//
//   - model:     the model
//   - allocr:    ggml_gallocr to use to allocate the compute buffer
//   - n_threads: number of threads to use
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
bool MLFrameworkGGML::ggml_eval(
    const std::vector<ggml_vocab::id> &embd_inp,
    std::vector<float> &embd_w) {
    const int N = embd_inp.size();

    const auto &hparams = model.hparams;

    const int n_vocab = hparams.n_vocab;

    struct ggml_cgraph *gf = ggml_graph(embd_inp.size());

    // allocate the graph tensors
    ggml_gallocr_alloc_graph(allocr, gf);

    // set the graph inputs
    struct ggml_tensor *embd = ggml_graph_get_tensor(gf, "embd");
    ggml_backend_tensor_set(embd, embd_inp.data(), 0, N * ggml_element_size(embd));

    struct ggml_tensor *position = ggml_graph_get_tensor(gf, "position");
    for (int i = 0; i < N; ++i) {
        int32_t v = i;
        ggml_backend_tensor_set(position, &v, i * sizeof(int32_t), sizeof(v));
    }

    struct ggml_tensor *KQ_mask = ggml_graph_get_tensor(gf, "KQ_mask");
    std::vector<float> data_buf(N * N);
    for (int j = 0; j < N; ++j) {
        for (int i = j + 1; i < N; ++i) {
            data_buf[j * N + i] = -INFINITY;
        }
    }
    ggml_backend_tensor_set(KQ_mask, data_buf.data(), 0, data_buf.size() * sizeof(float));
    // TODO (Lancelot): pre-allocate KQ_mask at graph loading time.

    // run the computation
    ggml_backend_graph_compute(model.backend, gf);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    // get the graph outputs
    struct ggml_tensor *logits = ggml_graph_get_tensor(gf, "logits");

    //embd_w.resize(n_vocab*N);
    //ggml_backend_tensor_get(logits, embd_w.data(), 0, sizeof(float)*n_vocab*N);

    // return result just for the last token
    embd_w.resize(n_vocab);
    ggml_backend_tensor_get(logits, embd_w.data(), (n_vocab * (N - 1)) * sizeof(float), sizeof(float) * n_vocab);

    return true;
}

#endif
