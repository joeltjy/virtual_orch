#pragma once

#include <onnxruntime_cxx_api.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace OrtEnvDetail {

/**
 * Linux: load cuDNN 8 (libcudnn.so.8) by absolute path before the CUDA provider looks for it.
 * The provider's own search path is a fixed /usr/local/cuda-12.2/lib, so without
 * LD_LIBRARY_PATH it failed to load: every model fell back to the CPU and IodPretrained
 * (ot_robust) did not load at all, a silent orchestra (Nov 23 2026). Once loaded, the provider's
 * dependency on libcudnn.so.8 is satisfied by the loaded library; cuDNN finds its own parts
 * next to itself ($ORIGIN). Tried in order: the normal search (LD_LIBRARY_PATH),
 * $VIRTUAL_ORCH_CUDNN_DIR, ~/.local/lib/cudnn8, ~/miniconda3/lib, ~/anaconda3/lib.
 * Returns where it came from ("" = not found).
 */
inline auto preloadCudnn() -> std::string {
#if defined(__linux__)
    constexpr const char *kName = "libcudnn.so.8";
    if (dlopen(kName, RTLD_NOW | RTLD_GLOBAL) != nullptr)
        return "search path";
    std::string dirs[5];
    if (const char *d = std::getenv("VIRTUAL_ORCH_CUDNN_DIR"))
        dirs[0] = d;
    if (const char *home = std::getenv("HOME")) {
        dirs[1] = std::string(home) + "/.local/lib/cudnn8";
        dirs[2] = std::string(home) + "/miniconda3/lib";
        dirs[3] = std::string(home) + "/anaconda3/lib";
    }
    for (const auto &dir: dirs) {
        if (dir.empty())
            continue;
        const auto path = dir + "/" + kName;
        if (dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL) != nullptr)
            return path;
    }
    std::fprintf(stderr, "[ort] cuDNN 8 (libcudnn.so.8) not found: CUDA models will run on the CPU and "
                         "IodPretrained will not load. Set VIRTUAL_ORCH_CUDNN_DIR to its folder.\n");
#endif
    return {};
}

} // namespace OrtEnvDetail

/**
 * Single process-wide ORT environment (multiple Env instances + CUDA can crash). Created before
 * any session, so cuDNN is preloaded here (OrtEnvDetail::preloadCudnn).
 *
 * Deliberately never destroyed: as a function-local static it was released by the exit handlers,
 * after the CUDA provider had started tearing down (see also Main.cpp / TestMain.cpp).
 */
inline auto sharedOrtEnv() -> Ort::Env & {
    static auto *env = [] {
        const auto cudnn = OrtEnvDetail::preloadCudnn();
        if (! cudnn.empty())
            std::fprintf(stderr, "[ort] cuDNN 8 from %s\n", cudnn.c_str());
        return new Ort::Env{ORT_LOGGING_LEVEL_WARNING, "VirtualOrch"};
    }();
    return *env;
}
