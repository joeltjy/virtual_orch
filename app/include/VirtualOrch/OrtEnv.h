#pragma once

#include <onnxruntime_cxx_api.h>

/** Single process-wide ORT environment (multiple Env instances + CUDA can crash). */
inline auto sharedOrtEnv() -> Ort::Env & {
    static Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "VirtualOrch"};
    return env;
}
