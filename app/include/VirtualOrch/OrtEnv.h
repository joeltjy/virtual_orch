#pragma once

#include <onnxruntime_cxx_api.h>

/**
 * Single process-wide ORT environment (multiple Env instances + CUDA can crash).
 *
 * Deliberately never destroyed: as a function-local static it was released by the exit handlers,
 * after the CUDA provider had started tearing down, and about half of the runs that had used a
 * CUDA session aborted at exit with "corrupted double-linked list" (the app on quit, and test
 * binaries after every test had passed; Nov 2026). The OS reclaims it at exit.
 */
inline auto sharedOrtEnv() -> Ort::Env & {
    static auto *env = new Ort::Env{ORT_LOGGING_LEVEL_WARNING, "VirtualOrch"};
    return *env;
}
