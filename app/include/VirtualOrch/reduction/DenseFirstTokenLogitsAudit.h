#pragma once

#include <JuceHeader.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

/**
 * Debug audit: on the first generateNewToken after user input (N notes), capture
 * the raw ONNX logits for the first sampled field (onset of note N+1). At
 * shutdown, re-run the same N-note prefix through the model and print to the
 * console if logits differ.
 */
struct DenseFirstTokenLogitsAudit {
    bool captured = false;
    int prefixNotes = 0;
    int32_t inputDurationCs = 0;
    /** Packed dense events (onset,dur,note,vel) present when note N+1 was first generated. */
    std::vector<int32_t> prefixInputData;
    /** Exact token ids fed to ONNX for that first forward (after context/sort/relativize). */
    std::vector<int32_t> modelInputIds;
    /** Raw last-position logits before RT masks / bias (onset field). */
    std::vector<float> onlineLogits;

    auto reset() -> void { *this = DenseFirstTokenLogitsAudit{}; }

    auto captureFirst(const std::vector<int32_t> &inputDataSnapshot,
                      const std::vector<int32_t> &historyForModel,
                      const std::vector<float> &rawLogits,
                      int eventWidth,
                      int32_t inputDuration) -> void {
        if (captured)
            return;
        if (historyForModel.empty() || rawLogits.empty() || eventWidth <= 0)
            return;
        captured = true;
        prefixInputData = inputDataSnapshot;
        modelInputIds = historyForModel;
        onlineLogits = rawLogits;
        inputDurationCs = inputDuration;
        prefixNotes = static_cast<int>(historyForModel.size() / static_cast<size_t>(eventWidth));
    }

    /**
     * @param label          e.g. "ReductionTransformerV2"
     * @param rebuildHistory Rebuild model input from `prefixInputData` the same way
     *                       generateNewToken does (context trim, quantize, sort, relativize).
     * @param runLogits      Forward pass returning last-position logits.
     */
    auto verifyOffline(
        const char *label,
        const std::function<std::vector<int32_t>(const std::vector<int32_t> &prefix,
                                                 int32_t inputDuration)> &rebuildHistory,
        const std::function<std::vector<float>(std::vector<int32_t> &history)> &runLogits)
        const -> void {
        if (! captured) {
            std::cerr << "[" << label
                      << "] first-token logits audit: nothing captured (no first generate)\n";
            return;
        }

        auto rebuilt = rebuildHistory(prefixInputData, inputDurationCs);
        if (rebuilt != modelInputIds) {
            std::cerr << "[" << label
                      << "] first-token logits audit ERROR: offline prefix rebuild differs from "
                         "online model input (onlineInts="
                      << modelInputIds.size() << " offlineInts=" << rebuilt.size()
                      << " prefixNotes=" << prefixNotes << ")\n";
            const size_t n = std::min(rebuilt.size(), modelInputIds.size());
            size_t mismatches = 0;
            for (size_t i = 0; i < n; ++i) {
                if (rebuilt[i] != modelInputIds[i]) {
                    if (mismatches < 8)
                        std::cerr << "  id[" << i << "] online=" << modelInputIds[i]
                                  << " offline=" << rebuilt[i] << "\n";
                    ++mismatches;
                }
            }
            if (rebuilt.size() != modelInputIds.size())
                std::cerr << "  (length mismatch; compared first " << n << " ints, "
                          << mismatches << " value mismatches)\n";
            else
                std::cerr << "  (" << mismatches << " value mismatches)\n";
            // Still run logits on the *online* history so a prep bug does not hide a
            // forward-pass mismatch.
            rebuilt = modelInputIds;
        }

        auto offlineLogits = runLogits(rebuilt);
        if (offlineLogits.size() != onlineLogits.size()) {
            std::cerr << "[" << label
                      << "] first-token logits audit ERROR: logit size mismatch online="
                      << onlineLogits.size() << " offline=" << offlineLogits.size()
                      << " prefixNotes=" << prefixNotes << "\n";
            return;
        }

        constexpr float absTol = 1.0e-4f;
        constexpr float relTol = 1.0e-4f;
        float maxAbs = 0.0f;
        size_t mismatchCount = 0;
        size_t worstIdx = 0;
        for (size_t i = 0; i < onlineLogits.size(); ++i) {
            const float a = onlineLogits[i];
            const float b = offlineLogits[i];
            const float absDiff = std::fabs(a - b);
            const float scale = std::max(std::fabs(a), std::fabs(b));
            const float tol = absTol + relTol * scale;
            if (absDiff > tol) {
                if (absDiff > maxAbs) {
                    maxAbs = absDiff;
                    worstIdx = i;
                }
                ++mismatchCount;
            }
        }

        if (mismatchCount == 0) {
            std::cerr << "[" << label
                      << "] first-token logits audit OK (prefixNotes=" << prefixNotes
                      << " vocab=" << onlineLogits.size() << ")\n";
            return;
        }

        std::cerr << "[" << label
                  << "] first-token logits audit ERROR: " << mismatchCount << "/"
                  << onlineLogits.size() << " logits differ (prefixNotes=" << prefixNotes
                  << " maxAbsDiff=" << maxAbs << " at idx=" << worstIdx
                  << " online=" << onlineLogits[worstIdx]
                  << " offline=" << offlineLogits[worstIdx] << ")\n";
    }
};
