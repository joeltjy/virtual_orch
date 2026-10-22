#pragma once

#include <algorithm>
#include <string>
#include <vector>

/**
 * Which kind of ONNX export a reduction model is, from its input names, so a reduction type
 * refuses a model it cannot run (it would fail on every draw instead):
 *   - full-sequence models (V1, V2, Piano Reduction, AMT): input_ids -> logits;
 *   - KV-cache exports (T3'', T7'', T8''): input_ids + past_key + past_value -> logits + present_*.
 */
namespace OnnxModelKind {

[[nodiscard]] inline auto isKvExport(const std::vector<std::string> &inputNames) -> bool {
    const auto has = [&](const char *name) {
        return std::find(inputNames.begin(), inputNames.end(), name) != inputNames.end();
    };
    return has("past_key") || has("past_value");
}

/** Why a full-sequence reduction refuses a KV export. */
inline constexpr const char *kKvExportInFullSequenceReduction =
    "this is a KV-cache export (it takes past_key / past_value), made for "
    "Anticipated Piano Reduction (T3'') or Melody First. Pick that reduction type, or pick a "
    "full-sequence model for this one.";

} // namespace OnnxModelKind
