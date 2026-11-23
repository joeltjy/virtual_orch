#pragma once

#include "VirtualOrch/orchestration-models/IodPretrained.h"

/**
 * IodPretrained fine-tuned for expressive timing (2026-11-23): iod_pretrained_nolora checkpoint-29000 trained on
 * a 70 / 30 mix of the normal IOD corpus and the piano concertos' full orchestra time-warped by the expressive
 * piano (orchestration/overnight/warp_full_orchestra.py, mix_iod_expr30.py). Same graph and decoding as
 * IodPretrained; only the weights differ (instrument_octave_predictions_v2.onnx).
 */
class IodPretrainedExpressive : public IodPretrained {
public:
    static constexpr const char *kOnnxFile = "instrument_octave_predictions_v2.onnx";
    [[nodiscard]] auto getName() const -> std::string override { return "IodPretrainedExpressive"; }
};
