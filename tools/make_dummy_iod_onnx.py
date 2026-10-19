"""Write a tiny IOD-shaped ONNX graph for tests (test/resources/iod_dummy.onnx).

Same I/O as the real iod_pretrained export (input_ids, attention_mask, combo_in, note_valid ->
logits [1, 128, 5, 8785]), but the logits are one fixed random vector expanded over every note
and step. It exercises the C++ pipeline (windowing, combo feedback, masks, decode, ensemble)
without a real model and loads in every ONNX Runtime build. Needs: pip install onnx numpy.
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

N_NOTES, COMBO_LEN, VOCAB, SEQ = 128, 5, 8785, 384

rng = np.random.default_rng(7)
base = rng.normal(0.0, 1.0, size=(1, 1, 1, VOCAB)).astype(np.float32)

graph = helper.make_graph(
    nodes=[
        helper.make_node("Expand", ["base_logits", "out_shape"], ["logits"]),
    ],
    name="iod_dummy",
    inputs=[
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, SEQ]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, SEQ]),
        helper.make_tensor_value_info("combo_in", TensorProto.INT64, [1, N_NOTES, COMBO_LEN]),
        helper.make_tensor_value_info("note_valid", TensorProto.INT64, [1, N_NOTES]),
    ],
    outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, N_NOTES, COMBO_LEN, VOCAB])],
    initializer=[
        numpy_helper.from_array(base, "base_logits"),
        numpy_helper.from_array(np.array([1, N_NOTES, COMBO_LEN, VOCAB], dtype=np.int64), "out_shape"),
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
out = sys.argv[1] if len(sys.argv) > 1 else "test/resources/iod_dummy.onnx"
onnx.save(model, out)
print("wrote", out)
