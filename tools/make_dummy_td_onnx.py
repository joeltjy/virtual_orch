"""Write a tiny T14-shaped ONNX graph for tests (test/resources/iod_td_dummy.onnx).

Same I/O as the real td_grpo_unk_T14 export: input_ids int64 [batch, sequence] -> logits float32
[batch, sequence, 20178]. The logits are one fixed random vector expanded over every position, so
it loads in every ONNX Runtime build and exercises the C++ pipeline (context building, slot
sampling, masks, decode). Needs: pip install onnx numpy.
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

VOCAB = 20178
rng = np.random.default_rng(11)
base = rng.normal(0.0, 1.0, size=(1, 1, VOCAB)).astype(np.float32)

graph = helper.make_graph(
    nodes=[
        helper.make_node("Shape", ["input_ids"], ["ids_shape"]),
        helper.make_node("Concat", ["ids_shape", "vocab"], ["out_shape"], axis=0),
        helper.make_node("Expand", ["base_logits", "out_shape"], ["logits"]),
    ],
    name="iod_td_dummy",
    inputs=[helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"])],
    outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", "sequence", VOCAB])],
    initializer=[
        numpy_helper.from_array(base, "base_logits"),
        numpy_helper.from_array(np.array([VOCAB], dtype=np.int64), "vocab"),
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
out = sys.argv[1] if len(sys.argv) > 1 else "test/resources/iod_td_dummy.onnx"
onnx.save(model, out)
print("wrote", out)
