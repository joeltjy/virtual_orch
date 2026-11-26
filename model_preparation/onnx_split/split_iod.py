# split_iod.py developed by Bob Tianqi Wei
"""Split the ot_robust ONNX export into the two graphs IodPretrained can load."""

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper
from onnx.utils import Extractor


def split(source: Path, destination: Path) -> None:
    model = onnx.load(source)
    graph = model.graph
    boundary = "/backbone/Reshape_2_output_0"
    inputs = {value.name: value for value in graph.input}
    weights = {weight.name: weight for weight in graph.initializer}
    batch, sequence = inputs["input_ids"].type.tensor_type.shape.dim
    hidden_size = weights["inner.backbone.ln_f.weight"].dims[0]
    hidden_shape = [batch.dim_param or batch.dim_value, sequence.dim_value, hidden_size]
    graph.value_info.append(helper.make_tensor_value_info(boundary, TensorProto.FLOAT, hidden_shape))
    encoder = Extractor(model).extract_model(["input_ids", "attention_mask"], [boundary])

    # Select one note before computing the vocabulary projection for its five steps.
    head = next(node for node in graph.node if "/combo_head/MatMul_output_0" in node.output)
    head_index = next(index for index, node in enumerate(graph.node) if node == head)
    graph.node.insert(head_index, helper.make_node(
        "Gather", [head.input[0], "note_index"], ["selected_note"], axis=1))
    graph.node[head_index + 1].input[0] = "selected_note"
    graph.input.append(helper.make_tensor_value_info("note_index", TensorProto.INT64, [1]))
    graph.output[0].type.tensor_type.shape.dim[1].dim_value = 1
    decoder = Extractor(model).extract_model([boundary, "combo_in", "note_valid", "note_index"], ["logits"])
    decoder.graph.input[0].name = "hidden"
    for node in decoder.graph.node:
        for index, name in enumerate(node.input):
            if name == boundary:
                node.input[index] = "hidden"

    destination.mkdir(parents=True, exist_ok=True)
    for suffix, part in (("enc", encoder), ("dec", decoder)):
        onnx.checker.check_model(part)
        onnx.save(part, destination / f"{source.stem}_{suffix}.onnx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    split(args.model, args.destination)
