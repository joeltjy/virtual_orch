# benchmark_rt.py developed by Bob Tianqi Wei
"""Compare MLX and ONNX Runtime on the current streaming RoPE RT exports."""

import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import mlx.core as mx
import numpy as np
import onnx
import onnxruntime as ort
from onnx.numpy_helper import to_array


class StreamingTransformer:
    def __init__(self, path: Path, dtype: mx.Dtype):
        graph = onnx.load(path).graph
        self.weights = {t.name: mx.array(to_array(t)).astype(dtype) for t in graph.initializer}
        self.head = next(node.input[1] for node in graph.node if node.name == "/model/lm_head/MatMul")
        self.dtype = dtype
        self.layers = sum(name.endswith(".ln_1.weight") for name in self.weights)
        mx.eval(*self.weights.values())

    def linear(self, x, name):
        return mx.addmm(self.weights[name + ".bias"], x, self.weights[name + ".weight"])

    def norm(self, x, name):
        return mx.fast.layer_norm(x, self.weights[name + ".weight"], self.weights[name + ".bias"], 1e-5)

    def step(self, ids, past_key, past_value):
        prefix = "model.transformer."
        x = self.weights[prefix + "wte.weight"][ids]
        batch, length = ids.shape
        offset = past_key.shape[3]
        total = offset + length
        mask = mx.arange(total)[None, :] <= mx.arange(offset, total)[:, None]
        keys, values = [], []
        for layer in range(self.layers):
            name = f"{prefix}h.{layer}."
            qkv = self.linear(self.norm(x, name + "ln_1"), name + "attn.c_attn")
            q, k, v = [
                part.reshape(batch, length, 16, 64).transpose(0, 2, 1, 3)
                for part in mx.split(qkv, 3, axis=-1)
            ]
            k = mx.concatenate((past_key[layer], k), axis=2)
            v = mx.concatenate((past_value[layer], v), axis=2)
            keys.append(k)
            values.append(v)
            q = mx.fast.rope(q, 64, traditional=False, base=10000, scale=1, offset=offset)
            rotated = mx.fast.rope(k, 64, traditional=False, base=10000, scale=1, offset=0)
            attention = mx.fast.scaled_dot_product_attention(q, rotated, v, scale=0.125 / (layer + 1), mask=mask)
            x = x + self.linear(attention.transpose(0, 2, 1, 3).reshape(batch, length, -1),
                                name + "attn.c_proj")
            y = self.linear(self.norm(x, name + "ln_2"), name + "mlp.c_fc")
            y = 0.5 * y * (1 + mx.tanh(0.7978846 * (y + 0.044715 * y**3)))
            x = x + self.linear(y, name + "mlp.c_proj")
        logits = self.norm(x, prefix + "ln_f") @ self.weights[self.head]
        return logits, mx.stack(keys), mx.stack(values)


def benchmark(path: Path, dtype: mx.Dtype):
    cpu = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    gpu = StreamingTransformer(path, dtype)
    context = np.array([[token for i in range(32)
                         for token in (i * 25, 10030, 11060 + i % 12, 11700)]], dtype=np.int64)
    empty = np.empty((gpu.layers, 1, 16, 0, 64), dtype=np.float32)
    _, past_key, past_value = cpu.run(None, {
        "input_ids": context, "past_key": empty, "past_value": empty,
    })
    for batch in (1, 5):
        key_input = np.repeat(past_key, batch, axis=1)
        value_input = np.repeat(past_value, batch, axis=1)
        gpu_key_input = mx.array(key_input).astype(dtype)
        gpu_value_input = mx.array(value_input).astype(dtype)
        mx.eval(gpu_key_input, gpu_value_input)
        cpu_times, gpu_times, errors, cache_errors, agreements = [], [], [], [], []
        for i in range(16):
            ids = np.arange(batch, dtype=np.int64).reshape(batch, 1) + 11060 + i
            gpu_ids = mx.array(ids, dtype=mx.int32)
            start = perf_counter()
            expected, key, value = cpu.run(None, {
                "input_ids": ids, "past_key": key_input, "past_value": value_input,
            })
            cpu_times.append((perf_counter() - start) * 1000)
            start = perf_counter()
            result = gpu.step(gpu_ids, gpu_key_input, gpu_value_input)
            mx.eval(*result)
            gpu_times.append((perf_counter() - start) * 1000)
            logits, gpu_key, gpu_value = (np.asarray(t.astype(mx.float32)) for t in result)
            errors.append(float(np.max(np.abs(expected - logits))))
            cache_errors.append(max(float(np.max(np.abs(a - b))) for a, b in
                                    ((key, gpu_key), (value, gpu_value))))
            agreements.append(int(np.sum(expected.argmax(-1) == logits.argmax(-1))))
        print(f"batch {batch} | ONNX CPU {median(cpu_times[2:]):.2f} ms | "
              f"MLX GPU {median(gpu_times[2:]):.2f} ms | max logit error {max(errors):.5g}, KV error {max(cache_errors):.5g} | "
              f"logit argmax agreement {sum(agreements)}/{batch * len(agreements)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    mx.set_default_device(mx.gpu)
    benchmark(args.model, mx.float16 if args.fp16 else mx.float32)
