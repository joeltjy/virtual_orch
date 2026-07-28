import sys
import os
import struct
import onnx
import numpy as np
import re


# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means y ou need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


if len(sys.argv) < 2:
    print("Usage: onnx_to_ggml.py onnx-model-path [use-f32]\n")
    sys.exit(1)

# output in the same directory as the model
model_path = sys.argv[1]
fname_out = os.path.join(os.path.dirname(model_path), "ggml-model-f16.ggml")

# fake vocab.json
encoder = {str(i): i for i in range(55027)}
encoder["<|endoftext|>"] = 55027

# use 16-bit or 32-bit floats
use_f16 = True
if len(sys.argv) > 2:
    use_f16 = False
    fname_out = os.path.join(os.path.dirname(model_path), "ggml-model-f32.ggml")

# Load model
model = onnx.load(model_path)

# Collect state_dictionary
state_dict = {}

# Iterate through the model's weights
for initializer in model.graph.initializer:
    state_dict[initializer.name] = onnx.numpy_helper.to_array(initializer).squeeze()

vocab_size = state_dict["transformer.wte.weight"].shape[0]
n_positions = state_dict["transformer.wpe.weight"].shape[0]
n_embd = state_dict["transformer.wte.weight"].shape[1]
n_layer = max([int(name.split(".")[2]) for name in state_dict.keys() if name.startswith("transformer.h.")]) + 1

# It's trickier to retrieve n_head, as it's not explicitly stored. We can infer it from the shape of the attention weights.
potential_n_heads = set()

for node in model.graph.node:
    if node.op_type == "Reshape" and any(inp for inp in node.input if 'attn' in inp.lower()) and len(node.input) > 1:
        for node2 in model.graph.node:
            if node2.op_type == "Constant" and len(node2.output) > 0 and node2.output[0] == node.input[1]:
                shape = onnx.numpy_helper.to_array(node2.attribute[0].t)
                break
        if len(shape) == 4:
            potential_n_heads.add(shape[2])

assert len(potential_n_heads) == 1, f"Expected exactly one unique n_head value, found: {potential_n_heads}"

n_head = potential_n_heads.pop()

print("vocab_size:", vocab_size)
print("n_positions:", n_positions)
print("n_embd:", n_embd)
print("n_head:", n_head)
print("n_layer:", n_layer)

fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676d6c))  # magic: ggml in hex
fout.write(struct.pack("i", vocab_size))
fout.write(struct.pack("i", n_positions))
fout.write(struct.pack("i", n_embd))
fout.write(struct.pack("i", n_head))
fout.write(struct.pack("i", n_layer))
fout.write(struct.pack("i", use_f16))

byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

fout.write(struct.pack("i", len(encoder)))

for key in encoder:
    text = bytearray([byte_decoder[c] for c in key])
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

for name, data in state_dict.items():
    print("Processing variable: " + name + " with shape: ", data.shape)

    # rename headers to keep compatibility
    if name == "transformer.ln_f.weight":
        name = "model/ln_f/g"
    elif name == "transformer.ln_f.bias":
        name = "model/ln_f/b"
    elif name == "transformer.wte.weight":
        name = "model/wte"
    elif name == "transformer.wpe.weight":
        name = "model/wpe"
    elif name == "lm_head.weight" or name == "onnx::MatMul_6555":  # TODO (Lancelot): find a more robust way to identify the lm_head weight, as the name can vary in ONNX exports
        name = "model/lm_head"
    elif re.match(r"transformer.h\.\d+\.ln_1\.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/ln_1/g"
    elif re.match(r"transformer.h\.\d+\.ln_1\.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/ln_1/b"
    elif re.match(r"transformer.h\.\d+\.attn\.c_attn\.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/attn/c_attn/w"
    elif re.match(r"transformer.h\.\d+\.attn\.c_attn\.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/attn/c_attn/b"
    elif re.match(r"transformer.h\.\d+\.attn\.c_proj\.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/attn/c_proj/w"
    elif re.match(r"transformer.h.\d+.attn.c_proj.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/attn/c_proj/b"
    elif re.match(r"transformer.h.\d+.ln_2.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/ln_2/g"
    elif re.match(r"transformer.h.\d+.ln_2.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/ln_2/b"
    elif re.match(r"transformer.h.\d+.mlp.c_fc.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/mlp/c_fc/w"
    elif re.match(r"transformer.h.\d+.mlp.c_fc.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/mlp/c_fc/b"
    elif re.match(r"transformer.h.\d+.mlp.c_proj.weight", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/mlp/c_proj/w"
    elif re.match(r"transformer.h.\d+.mlp.c_proj.bias", name):
        i = re.findall("\d+", name)[0]
        name = f"model/h{i}/mlp/c_proj/b"
    else:
        print("Unrecognized variable name. %s", name)

    # we don't need these
    if name.endswith("attn.masked_bias") or name.endswith(".attn.bias"):
        print("  Skipping variable: " + name)
        continue

    n_dims = len(data.shape);

    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype = 0;
    if use_f16:
        if (name == "model/wte" or name == "model/lm_head" or name[-2:] == "/g" or name[-2:] == "/w") and n_dims == 2:
            print("  Converting to float16")
            data = data.astype(np.float16)
            ftype = 1
        else:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype = 0

    # for efficiency - transpose the projection matrices
    # "model/h.*/attn/c_attn/w"
    # "model/h.*/attn/c_proj/w"
    # "model/h.*/mlp/c_fc/w"
    # "model/h.*/mlp/c_proj/w"
    if name[-14:] == "/attn/c_attn/w" or \
            name[-14:] == "/attn/c_proj/w" or \
            name[-11:] == "/mlp/c_fc/w" or \
            name[-13:] == "/mlp/c_proj/w":
        print("  Transposing")
        data = data.transpose()

    # We also need to transpose the lm_head in ONNX
    if name == "model/lm_head":
        print("  Transposing lm_head")
        data = data.transpose()

    # header
    str = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str), ftype))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str);

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print("")
