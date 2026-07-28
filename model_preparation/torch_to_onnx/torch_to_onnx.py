from onnxruntime.transformers.models.gpt2.gpt2_helper import Gpt2Helper
from onnxruntime.transformers.torch_onnx_export_helper import torch_onnx_export
from transformers import AutoConfig, GPT2LMHeadModel
import torch
import os
import itertools

model_path = "MODEL_PATH"
output_dir = "OUTPUT_DIR"

OUTPUT_ATTENTIONS = False      # Change those as needed for conversion
OUTPUT_HIDDEN_STATES = False   # Change those as needed for conversion

config = AutoConfig.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path, config=config)

dummy_input = torch.randint(
    low=0,
    high=config.vocab_size - 1,
    size=(1, 1024),
    dtype=torch.int32,
    device=torch.device("cpu"),
)

torch_onnx_export(
    model,
    args=tuple([dummy_input, { "output_attentions": OUTPUT_ATTENTIONS, "output_hidden_states": OUTPUT_HIDDEN_STATES }]),
    f=os.path.join(output_dir, "convertedModel.onnx"),
    export_params=True,
    input_names=["input_ids"],
    output_names=["logits"],
    opset_version=17,
    do_constant_folding=True,
    use_external_data_format=False,\
    verbose=False,
)
