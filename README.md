# JordanAI

This is a repository for the JordanAI project. The version we used for the September, 21st concert is in the
branch `september-concert-edits`.

I am currently on developing ports of the model to GGML, PyTorch, and CoreML using this branch network:

```
- - - - - - main
  | - - - - ggml-implementation
    | - - - pytorch-implementation  
        | - coreml-implementation
```

We'll have to rebase at some point.

## Model Porting Instructions

### PyTorch

Run the following:

```
from transformers import GPT2LMHeadModel
import torch

model = GPT2LMHeadModel.from_pretrained("/Users/lancelotblanchard/Downloads/zdvi9mq4_2000", torchscript=True)
traced = torch.jit.trace(model, tokens)
torch.jit.save_jit_module_to_flatbuffer(traced, "bassAndChords.pt")
```

> [!IMPORTANT]
> If exporting a model with `.to("mps")` or `.to("cuda")` for MPS or CUDA support, save it as `model.mps_pt`
> or `model.cuda_pt`. Unfortunately, the tracing process also captures the logic of the device for every tensor being
> created.