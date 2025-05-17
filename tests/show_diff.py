# SPDX-License-Identifier: Apache-2.0
import os

import torch

file_names = [
    f"cpu_offload.model.layers.{i}.{j}.pt" for i in range(28)
    for j in ("self_attn", "post_attention_layernorm", "mlp")
]

for file in file_names:
    offload = torch.load(os.path.join("saved_tensors", file))
    no_offload = torch.load(
        os.path.join("saved_tensors", file.replace("cpu_offload",
                                                   "no_offload")))
    diff = torch.abs(offload - no_offload)
    print(f"diff {file}: max {diff.max()}, mean {diff.mean()}")
