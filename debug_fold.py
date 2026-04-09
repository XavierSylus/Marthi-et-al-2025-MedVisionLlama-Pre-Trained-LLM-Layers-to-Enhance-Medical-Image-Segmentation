import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["MEDV_DEBUG_SHAPES"] = "1"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

import torch
from src.models.layers import OutputProjection

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device =", device, flush=True)

image_size = (128, 128, 128)
patch_size = (8, 8, 8)
embed_size = 64
out_channels = 1

model = OutputProjection(image_size, patch_size, embed_size, out_channels).to(device)
print("model built", flush=True)

x = torch.randn(1, 4096, 64, device=device)
print("input ready:", x.shape, flush=True)

with torch.no_grad():
    y = model(x)

print("y.shape =", y.shape, flush=True)
