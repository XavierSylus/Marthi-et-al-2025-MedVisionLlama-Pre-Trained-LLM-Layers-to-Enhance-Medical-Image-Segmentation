import os
import torch
import torch.nn as nn
from src.models.layers import ImageToPatches3D, PatchEmbedding3D, SelfAttentionEncoderBlock, OutputProjection

DEBUG_SHAPES = os.environ.get("MEDV_DEBUG_SHAPES", "1") == "1"

def dbg(msg, x=None):
    if not DEBUG_SHAPES:
        return
    if x is None:
        print(f"[MEDV-DEBUG] {msg}", flush=True)
        return
    if isinstance(x, torch.Tensor):
        print(
            f"[MEDV-DEBUG] {msg}: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}",
            flush=True
        )
    else:
        print(f"[MEDV-DEBUG] {msg}: {x}", flush=True)

class ViT_Baseline(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, out_channels, embed_size, num_blocks, num_heads, dropout):
        super().__init__()
        self.image_size = tuple(image_size)
        self.patch_size = tuple(patch_size)
        self.embed_size = embed_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout

        self.i2p3d = ImageToPatches3D(image_size, patch_size)
        self.pe = PatchEmbedding3D(patch_size[0] * patch_size[1] * patch_size[2] * in_channels, embed_size)

        self.num_patches = (
            (image_size[0] // patch_size[0]) *
            (image_size[1] // patch_size[1]) *
            (image_size[2] // patch_size[2])
        )
        self.position_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_size))

        self.attention_blocks = nn.ModuleList(
            [SelfAttentionEncoderBlock(embed_size, num_heads, dropout) for _ in range(num_blocks)]
        )

        self.output_proj = OutputProjection(image_size, patch_size, embed_size, out_channels)
        self.sigmoid = nn.Sigmoid()
        self.vis_output_proj = OutputProjection(image_size, patch_size, embed_size, out_channels)

        dbg("========== ViT_Baseline init ==========")
        dbg("self.image_size", self.image_size)
        dbg("self.patch_size", self.patch_size)
        dbg("self.num_patches", self.num_patches)
        dbg("position_embed.shape", tuple(self.position_embed.shape))
        dbg("======================================")

    def forward(self, x, visualize=False):
        activations = []

        dbg("========== ViT_Baseline.forward ==========")
        dbg("forward.input_x", x)

        assert x.dim() == 5, f"Expected [B,C,H,W,D], got {tuple(x.shape)}"

        runtime_hwd = (x.shape[2], x.shape[3], x.shape[4])
        dbg("forward.runtime_input_hwd", runtime_hwd)

        expected_tokens = (
            (x.shape[2] // self.patch_size[0]) *
            (x.shape[3] // self.patch_size[1]) *
            (x.shape[4] // self.patch_size[2])
        )
        dbg("forward.expected_tokens_from_runtime_input", expected_tokens)

        activations.append(x.clone().detach())

        dbg("before i2p3d", x)
        x = self.i2p3d(x)
        dbg("after i2p3d", x)

        assert x.dim() == 3, f"after i2p3d expected [B,T,C], got {tuple(x.shape)}"
        assert x.shape[1] == expected_tokens, (
            f"Token mismatch after i2p3d: got {x.shape[1]}, expected {expected_tokens}"
        )

        dbg("before patch embedding", x)
        x = self.pe(x)
        dbg("after patch embedding", x)

        if visualize:
            dbg("before vis_output_proj", x)
            vis_output = self.vis_output_proj(x)
            vis_output = self.sigmoid(vis_output)
            activations.append(vis_output.clone().detach())

        dbg("before add position_embed", x)
        dbg("position_embed", self.position_embed)

        assert x.shape[1] == self.position_embed.shape[1], (
            f"Position embedding mismatch: x.shape={tuple(x.shape)}, "
            f"position_embed.shape={tuple(self.position_embed.shape)}"
        )

        x = x + self.position_embed
        dbg("after add position_embed", x)

        for idx, head in enumerate(self.attention_blocks):
            dbg(f"before attention_block_{idx}", x)
            x = head(x)
            dbg(f"after attention_block_{idx}", x)

        dbg("before output_proj", x)
        x = self.output_proj(x)
        dbg("after output_proj", x)

        activations.append(x.clone().detach())

        x = self.sigmoid(x)
        dbg("after sigmoid", x)
        activations.append(x.clone().detach())

        dbg("========== ViT_Baseline.forward end ==========")
        return x, activations