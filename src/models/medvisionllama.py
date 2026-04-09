import os
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn

from src.models.layers import (
    ImageToPatches3D,
    PatchEmbedding3D,
    SelfAttentionEncoderBlock,
    OutputProjection,
)

from src.llm.llama import LlamaTransformer

# 注意：
# 这里假设你的本地环境里已经有 lora 模块，且你之前已经能跑到 Llama forward。
# 如果你的项目里 lora 的真实 import 路径不同，请按你本地已修复版本替换这一行。
import loralib as lora


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG_SHAPES = os.environ.get("MEDV_DEBUG_SHAPES", "1") == "1"


def dbg(msg, x=None):
    if not DEBUG_SHAPES:
        return

    if x is None:
        print(f"[MEDV-DEBUG] {msg}", flush=True)
        return

    if isinstance(x, torch.Tensor):
        print(
            f"[MEDV-DEBUG] {msg}: "
            f"shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}",
            flush=True,
        )
    else:
        print(f"[MEDV-DEBUG] {msg}: {x}", flush=True)


class MedVisionLlama(nn.Module):
    """
    Vision Transformer with Llama integration for 3D medical image segmentation.
    """

    def __init__(
        self,
        image_size,
        patch_size,
        in_channels,
        out_channels,
        embed_size,
        num_blocks,
        num_heads,
        dropout,
        # llm_path="/path/to/your/project/LLaMA_v3.1/llama-3.1-8b",
        llm_path = "/root/autodl-tmp/llama-3.1-8b/original",
        lora_rank=4,
    ):
        super().__init__()

        self.image_size = tuple(image_size)
        self.patch_size = tuple(patch_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_size = embed_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout

        self.i2p3d = ImageToPatches3D(self.image_size, self.patch_size)
        self.pe = PatchEmbedding3D(
            self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * in_channels,
            embed_size,
        )

        self.num_patches = (
            (self.image_size[0] // self.patch_size[0]) *
            (self.image_size[1] // self.patch_size[1]) *
            (self.image_size[2] // self.patch_size[2])
        )
        self.position_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_size))

        llm_default_config = {
            "dim": 4096,
            "multiple_of": 256,
            "n_heads": 32,
            "n_layers": 32,
            "norm_eps": 1.0e-5,
            "vocab_size": -1,
            "first_layer": 31,
            "kv_heads": 8,
        }
        self.llm = LlamaTransformer(llm_default_config)

        logger.info("Loading Llama checkpoints")
        start_time = time.time()
        checkpoints = sorted(Path(llm_path).glob("*.pth"))
        if len(checkpoints) == 0:
            raise FileNotFoundError(f"No .pth checkpoint found under: {llm_path}")
        ckpt_path = checkpoints[0]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.llm.custom_load_state_dict(checkpoint, tail=True, strict=False)
        logger.info(f"Loaded in {time.time() - start_time:.2f} seconds")

        lora.mark_only_lora_as_trainable(self.llm)

        self.llm_dim_mapper1 = lora.Linear(embed_size, 4096, r=lora_rank)
        self.llm_dim_mapper2 = lora.Linear(4096, embed_size, r=lora_rank)

        self.attention_blocks = nn.ModuleList(
            [SelfAttentionEncoderBlock(embed_size, num_heads, dropout) for _ in range(num_blocks)]
        )

        self.output_proj = OutputProjection(self.image_size, self.patch_size, embed_size, out_channels)

        self.sigmoid = nn.Sigmoid()

        self.vis_output_proj = OutputProjection(self.image_size, self.patch_size, embed_size, out_channels)
        self.vis_output_proj_ll1 = OutputProjection(self.image_size, self.patch_size, 4096, out_channels)
        self.vis_output_proj_llm = OutputProjection(self.image_size, self.patch_size, 4096, out_channels)
        self.vis_output_proj_ll2 = OutputProjection(self.image_size, self.patch_size, embed_size, out_channels)

        dbg("========== MedVisionLlama init ==========")
        dbg("self.image_size", self.image_size)
        dbg("self.patch_size", self.patch_size)
        dbg("self.in_channels", self.in_channels)
        dbg("self.out_channels", self.out_channels)
        dbg("self.embed_size", self.embed_size)
        dbg("self.num_patches", self.num_patches)
        dbg("position_embed.shape", tuple(self.position_embed.shape))
        dbg("========================================")

    def forward(self, x, visualize=False):
        """
        Input:  [B, C, H, W, D]
        Output: [B, out_channels, H, W, D], activations
        """
        activations = []

        dbg("========== MedVisionLlama.forward ==========")
        dbg("forward.input_x", x)

        assert x.dim() == 5, f"Expected model input [B,C,H,W,D], got {tuple(x.shape)}"

        runtime_hwd = (x.shape[2], x.shape[3], x.shape[4])
        dbg("forward.runtime_input_hwd", runtime_hwd)

        expected_tokens_from_runtime_input = (
            (x.shape[2] // self.patch_size[0]) *
            (x.shape[3] // self.patch_size[1]) *
            (x.shape[4] // self.patch_size[2])
        )
        dbg("forward.expected_tokens_from_runtime_input", expected_tokens_from_runtime_input)
        dbg("forward.expected_tokens_from_model_config", self.num_patches)

        activations.append(x.clone().detach())

        # Step 1: image -> patches
        dbg("before i2p3d", x)
        x = self.i2p3d(x)
        dbg("after i2p3d", x)

        assert x.dim() == 3, f"after i2p3d expected [B,T,C], got {tuple(x.shape)}"
        dbg("after i2p3d token_count", x.shape[1])

        # Step 2: patch embedding
        dbg("before patch embedding", x)
        x = self.pe(x)
        dbg("after patch embedding", x)

        # Step 3: map to llama dim
        dbg("before llm_dim_mapper1", x)
        x = self.llm_dim_mapper1(x)
        dbg("after llm_dim_mapper1", x)

        if visualize:
            vis_output = self.vis_output_proj_ll1(x)
            vis_output = self.sigmoid(vis_output)
            activations.append(vis_output.clone().detach())

        # Step 4: llama
        dbg("before llama backbone", x)
        x = self.llm(x) + x
        dbg("after llama backbone", x)

        if visualize:
            vis_output = self.vis_output_proj_llm(x)
            vis_output = self.sigmoid(vis_output)
            activations.append(vis_output.clone().detach())

        # Step 5: map back
        dbg("before llm_dim_mapper2", x)
        x = self.llm_dim_mapper2(x)
        dbg("after llm_dim_mapper2", x)

        if visualize:
            vis_output = self.vis_output_proj_ll2(x)
            vis_output = self.sigmoid(vis_output)
            activations.append(vis_output.clone().detach())

        # Step 6: add positional embeddings
        dbg("before add position_embed", x)
        dbg("position_embed", self.position_embed)

        assert x.shape[1] == self.position_embed.shape[1], (
            f"Token mismatch before position add: x.shape={tuple(x.shape)}, "
            f"position_embed.shape={tuple(self.position_embed.shape)}"
        )

        x = x + self.position_embed
        dbg("after add position_embed", x)

        # Step 7: attention blocks
        for idx, head in enumerate(self.attention_blocks):
            dbg(f"before attention_block_{idx}", x)
            x = head(x)
            dbg(f"after attention_block_{idx}", x)

        # Step 8: output projection
        dbg("before output_proj", x)

        assert x.dim() == 3, f"before output_proj expected [B,T,C], got {tuple(x.shape)}"
        assert x.shape[1] == self.num_patches, (
            f"before output_proj token mismatch: got T={x.shape[1]}, expected={self.num_patches}, "
            f"image_size={self.image_size}, patch_size={self.patch_size}"
        )

        x = self.output_proj(x)
        dbg("after output_proj", x)

        activations.append(x.clone().detach())

        x = self.sigmoid(x)
        dbg("after sigmoid", x)

        activations.append(x.clone().detach())
        dbg("========== MedVisionLlama.forward end ==========")

        return x, activations