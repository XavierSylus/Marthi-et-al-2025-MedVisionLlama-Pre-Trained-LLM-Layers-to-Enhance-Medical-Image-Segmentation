import os
import dataclasses

import torch
import torch.nn as nn
import unfoldNd


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


class ImageToPatches3D(nn.Module):
    """
    Converts a 3D image into non-overlapping patches for ViT processing.
    Input:  [B, C, H, W, D]
    Output: [B, T, patch_volume * C]
    """

    def __init__(self, image_size, patch_size):
        super().__init__()
        self.image_size = tuple(image_size)
        self.patch_size = tuple(patch_size)
        self.unfold = unfoldNd.UnfoldNd(kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        assert len(x.size()) == 5, f"ImageToPatches3D expects [B,C,H,W,D], got {tuple(x.shape)}"

        dbg("=== ImageToPatches3D.forward ===")
        dbg("i2p3d.input", x)
        dbg("i2p3d.image_size", self.image_size)
        dbg("i2p3d.patch_size", self.patch_size)

        h, w, d = x.shape[2], x.shape[3], x.shape[4]
        ph, pw, pd = self.patch_size

        assert h % ph == 0 and w % pw == 0 and d % pd == 0, (
            f"Input spatial dims must be divisible by patch_size. "
            f"Got input={(h, w, d)}, patch_size={self.patch_size}"
        )

        expected_tokens = (h // ph) * (w // pw) * (d // pd)
        dbg("i2p3d.expected_tokens_from_runtime_input", expected_tokens)

        x_unfolded = self.unfold(x)
        dbg("i2p3d.after_unfold", x_unfolded)  # [B, patch_volume*C, T]

        assert x_unfolded.dim() == 3, (
            f"Unfold output must be 3D [B, patch_volume*C, T], got {tuple(x_unfolded.shape)}"
        )
        assert x_unfolded.shape[-1] == expected_tokens, (
            f"Unfold token mismatch: got T={x_unfolded.shape[-1]}, expected={expected_tokens}, "
            f"input={(h, w, d)}, patch_size={self.patch_size}"
        )

        x_unfolded = x_unfolded.permute(0, 2, 1).contiguous()
        dbg("i2p3d.after_permute", x_unfolded)  # [B, T, patch_volume*C]

        assert x_unfolded.shape[1] == expected_tokens, (
            f"Patch sequence mismatch after permute: got T={x_unfolded.shape[1]}, "
            f"expected={expected_tokens}"
        )

        dbg("=== ImageToPatches3D.forward end ===")
        return x_unfolded


class PatchEmbedding3D(nn.Module):
    """
    Embeds flattened 3D patches.
    Input:  [B, T, patch_volume * C]
    Output: [B, T, embed_size]
    """

    def __init__(self, in_channels, embed_size):
        super().__init__()
        self.in_channels = in_channels
        self.embed_size = embed_size
        self.embed_layer = nn.Linear(in_features=in_channels, out_features=embed_size)

    def forward(self, x):
        assert len(x.size()) == 3, f"PatchEmbedding3D expects [B,T,C], got {tuple(x.shape)}"
        dbg("PatchEmbedding3D.input", x)
        x = self.embed_layer(x)
        dbg("PatchEmbedding3D.output", x)
        return x


class MultiLayerPerceptron(nn.Module):
    def __init__(self, embed_size, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.layers(x)


class SelfAttentionEncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.embed_size = embed_size
        self.ln1 = nn.LayerNorm(embed_size)
        self.mha = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = MultiLayerPerceptron(embed_size, dropout)

    def forward(self, x):
        y = self.ln1(x)
        x = x + self.mha(y, y, y, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x


# class OutputProjection(nn.Module):
#     """
#     Projects token embeddings back to 3D image space.

#     Expected input:  [B, T, embed_size]
#     After linear:    [B, T, patch_volume * out_channels]
#     Before FoldNd:   [B, patch_volume * out_channels, T]
#     After FoldNd:    [B, out_channels, H, W, D]
#     """

#     def __init__(self, image_size, patch_size, embed_size, out_channels):
#         super().__init__()
#         self.image_size = tuple(image_size)
#         self.patch_size = tuple(patch_size)
#         self.embed_size = embed_size
#         self.out_channels = out_channels

#         self.patch_volume = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
#         self.projection = nn.Linear(embed_size, self.patch_volume * out_channels)
#         self.fold = unfoldNd.FoldNd(
#             output_size=self.image_size,
#             kernel_size=self.patch_size,
#             stride=self.patch_size,
#         )

#     def forward(self, x):
#         dbg("=== OutputProjection.forward ===")
#         dbg("output_proj.input", x)

#         assert x.dim() == 3, f"OutputProjection expects [B,T,C], got {tuple(x.shape)}"
#         b, t, c = x.shape
#         dbg("output_proj.input_B_T_C", (b, t, c))
#         dbg("output_proj.image_size", self.image_size)
#         dbg("output_proj.patch_size", self.patch_size)

#         expected_tokens = (
#             (self.image_size[0] // self.patch_size[0]) *
#             (self.image_size[1] // self.patch_size[1]) *
#             (self.image_size[2] // self.patch_size[2])
#         )
#         dbg("output_proj.expected_tokens", expected_tokens)

#         assert t == expected_tokens, (
#             f"OutputProjection token mismatch before projection: got T={t}, expected={expected_tokens}, "
#             f"image_size={self.image_size}, patch_size={self.patch_size}"
#         )

#         x = self.projection(x)
#         dbg("output_proj.after_projection", x)

#         assert x.dim() == 3, f"After projection expected [B,T,C2], got {tuple(x.shape)}"
#         assert x.shape[-1] == self.patch_volume * self.out_channels, (
#             f"Projection channel mismatch: got C2={x.shape[-1]}, "
#             f"expected={self.patch_volume * self.out_channels}"
#         )

#         x = x.permute(0, 2, 1).contiguous()
#         dbg("output_proj.after_permute_for_fold", x)

#         assert x.shape[-1] == expected_tokens, (
#             f"Fold input token mismatch: got L={x.shape[-1]}, expected={expected_tokens}"
#         )

#         x = self.fold(x)
#         dbg("output_proj.after_fold", x)

#         dbg("=== OutputProjection.forward end ===")
#         return x


class OutputProjection(nn.Module):
    """
    Projects token embeddings back to 3D image space for non-overlapping patches.

    Input:
        x: [B, T, embed_size]

    Output:
        [B, out_channels, H, W, D]
    """

    def __init__(self, image_size, patch_size, embed_size, out_channels):
        super().__init__()
        self.image_size = tuple(image_size)
        self.patch_size = tuple(patch_size)
        self.embed_size = embed_size
        self.out_channels = out_channels

        self.patch_volume = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        self.projection = nn.Linear(embed_size, self.patch_volume * out_channels)

    def forward(self, x):
        dbg("=== OutputProjection.forward ===")
        dbg("output_proj.input", x)

        assert x.dim() == 3, f"OutputProjection expects [B,T,C], got {tuple(x.shape)}"
        b, t, c = x.shape

        H, W, D = self.image_size
        pH, pW, pD = self.patch_size

        assert H % pH == 0 and W % pW == 0 and D % pD == 0, (
            f"image_size must be divisible by patch_size, got image_size={self.image_size}, patch_size={self.patch_size}"
        )

        nH = H // pH
        nW = W // pW
        nD = D // pD
        expected_tokens = nH * nW * nD

        dbg("output_proj.image_size", self.image_size)
        dbg("output_proj.patch_size", self.patch_size)
        dbg("output_proj.expected_tokens", expected_tokens)

        assert t == expected_tokens, (
            f"OutputProjection token mismatch: got T={t}, expected={expected_tokens}, "
            f"image_size={self.image_size}, patch_size={self.patch_size}"
        )

        # [B, T, out_channels * pH * pW * pD]
        x = self.projection(x)
        dbg("output_proj.after_projection", x)

        expected_patch_dim = self.out_channels * pH * pW * pD
        assert x.shape[-1] == expected_patch_dim, (
            f"Projected patch dim mismatch: got {x.shape[-1]}, expected {expected_patch_dim}"
        )

        # [B, nH, nW, nD, out_channels, pH, pW, pD]
        x = x.view(b, nH, nW, nD, self.out_channels, pH, pW, pD)
        dbg("output_proj.after_view_grid", x)

        # -> [B, out_channels, nH, pH, nW, pW, nD, pD]
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        dbg("output_proj.after_permute", x)

        # -> [B, out_channels, H, W, D]
        x = x.view(b, self.out_channels, H, W, D)
        dbg("output_proj.after_reconstruct", x)

        dbg("=== OutputProjection.forward end ===")
        return x


@dataclasses.dataclass
class ViTArgs:
    image_size: tuple = (64, 64, 64)
    patch_size: tuple = (16, 16, 16)
    in_channels: int = 1
    out_channels: int = 1
    embed_size: int = 64
    num_blocks: int = 16
    num_heads: int = 4
    dropout: float = 0.2


@dataclasses.dataclass
class ViTArgs_Depth:
    image_size: tuple = (64, 64, 64)
    patch_size: tuple = (16, 16, 16)
    in_channels: int = 1
    out_channels: int = 1
    embed_size: int = 1024
    num_blocks: int = 20
    num_heads: int = 32
    dropout: float = 0.1