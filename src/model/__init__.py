from typing import Any, Optional, Tuple, List
import math
import random
import torch
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        norm: Optional[nn.Module] = None,
    ):
        """
        Args:
            patch_size: patch size
            in_channels: number of dimensions of input time series
            embed_dim: number of dimensions of embedding
            norm: normalization layer
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.norm = norm

        self.project_layer = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        if norm is not None:
            self.norm = norm(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x, **kwargs):
        B, N, L, C = x.shape
        x = x.unsqueeze(-1).transpose(2, 3)  # [B, N, C, L, 1]
        x = x.reshape(B * N, C, L, 1)  # [B*N, C, L, 1]
        x = self.project_layer(x)  # [B*N, E, P, 1]
        x = self.norm(x)
        x = x.squeeze(-1).view(B, N, self.embed_dim, -1)  # [B, N, E, P]
        x = x.transpose(2, 3)  # [B, N, P, E]
        return x


class PatchMask:
    def __init__(self, num_patches: int, mask_ratio: float, patch_size: int):
        """
        Args:
            num_patches: number of patches
            mask_ratio: ratio of patches to be masked
            patch_size: patch size
        """
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.num_masked_patches = int(num_patches * mask_ratio)

    def __call__(self) -> Tuple[List[int]]:
        """
        Returns:
            mask_idx: list of indices of masked time steps
            unmasked_idx: list of indices of unmasked time steps
            mask_patch_idx: list of indices of masked patches
            unmasked_patch_idx: list of indices of unmasked patches
        """
        mask = list(range(int(self.num_patches)))
        random.shuffle(mask)
        self.masked_patch_idx = sorted(mask[: self.num_masked_patches])
        self.masked_idx = [
            j
            for i in self.mask_patch_idx
            for j in list(range(i * self.patch_size, (i + 1) * self.patch_size))
        ]

        self.unmasked_patch_idx = sorted(mask[self.num_masked_patches :])
        self.unmasked_idx = [
            j
            for i in self.unmasked_patch_idx
            for j in list(range(i * self.patch_size, (i + 1) * self.patch_size))
        ]

        return (
            self.masked_idx,
            self.unmasked_idx,
            self.masked_patch_idx,
            self.unmasked_patch_idx,
        )


class TransformerLayers(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        depth: int,
        dropout: float = 0.1,
        norm: Optional[nn.Module] = None,
    ):
        """
        Args:
            embed_dim: number of dimensions of embedding
            num_heads: number of heads in multi-head attention
            mlp_ratio: ratio of dimensions of hidden layer to embedding
            dropout: dropout rate
            norm: normalization layer
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=depth, norm=norm
        )

    def forward(self, x):
        """
        input: [B, N, L, E]
        output: [B, N, L, E]
        """
        x = x * math.sqrt(self.embed_dim)
        x = self.encoder(x)
        return x


class PTSM(nn.Module):
    """
    Pretrained Time Series Model.

    Notation:
        B: batch size
        N: number of time series
        C: number of channels of input time series
        E: number of dimensions of embedding
        P: number of patches
    """

    def __init__(
        self,
        input_len: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        depth: int,
        mask_ratio: float,
        dropout: float = 0.1,
        norm: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.input_len = input_len
        self.patch_size = patch_size
        assert (
            input_len % patch_size == 0
        ), f"input length {input_len} should be divided by patch size {patch_size}."
        self.num_patches = input_len // patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.depth = depth
        self.mask_ratio = mask_ratio
        self.norm = norm

        self.inp_embed = nn.Linear(in_channels, embed_dim)

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            norm=nn.LayerNorm(embed_dim),
        )

        self.pos_embed = nn.Embedding(
            input_len, embed_dim
        )  # TODO: better way to embed position

        self.embed_mask = PatchMask(
            num_patches=self.num_patches, mask_ratio=mask_ratio, patch_size=patch_size
        )

        self.encoder = TransformerLayers(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            dropout=dropout,
            norm=norm,
        )

        # project to input shape
        self.head = nn.Linear(embed_dim, in_channels)

    def forward(self, x):
        """
        input: [B, N, L, C]
        output: [B, N, L, C]
        """
        pa_x = self.patch_embed(x).repeat_interleave(
            self.patch_size, dim=2
        )  # [B, N, L, E]
        i_x = self.inp_embed(x)  # [B, N, L, E]
        po_x = self.pos_embed(x)  # [B, N, L, E]
        x = pa_x + i_x + po_x  # [B, N, L, E]

        masked_idx, unmasked_idx, _, _ = self.embed_mask()
        x[:, :, masked_idx, :] = 0.0

        x = self.encoder(x)
        x = self.head(x)
        return x
