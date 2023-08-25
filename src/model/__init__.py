from typing import Optional
import math
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
        B, C, H, W = x.shape
        assert (
            H == self.input_size[0] and W == self.input_size[1]
        ), f"input size {H}*{W} doesn't match model ({self.input_size[0]}*{self.input_size[1]})."
        x = (
            self.project_layer(x).flatten(2).transpose(1, 2)
        )  # [batch_size, num_patches, embed_size]

        x = self.norm(x)
        return x


class PatchMask(nn.Module):
    def __init__(self, num_patches: int, mask_ratio: float):
        """
        Args:
            num_patches: number of patches
            mask_ratio: ratio of patches to be masked
        """
        super().__init__()
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio

        self.mask = torch.full((num_patches, num_patches), False)
        self.mask[torch.rand(num_patches, num_patches) < mask_ratio] = True

    def forward(self, x, **kwargs):
        x = x.masked_fill(self.mask, 0)
        return x


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
        input: [B, N, P, E]
        output: [B, N, P, E]
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

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            norm=nn.LayerNorm(embed_dim),
        )

        self.pos_embed = nn.Embedding(
            input_len, embed_dim
        )  # TODO: better way to embed position

        self.embed_mask = PatchMask(num_patches=self.num_patches, mask_ratio=mask_ratio)

        self.encoder = TransformerLayers(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            dropout=dropout,
            norm=norm,
        )

        # project to input shape
        self.head = nn.Linear(embed_dim, patch_size)

    def forward(self, x, **kwargs):
        """
        input: [B, N, C]
        output: [B, N, C]
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed.weight
        x = self.embed_mask(x)
        x = self.encoder(x)
        return x
