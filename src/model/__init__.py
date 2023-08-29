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
            embed_dim, embed_dim, kernel_size=(patch_size, 1), stride=patch_size
        )

        if norm is not None:
            self.norm = norm(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x, **kwargs):
        B, L, C = x.shape
        x = x.unsqueeze(-1).transpose(1, 2)  # [B, C, L, 1]
        x = self.project_layer(x)  # [B, E, P, 1]
        x = x.squeeze(-1)  # [B, E, P]
        x = x.transpose(1, 2)  # [B, P, E]

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
            for i in self.masked_patch_idx
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
    ):
        """
        Args:
            embed_dim: number of dimensions of embedding
            num_heads: number of heads in multi-head attention
            mlp_ratio: ratio of dimensions of hidden layer to embedding
            dropout: dropout rate
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
            encoder_layer=self.encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(embed_dim),
        )

    def forward(self, x, src_key_padding_mask=None):
        """
        input: [B, L, E]
        output: [B, L, E]
        """
        x = x * math.sqrt(self.embed_dim)

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x


class PTSM(nn.Module):
    """
    Pretrained Time Series Model.

    Notation:
        B: batch size
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

        self.inp_embed = nn.Linear(in_channels, embed_dim)

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            norm=None,
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
        )

        # project to input shape
        self.head = nn.Linear(embed_dim, in_channels)

    def forward(self, x, missing_mask=None):
        """
        x: [B, L, C]
        missing_mask: [B, L]
        output: [B, L, C]
        """
        device = x.device
        b, l, c = x.size()

        pos = torch.arange(0, l, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)
        pos_x = self.pos_embed(pos)  # [B, L, E]

        inp_x = self.inp_embed(x)  # [B, L, E]

        if missing_mask is not None:
            missing_mask = ~missing_mask.bool()
            inp_x = inp_x * missing_mask.unsqueeze(-1)

        pat_x = self.patch_embed(inp_x).repeat_interleave(
            self.patch_size, dim=1
        )  # [B, L, E]

        if missing_mask is not None:
            pat_x = pat_x * missing_mask.unsqueeze(-1)

        x = pos_x + inp_x + pat_x  # [B, L, E]

        if self.training:  # only mask during training
            masked_idx, _, _, _ = self.embed_mask()
            x[:, masked_idx, :] = 0.0
        else:
            masked_idx = list(range(self.input_len))

        x = self.encoder(x, src_key_padding_mask=~missing_mask)  # [B, L, E]

        x = self.head(x)

        return x, masked_idx

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
