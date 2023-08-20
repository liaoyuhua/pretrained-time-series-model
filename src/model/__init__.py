from typing import Optional
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


class PTSM(nn.Module):
    """
    Pretrained Time Series Model
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, **kwargs):
        pass
