import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class RoPE4D(nn.Module):
    """
    4D Rotary Positional Encoding for ZUNA.
    Following Heo et al. (2024) and the ZUNA paper.
    """
    def __init__(self, dim: int, max_bins: int = 50):
        super().__init__()
        self.dim = dim
        self.max_bins = max_bins
        # Each dimension (x, y, z, t) gets a quarter of the total dimension
        self.dim_per_coord = dim // 4
        
        # Precompute theta for each dimension
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim_per_coord, 2).float() / self.dim_per_coord))
        self.register_buffer("inv_freq", inv_freq)

    def _get_sin_cos(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # pos: [Batch, Sequence, 4] where 4 are (x, y, z, t)
        
        # Expand pos to match inv_freq
        # [B, S, 4, 1] * [dim_per_coord / 2] -> [B, S, 4, dim_per_coord / 2]
        sinusoid_inp = torch.einsum("bsd,i->bsdi", pos, self.inv_freq)
        
        # [B, S, 4, dim_per_coord]
        sin = torch.sin(sinusoid_inp).repeat_interleave(2, dim=-1)
        cos = torch.cos(sinusoid_inp).repeat_interleave(2, dim=-1)
        
        # Reshape to [B, S, dim]
        sin = sin.reshape(sin.shape[0], sin.shape[1], -1)
        cos = cos.reshape(cos.shape[0], cos.shape[1], -1)
        
        return sin, cos

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, Seq, Dim]
        pos: [Batch, Seq, 4] - discretized bins [0, 49]
        """
        sin, cos = self._get_sin_cos(pos)
        
        # Rotate x: [x_1*cos - x_2*sin, x_1*sin + x_2*cos, ...]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        # We need to apply rotation per 4D dimension block
        # But since we reshaped sin/cos to [B, S, dim], we can do it globally
        # if the dimensions are aligned.
        
        rotated_x = torch.empty_like(x)
        rotated_x[..., 0::2] = x1 * cos[..., 0::2] - x2 * sin[..., 0::2]
        rotated_x[..., 1::2] = x1 * sin[..., 1::2] + x2 * cos[..., 1::2]
        
        return rotated_x

def discretize_coords(coords: torch.Tensor, n_bins: int = 50) -> torch.Tensor:
    """
    Discretizes 3D coordinates into fixed bins.
    coords: [..., 3]
    """
    # Assuming coords are normalized to [-1, 1] or similar
    # Shift to [0, 1]
    min_val = coords.min()
    max_val = coords.max()
    norm_coords = (coords - min_val) / (max_val - min_val + 1e-8)
    return (norm_coords * (n_bins - 1)).long()
