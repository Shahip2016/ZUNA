import torch
import torch.nn as nn
from typing import Tuple, List

class EEGTokenizer(nn.Module):
    """
    Rasterized channel-time tokenizer for ZUNA.
    Segments EEG channels into windows and computes tokens.
    """
    def __init__(self, 
                 window_size: int = 32, 
                 d_model: int = 1024,
                 n_bins: int = 50):
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model
        
        # token-encoder MLP: [window_size] -> [d_model]
        self.encoder = nn.Sequential(
            nn.Linear(window_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [Batch, Channels, Time]
        coords: [Batch, Channels, 3] (Discretized x,y,z)
        
        Returns:
            tokens: [Batch, Seq, d_model]
            pos_indices: [Batch, Seq, 4] (x, y, z, t)
        """
        B, C, T = x.shape
        M = T // self.window_size # Number of coarse windows per channel
        
        # Reshape to [B, C, M, window_size]
        x_windows = x.reshape(B, C, M, self.window_size)
        
        # Encode windows: [B, C, M, d_model]
        tokens_3d = self.encoder(x_windows)
        
        # Raster-scan order: all channels for time 1, then all for time 2...
        # Current shape: [B, C, M, d_model]
        # Transpose to [B, M, C, d_model] and flatten
        tokens = tokens_3d.transpose(1, 2).reshape(B, M * C, self.d_model)
        
        # Compute 4D position indices
        # coords: [B, C, 3]
        # time_indices: [M]
        time_indices = torch.arange(M, device=x.device) # [M]
        
        # pos_indices should be [B, M*C, 4]
        # For each coarse time m and each channel c: (coords[b,c,0], coords[b,c,1], coords[b,c,2], m)
        
        # [B, 1, C, 3] -> [B, M, C, 3]
        spatial_pos = coords.unsqueeze(1).repeat(1, M, 1, 1)
        # [1, M, 1, 1] -> [B, M, C, 1]
        temporal_pos = time_indices.reshape(1, M, 1, 1).repeat(B, 1, C, 1)
        
        # Concatenate: [B, M, C, 4]
        pos_indices_4d = torch.cat([spatial_pos, temporal_pos], dim=-1)
        
        # Transpose and flatten like tokens: [B, M*C, 4]
        pos_indices = pos_indices_4d.reshape(B, M * C, 4)
        
        return tokens, pos_indices

class EEGDetokenizer(nn.Module):
    """Inverse of the tokenizer for reconstruction."""
    def __init__(self, window_size: int = 32, d_model: int = 1024):
        super().__init__()
        self.window_size = window_size
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, window_size)
        )
        
    def forward(self, tokens: torch.Tensor, B: int, C: int, M: int) -> torch.Tensor:
        """
        tokens: [Batch, M*C, d_model]
        Returns: [Batch, Channels, Time]
        """
        # [Batch, M*C, window_size]
        x_windows = self.decoder(tokens)
        
        # Reshape back to [Batch, M, C, window_size]
        x_3d = x_windows.reshape(B, M, C, self.window_size)
        
        # Transpose to [Batch, Channels, M, window_size] and reshape to [B, C, T]
        x = x_3d.transpose(1, 2).reshape(B, C, M * self.window_size)
        
        return x
