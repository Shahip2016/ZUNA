import torch
import torch.nn as nn
from .layers import RMSNorm, Attention, FeedForward
from .tokenizer import EEGTokenizer
from .rope_4d import RoPE4D

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, multiple_of: int = 256):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(dim, int(4 * dim * 2 / 3)) # SwiGLU style
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x, rope_4d, pos_indices):
        # Apply RoPE to the input or within attention
        h = x + self.attention(rope_4d(self.attention_norm(x), pos_indices))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class ZUNAEncoder(nn.Module):
    """
    ZUNA Encoder implementation.
    380M parameters (shared with Decoder blocks).
    """
    def __init__(self, 
                 d_model: int = 1024, 
                 n_layers: int = 16, 
                 n_heads: int = 16,
                 window_size: int = 32):
        super().__init__()
        self.tokenizer = EEGTokenizer(window_size=window_size, d_model=d_model)
        self.rope_4d = RoPE4D(dim=d_model)
        
        # Interleaved register tokens
        # The paper says d=1 downsampling, so for each data token, there's a register token.
        self.register_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        
    def forward(self, x, coords):
        # x: [B, C, T], coords: [B, C, 3]
        tokens, pos_indices = self.tokenizer(x, coords)
        B, S, D = tokens.shape
        
        # Interleave register tokens: [B, S, D] -> [B, 2*S, D]
        # Register token is r, data token is h: [r_1, h_1, r_2, h_2, ...]
        registers = self.register_token.expand(B, S, -1)
        
        # Interleave
        interleaved = torch.stack([registers, tokens], dim=2).view(B, 2 * S, D)
        
        # We need pos_indices for registers too. 
        # Paper says they share the same spatial position as the following data token.
        interleaved_pos = torch.stack([pos_indices, pos_indices], dim=2).view(B, 2 * S, 4)
        
        h = interleaved
        for layer in self.layers:
            h = layer(h, self.rope_4d, interleaved_pos)
            
        return self.norm(h)
