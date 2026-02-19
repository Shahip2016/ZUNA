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

class DecoderBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, cond_dim: int):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.cross_attention = CrossAttention(dim, n_heads)
        self.feed_forward = FeedForward(dim, int(4 * dim * 2 / 3))
        
        self.attention_norm = AdaRMSNorm(dim, cond_dim)
        self.cross_attn_norm = AdaRMSNorm(dim, cond_dim)
        self.ffn_norm = AdaRMSNorm(dim, cond_dim)

    def forward(self, x, latent, cond, rope_4d, pos_indices):
        h = x + self.attention(rope_4d(self.attention_norm(x, cond), pos_indices))
        h = h + self.cross_attention(self.cross_attn_norm(h, cond), latent)
        out = h + self.feed_forward(self.ffn_norm(h, cond))
        return out

class ZUNADecoder(nn.Module):
    """
    ZUNA Decoder implementation.
    Receives noised tokens and conditions on encoder latent + time step.
    """
    def __init__(self, 
                 d_model: int = 1024, 
                 n_layers: int = 16, 
                 n_heads: int = 16,
                 cond_dim: int = 256):
        super().__init__()
        self.rope_4d = RoPE4D(dim=d_model)
        self.time_embed = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, cond_dim) for _ in range(n_layers)
        ])
        self.norm = AdaRMSNorm(d_model, cond_dim)
        
    def forward(self, x_noised, latent, t, pos_indices):
        # x_noised: [B, S, D], latent: [B, SL, D], t: [B, 1]
        cond = self.time_embed(t)
        
        h = x_noised
        for layer in self.layers:
            h = layer(h, latent, cond, self.rope_4d, pos_indices)
            
        return self.norm(h, cond)

class ZUNAModel(nn.Module):
    """
    Full ZUNA Diffusion Autoencoder.
    """
    def __init__(self, 
                 d_model: int = 1024, 
                 n_layers: int = 16, 
                 n_heads: int = 16,
                 window_size: int = 32):
        super().__init__()
        self.encoder = ZUNAEncoder(d_model, n_layers, n_heads, window_size)
        self.decoder = ZUNADecoder(d_model, n_layers, n_heads)
        # Detokenizer uses the same window size/model dim
        from .tokenizer import EEGDetokenizer
        self.detokenizer = EEGDetokenizer(window_size=window_size, d_model=d_model)
        
    def forward_encoder(self, x, coords):
        return self.encoder(x, coords)
        
    def forward_decoder(self, tokens_noised, latent, t, pos_indices, B, C, M):
        # 1. Decoder pass
        h = self.decoder(tokens_noised, latent, t, pos_indices)
        
        # 2. Detokenization
        # We need to remove registers before detokenizing if they were interleaved
        # In this implementation, they are interleaved by the encoder.
        # But the decoder receives noised tokens which should match the encoder output shape.
        # If registers are interleaved, detokenizer needs to handle them.
        # Let's assume detokenizer expects only data tokens.
        
        # registers at 0, 2, 4... data at 1, 3, 5...
        data_tokens = h[:, 1::2, :]
        
        return self.detokenizer(data_tokens, B, C, M)
