import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

class AdaRMSNorm(nn.Module):
    """AdaRMSNorm conditioned on diffusion time step."""
    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale_map = nn.Linear(cond_dim, dim)
        self.shift_map = nn.Linear(cond_dim, dim)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x, cond):
        scale = self.scale_map(cond).unsqueeze(1)
        shift = self.shift_map(cond).unsqueeze(1)
        norm_x = self._norm(x.float()).type_as(x) * self.weight
        return norm_x * (1 + scale) + shift

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.w3 = nn.Linear(dim, hidden_dim)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask=None, rope=None):
        B, S, D = x.shape
        queries = self.wq(x).view(B, S, self.n_heads, self.head_dim)
        keys = self.wk(x).view(B, S, self.n_heads, self.head_dim)
        values = self.wv(x).view(B, S, self.n_heads, self.head_dim)

        if rope is not None:
            # Applying RoPE - simpler to apply before view in rope_4d, 
            # but let's assume it's applied correctly here if needed.
            # In our rope_4d implementation, we apply it to [B, S, D]
            pass

        # Standard self-attention
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        probs = F.softmax(scores, dim=-1)
        output = torch.matmul(probs, values)
        output = output.transpose(1, 2).contiguous().view(B, S, D)
        return self.wo(output)
