import torch
import torch.nn as nn
from typing import Tuple, Optional

class RectifiedFlow(nn.Module):
    """
    Rectified Flow implementation for ZUNA.
    Following Liu et al. (2022) and the ZUNA paper.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def get_velocity(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Computes the target velocity for training."""
        # x_0: Data tokens, x_1: Noise tokens, t: Time [B, 1]
        return x_1 - x_0

    def get_x_t(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between data and noise."""
        return (1 - t) * x_0 + t * x_1

    def loss(self, 
             x_0: torch.Tensor, 
             latent: torch.Tensor, 
             pos_indices: torch.Tensor,
             B: int, C: int, M: int) -> torch.Tensor:
        """
        Training loss implementation.
        x_0: [B, S, D] Data tokens
        latent: [B, SL, D] Encoder latent
        pos_indices: [B, S, 4]
        """
        t = torch.rand(B, 1, device=x_0.device)
        x_1 = torch.randn_like(x_0)
        
        x_t = self.get_x_t(x_0, x_1, t)
        v_target = self.get_velocity(x_0, x_1, t)
        
        # Decoder predicts velocity
        # Note: In the paper, decoder predicts velocity or noised signal.
        # We'll use velocity prediction for Rectified Flow.
        v_pred = self.model.decoder(x_t, latent, t, pos_indices)
        
        # MSE Loss
        loss = torch.mean((v_pred - v_target) ** 2)
        return loss

    @torch.no_grad()
    def sample(self, 
               latent: torch.Tensor, 
               pos_indices: torch.Tensor, 
               steps: int = 50) -> torch.Tensor:
        """
        Deterministic sampling using Euler ODE solver.
        latent: [B, SL, D]
        pos_indices: [B, S, 4]
        """
        B, S, D = pos_indices.shape[0], pos_indices.shape[1], latent.shape[-1]
        x_t = torch.randn(B, S, D, device=latent.device)
        
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B, 1), i / steps, device=latent.device)
            v_pred = self.model.decoder(x_t, latent, t, pos_indices)
            x_t = x_t - v_pred * dt # Note: if x_t = (1-t)x_0 + t*x1, then dx/dt = x1 - x0
            # So x_0 = x_1 - v * 1.0
            
        # At t=1, x_t is x_1 (noise). We want x_0.
        # Flow is x_t = x_0 + t(x_1 - x_0). dx/dt = x_1 - x_0 = v.
        # So x_0 = x_t - v*t. Here t=1 at start of reverse process?
        # Standard Rectified Flow: x_1 is noise, x_0 is data.
        # Forward: x_t = (1-t)x_0 + t*x_1. v = x_1 - x_0.
        # x_0 = x_t - t*v.
        # Sampling from t=1 down to t=0:
        # x_{t-dt} = x_t - v * dt.
        
        return x_t # At the end of loop, t=0, so x_t should be x_0.
