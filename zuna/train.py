import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .model import ZUNAModel
from .diffusion import RectifiedFlow
from .dataset import EEGDataset, collate_fn
import numpy as np

def compute_mmd(x, y, kernel='rbf'):
    """
    Maximum Mean Discrepancy (MMD) loss for encoder regularization.
    """
    def gaussian_kernel(x, y, sigma=1.0):
        # x: [B, D], y: [B, D]
        dist = torch.cdist(x, y).pow(2)
        return torch.exp(-dist / (2 * sigma ** 2))

    if kernel == 'rbf':
        k_xx = gaussian_kernel(x, x).mean()
        k_yy = gaussian_kernel(y, y).mean()
        k_xy = gaussian_kernel(x, y).mean()
        return k_xx + k_yy - 2 * k_xy
    return torch.tensor(0.0, device=x.device)

def train(model_config, train_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Model
    model = ZUNAModel(**model_config).to(device)
    rf = RectifiedFlow(model)
    
    # 2. Setup Dataset
    dataset = EEGDataset(is_training=True, **train_config['dataset_params'])
    dataloader = DataLoader(dataset, batch_size=train_config['batch_size'], 
                             shuffle=True, collate_fn=collate_fn)
    
    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), 
                            lr=train_config['lr'], 
                            betas=(0.9, 0.95), 
                            weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                    T_max=train_config['steps'], 
                                                    eta_min=1e-6)
    
    # 4. Training Loop
    model.train()
    step = 0
    while step < train_config['steps']:
        for x_in, x_cl, coords, masks in dataloader:
            x_in, x_cl, coords = x_in.to(device), x_cl.to(device), coords.to(device)
            B, C, T = x_in.shape
            M = T // model_config['window_size']
            
            optimizer.zero_grad()
            
            # Encoder Pass
            latent = model.forward_encoder(x_in, coords)
            
            # Tokenize targets for diffusion loss
            # target_tokens: [B, S, D], target_pos: [B, S, 4]
            target_tokens, target_pos = model.encoder.tokenizer(x_cl, coords)
            
            # Diffusion Loss (Rectified Flow)
            diff_loss = rf.loss(target_tokens, latent, target_pos, B, C, M)
            
            # Auxiliary MMD Loss on latent
            # Regularize tokens against a standard normal distribution
            prior_samples = torch.randn_like(latent)
            mmd_loss = compute_mmd(latent.reshape(-1, latent.shape[-1]), 
                                  prior_samples.reshape(-1, prior_samples.shape[-1]))
            
            total_loss = diff_loss + 0.1 * mmd_loss # Weighting as per common practice
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            if step % 100 == 0:
                print(f"Step {step}: Loss {total_loss.item():.4f} (Diff: {diff_loss.item():.4f}, MMD: {mmd_loss.item():.4f})")
            
            step += 1
            if step >= train_config['steps']:
                break
                
    # Save model
    torch.save(model.state_dict(), "zuna_model.pt")

if __name__ == "__main__":
    # Example config
    m_config = {
        'd_model': 1024,
        'n_layers': 16,
        'n_heads': 16,
        'window_size': 32
    }
    t_config = {
        'lr': 1e-4,
        'batch_size': 8,
        'steps': 150000,
        'dataset_params': {
            'data_paths': ['dummy_path'],
            'sfreq': 256.0
        }
    }
    # train(m_config, t_config)
