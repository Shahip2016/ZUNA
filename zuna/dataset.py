import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import mne
from .data_utils import normalize_epochs, get_3d_coordinates

class EEGDataset(Dataset):
    """
    Dataset for ZUNA EEG foundation model.
    Handles heavy channel dropout as described in the paper.
    """
    def __init__(self, 
                 data_paths: List[str], 
                 sfreq: float = 256.0, 
                 epoch_duration: float = 5.0,
                 dropout_prob: float = 0.9,
                 is_training: bool = True):
        self.data_paths = data_paths
        self.sfreq = sfreq
        self.epoch_duration = epoch_duration
        self.dropout_prob = dropout_prob
        self.is_training = is_training
        
        # In a real implementation, we would pre-load or index all sessions.
        # Here we simulate with one path for demonstration.
        self.epochs_data = [] # List of np.ndarray [channels, samples]
        self.channel_coords = [] # List of np.ndarray [channels, 3]
        
    def add_mne_raw(self, raw: mne.io.Raw):
        """Processes and adds an MNE Raw object to the dataset."""
        # Preprocessing should have been done via data_utils
        data = raw.get_data()
        coords = get_3d_coordinates(raw.info)
        
        # Segmenting manually for simplicity in this demo
        samples_per_epoch = int(self.sfreq * self.epoch_duration)
        n_epochs = data.shape[1] // samples_per_epoch
        
        for i in range(n_epochs):
            start = i * samples_per_epoch
            end = start + samples_per_epoch
            epoch_segment = data[:, start:end]
            
            # Normalize
            epoch_segment = normalize_epochs(epoch_segment)
            
            self.epochs_data.append(epoch_segment)
            self.channel_coords.append(coords)

    def __len__(self) -> int:
        return len(self.epochs_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_input: [C, T] EEG signal with dropout (zeros)
            y_target: [C, T] Original clean EEG signal
            coords: [C, 3] 3D spatial coordinates (never dropped)
        """
        x_clean = self.epochs_data[idx]
        coords = self.channel_coords[idx]
        C, T = x_clean.shape
        
        x_input = x_clean.copy()
        
        if self.is_training and np.random.rand() < self.dropout_prob:
            # Heavy channel dropout scheme
            # Randomly select between 1 and C/2 channels with 80% prob
            # Or between C/2 and C-1 channels with 20% prob
            if np.random.rand() < 0.8:
                n_drop = np.random.randint(1, max(2, C // 2 + 1))
            else:
                n_drop = np.random.randint(max(1, C // 2), max(2, C))
            
            drop_indices = np.random.choice(C, n_drop, replace=False)
            x_input[drop_indices, :] = 0.0
            
        return torch.from_numpy(x_input).float(), \
               torch.from_numpy(x_clean).float(), \
               torch.from_numpy(coords).float()

def collate_fn(batch):
    """
    Handles variable number of channels across samples using padding or packing.
    The paper uses 'packing with sample-masking using flex attention'.
    For this implementation, we'll return a list or pad to max channels in batch.
    """
    x_inputs, x_cleans, coords = zip(*batch)
    
    # Pad channels to the largest number of channels in the batch
    max_channels = max([x.shape[0] for x in x_inputs])
    T = x_inputs[0].shape[1]
    
    padded_inputs = []
    padded_cleans = []
    padded_coords = []
    masks = []
    
    for x_in, x_cl, co in zip(x_inputs, x_cleans, coords):
        C = x_in.shape[0]
        pad_size = max_channels - C
        
        # Pad with zeros
        padded_inputs.append(torch.cat([x_in, torch.zeros(pad_size, T)], dim=0))
        padded_cleans.append(torch.cat([x_cl, torch.zeros(pad_size, T)], dim=0))
        # Mask for valid channels
        mask = torch.ones(max_channels)
        mask[C:] = 0
        masks.append(mask)
        
        # Pad coordinates with zeros (not ideal for real training, but okay for structure)
        padded_coords.append(torch.cat([co, torch.zeros(pad_size, 3)], dim=0))
        
    return torch.stack(padded_inputs), \
           torch.stack(padded_cleans), \
           torch.stack(padded_coords), \
           torch.stack(masks)
