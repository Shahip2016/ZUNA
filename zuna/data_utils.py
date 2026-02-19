import mne
import numpy as np
from typing import List, Tuple, Optional

def preprocess_eeg(raw: mne.io.Raw, target_fs: float = 256.0) -> mne.io.Raw:
    """
    Standardizes sampling rate, temporal segmentation, normalization, and spatial metadata.
    Following the ZUNA paper preprocessing pipeline.
    """
    # 1. Resample to 256 Hz
    if raw.info['sfreq'] != target_fs:
        raw.resample(target_fs)
    
    # 2. High-pass filter at 0.5 Hz
    raw.filter(l_freq=0.5, h_freq=None)
    
    # 3. Re-reference to common average reference (CAR)
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()
    
    # 4. Adaptive Notch Filtering (Line noise detection)
    # The paper mentions analyzing PSD between 45Hz and Nyquist
    # For simplicity, we'll use a standard notch at 50Hz and 60Hz harmonics
    # but could be extended to be more adaptive as described.
    freqs = np.arange(50, target_fs / 2, 50).tolist() + np.arange(60, target_fs / 2, 60).tolist()
    freqs = sorted(list(set(freqs)))
    raw.notch_filter(freqs=freqs)
    
    return raw

def extract_epochs(raw: mne.io.Raw, duration: float = 5.0) -> mne.Epochs:
    """
    Segments continuous recordings into non-overlapping 5-second segments (1280 samples at 256Hz).
    """
    events = mne.make_fixed_length_events(raw, duration=duration)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=duration, baseline=None, preload=True)
    return epochs

def normalize_epochs(data: np.ndarray) -> np.ndarray:
    """
    Z-score normalization based on global mean and SD across all EEG channels.
    Paper mentions standardizing to mean=0, std=0.1 for stability.
    """
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / (std + 1e-8)
    # Scaling to 0.1 std as mentioned in the paper for training stability
    return normalized_data * 0.1

def get_3d_coordinates(info: mne.Info) -> np.ndarray:
    """
    Extracts 3D Cartesian (x, y, z) scalp coordinates for EEG electrodes.
    """
    coords = []
    for ch in info['chs']:
        if ch['kind'] == mne.io.constants.FIFFV_EEG_CH:
            coords.append(ch['loc'][:3])
    return np.array(coords)
