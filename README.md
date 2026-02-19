# ZUNA: Flexible EEG Super-resolution

This repository contains an implementation of **ZUNA**, a 380M-parameter masked diffusion autoencoder for EEG channel infilling and super-resolution, as described in the paper:

> **ZUNA: Flexible EEG Superresolution with Position-Aware Diffusion Autoencoders**
> *Christopher Warner, Jonas Mago, JR Huml, Mohamed Osman, Beren Millidge*

## Overview

ZUNA is a foundation model designed to perform masked channel infilling and super-resolution for arbitrary electrode numbers and positions. It tokenizes multichannel EEG into short temporal windows and injects spatiotemporal structure via a **4D Rotary Positional Encoding (RoPE)** over $(x, y, z, t)$.

### Key Features

- **Flexible Montage**: Generalizes to arbitrary channel subsets and positions.
- **4D RoPE**: Handles spatial (3D electrode coordinates) and temporal (window index) structure.
- **Diffusion Autoencoder**: Uses **Rectified Flow** for stable and efficient generative reconstruction.
- **Efficient Scaling**: 380M parameters, optimized for performance on consumer GPUs.
- **Robust Training**: Trained with heavy channel dropout (up to 90%) to ensure high reconstruction fidelity.

## Architecture

ZUNA follows a transformer-based encoder-decoder architecture:

1.  **Rasterized Tokenization**: EEG signals are segmented into 0.125s windows and processed in a raster-scan order (time-first).
2.  **Learnable Registers**: Encoder employs interleaved register tokens to improve latent representation learning.
3.  **Cross-Attention Decoder**: The decoder takes noised tokens and conditions on the encoder's latent representation and diffusion time-step.
4.  **AdaRMSNorm**: Adaptive RMS normalization is used in the decoder for time-step conditioning.

## Project Structure

```text
zuna/
├── __init__.py
├── data_utils.py    # MNE-compatible preprocessing and coordinates
├── dataset.py       # EEG Dataset with heavy channel dropout (80/20 split)
├── rope_4d.py       # 4D Rotary Positional Encoding (x, y, z, t)
├── tokenizer.py     # Rasterized channel-time tokenizer/detokenizer
├── layers.py        # RMSNorm, AdaRMSNorm, Attention, Cross-Attention
├── model.py         # Full Encoder-Decoder (ZUNAModel)
├── diffusion.py     # Rectified Flow (Velocity prediction)
└── train.py         # Training pipeline with MMD latent regularization
```

## Installation

```bash
# Clone the repository
git clone git@github.com:Shahip2016/ZUNA.git
cd ZUNA

# Install dependencies
pip install mne numpy torch scipy
```

## Methodology

### Channel Dropout
During training, ZUNA uses a heavy channel dropout scheme:
- 80% probability to drop between 1 and $C/2$ channels.
- 20% probability to drop between $C/2$ and $C-1$ channels.
- $C$ is the total number of channels.

### Rectified Flow
ZUNA implements Rectified Flow where the model learns to predict the "velocity" $v = x_1 - x_0$, enabling deterministic sampling from noise ($x_1$) to clean data ($x_0$) via Euler ODE solving.

## License

This project is licensed under the Apache 2.0 License.
