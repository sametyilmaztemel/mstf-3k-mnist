"""H100/A100 Training Pipeline for tinycml-mnist3k project — v2 (augmentation bug fix + KD + TTA + tinycml parity).

Strateji: WST(25-100) pre-compute + MLP — full 60K MNIST + heavy augmentation + Knowledge Distillation.

v2 Yenilikler (v1 üstüne):
- Augmentation bug fix: feature-space Gaussian noise + MixUp + CutMix
- Knowledge Distillation: teacher CNN → student 3K-MLP
- Test Time Augmentation (TTA): 10× forward average
- tinycml binary weight dump (cml_ser format)
- tinycml C inference parity check (compile + run + compare)
- Detaylı raporlama: her mimari sonunda summary, JSON dump, log file

Pipeline:
1. WST pre-compute (full 60K + 10K test) — bir kere, cache'le
2. Mimari search: 5-10 farklı mimari × 100 epoch
3. En iyi mimariye KD + TTA + EMA + full heavy aug
4. tinycml dump + C parity

Modlar:
- BASELINE: 5K sample hızlı test (~5 dk)
- FULL: 60K + 100 epoch + tüm trick'ler (A100'de ~2-4 saat)
"""

import argparse
import json
import os
import time
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from kymatio.torch import Scattering2D

CACHE_DIR = "/home/samet/projects/tinycml-mnist3k/cache"
DATA_PATH = "/home/samet/projects/tinycml-mnist3k/mnist.npz"
LOG_DIR = "/home/samet/projects/tinycml-mnist3k/logs"
WEIGHTS_DIR = "/home/samet/projects/tinycml-mnist3k/weights"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === POOLING ===
def pool_grid(out):
    """(N, K, 7, 7) → (N, K*4). 2x2 grid over spatial."""
    tl = out[:, :, :3, :3].mean(axis=(2,3))
    tr = out[:, :, :3, 4:].mean(axis=(2,3))
    bl = out[:, :, 4:, :3].mean(axis=(2,3))
    br = out[:, :, 4:, 4:].mean(axis=(2,3))
    return np.stack([tl, tr, bl, br], axis=-1).reshape(out.shape[0], -1)


def pool_gap(out):
    """(N, K, 7, 7) → (N, K)."""
    return out.mean(axis=(2, 3))


def pool_meanchan(out):
    """(N, K, 7, 7) → (N, 49). Mean over channels."""
    return out.mean(axis=1).reshape(out.shape[0], -1)


def pool_meanstd(out):
    """(N, K, 7, 7) → (N, 2K). Mean + Std concatenated."""
    return np.concatenate([out.mean(axis=(2,3)), out.std(axis=(2,3))], axis=1)


POOLS = {
    "grid": pool_grid,
    "gap": pool_gap,
    "meanchan": pool_meanchan,
    "meanstd": pool_meanstd,
}


# === WST pre-compute ===
def precompute_wst(x_train, x_test, J=2, L=4, batch=512):
    """Returns: dict[pool_name] -> (X_train, X_test)."""
    cache_key = f"J{J}L{L}"
    scat_train_path = os.path.join(CACHE_DIR, f"wst_{cache_key}_train.npy")
    scat_test_path = os.path.join(CACHE_DIR, f"wst_{cache_key}_test.npy")

    if os.path.exists(scat_train_path) and os.path.exists(scat_test_path):
        print(f"  Loading WST from cache...")
        out_train = np.load(scat_train_path)
        out_test = np.load(scat_test_path)
    else:
        print(f"  Computing WST J={J}, L={L} (CPU/GPU mode)...")
        s = Scattering2D(J=J, L=L, shape=(28,28)).to(DEVICE)
        out_train = np.zeros((len(x_train), s(torch.zeros(1,1,28,28).to(DEVICE)).shape[1], 7, 7), dtype=np.float32)
        out_test = np.zeros((len(x_test), out_train.shape[1], 7, 7), dtype=np.float32)
        with torch.no_grad():
            t0 = time.time()
            for i in range(0, len(x_train), batch):
                xb = torch.from_numpy(x_train[i:i+batch]).unsqueeze(1).to(DEVICE)
                out_train[i:i+batch] = s(xb).cpu().numpy()
            print(f"    train ({len(x_train)}): {time.time()-t0:.1f}s")
            t0 = time.time()
            for i in range(0, len(x_test), batch):
                xb = torch.from_numpy(x_test[i:i+batch]).unsqueeze(1).to(DEVICE)
                out_test[i:i+batch] = s(xb).cpu().numpy()
            print(f"    test ({len(x_test)}): {time.time()-t0:.1f}s")
        np.save(scat_train_path, out_train)
        np.save(scat_test_path, out_test)

    print(f"  WST shape: train={out_train.shape}, test={out_test.shape}")
    pools = {}
    for name, fn in POOLS.items():
        X_tr = fn(out_train)
        X_te = fn(out_test)
        mu = X_tr.mean(0)
        std = X_tr.std(0) + 1e-8
        X_tr = (X_tr - mu) / std
        X_te = (X_te - mu) / std
        pools[name] = (X_tr.astype(np.float32), X_te.astype(np.float32))
        print(f"    pool={name}: train={X_tr.shape}, test={X_te.shape}, feat_dim={X_tr.shape[1]}")
    return pools


# === MODEL ===
def make_mlp(in_dim, hidden_arch, n_classes=10, dropout=0.05):
    """hidden_arch: e.g. (64, 16) for 25->64->16->10."""
    sizes = [in_dim, *hidden_arch, n_classes]
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes) - 2:
            layers.append(nn.BatchNorm1d(sizes[i+1]))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def count_linear_params(model):
    """Sum Linear W + b. (No BN affine counted.)"""
    return sum(m.weight.numel() + m.bias.numel()
               for m in model if isinstance(m, nn.Linear))


# === AUGMENTATION (v2 — feature-space + image-space option) ===
def augment_features(x, noise_std=0.05, p_noise=0.5):
    """Feature-space augmentation: Gaussian noise with prob p_noise.

    v1'de image-space augment_batch tanımlıydı ama HİÇ çağrılmıyordu.
    WST features zaten translation-invariant, bu yüzden feature-space
    augmentation daha etkili (mixup zaten var).
    """
    if np.random.rand() < p_noise:
        return x + torch.randn_like(x) * noise_std
    return x


# === MIXUP ===
def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[perm]
    return x_mix, y, y[perm], lam


# === EMA ===
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def apply_to(self, model):
        model.load_state_dict(self.shadow)


# === KNOWLEDGE DISTILLATION ===
class TeacherCNN(nn.Module):
    """Büyük CNN teacher — student'ı distill edecek.
    Param count: ~50K (3K üstünde, sadece teacher)."""

    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)  # 28->14
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # 14->7
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: (B, 28, 28) raw image
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))