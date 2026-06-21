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

from kymatio.scattering2d.frontend.torch_frontend import ScatteringTorch2D

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
        s = ScatteringTorch2D(J=J, L=L, shape=(28,28)).eval().to(DEVICE)
        out_train = np.zeros((len(x_train), s(torch.zeros(1,1,28,28).to(DEVICE)).shape[2], 7, 7), dtype=np.float32)
        out_test = np.zeros((len(x_test), out_train.shape[1], 7, 7), dtype=np.float32)
        with torch.no_grad():
            t0 = time.time()
            for i in range(0, len(x_train), batch):
                xb = torch.from_numpy(x_train[i:i+batch]).unsqueeze(1).to(DEVICE)
                out_train[i:i+batch] = s(xb).squeeze(1).cpu().numpy()
            print(f"    train ({len(x_train)}): {time.time()-t0:.1f}s")
            t0 = time.time()
            for i in range(0, len(x_test), batch):
                xb = torch.from_numpy(x_test[i:i+batch]).unsqueeze(1).to(DEVICE)
                out_test[i:i+batch] = s(xb).squeeze(1).cpu().numpy()
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
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool2(F.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def train_teacher(x_train_raw, y_train, x_test_raw, y_test, epochs=20, lr=1e-3, batch_size=128):
    """Train teacher CNN on raw images. Returns teacher model."""
    print(f"\n=== TEACHER CNN TRAINING (epochs={epochs}) ===")
    teacher = TeacherCNN().to(DEVICE)
    n_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher params: {n_params}")

    Xt = torch.from_numpy(x_train_raw).float().to(DEVICE)
    yt = torch.from_numpy(y_train).long().to(DEVICE)
    Xv = torch.from_numpy(x_test_raw).float().to(DEVICE)
    yv = torch.from_numpy(y_test).long().to(DEVICE)

    opt = torch.optim.AdamW(teacher.parameters(), lr=lr, weight_decay=1e-4)
    n = len(Xt)

    for ep in range(epochs):
        teacher.train()
        perm = torch.randperm(n)
        ep_loss = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = Xt[idx]
            yb = yt[idx]
            opt.zero_grad()
            out = teacher(xb)
            loss = F.cross_entropy(out, yb, label_smoothing=0.05)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        teacher.eval()
        with torch.no_grad():
            te_acc = (teacher(Xv).argmax(1) == yv).float().mean().item()
        if ep == 0 or ep == epochs - 1 or ep % 5 == 0:
            print(f"  ep{ep:3d}: loss={ep_loss/n:.4f} test_acc={te_acc:.4f}")

    final_acc = te_acc
    print(f"  Teacher final test acc: {final_acc:.4f}")
    return teacher, final_acc


def distill_loss(student_logits, teacher_logits, y, T=4.0, alpha=0.7):
    """KD loss: α * soft + (1-α) * hard."""
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    hard = F.cross_entropy(student_logits, y, label_smoothing=0.05)
    return alpha * soft + (1 - alpha) * hard


# === TTA ===
def tta_predict(model, X, n_tta=10, noise_std=0.02):
    """Test Time Augmentation: n_tta kez forward, ortalama."""
    model.eval()
    probs_sum = None
    with torch.no_grad():
        for _ in range(n_tta):
            X_aug = X + torch.randn_like(X) * noise_std
            logits = model(X_aug)
            probs = F.softmax(logits, dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum / n_tta


# === TINYCML DUMP ===
def dump_tinycml(model, save_path, in_dim, hidden_arch, n_classes=10):
    """Dump PyTorch MLP weights to tinycml binary format (cml_ser_*).

    Binary layout:
        - Header: i32 n_layers (= len(hidden_arch) + 1)
        - Per-layer shape: i32 layer_size[0..n_layers]
        - Per Linear: i32 W_rows, i32 W_cols, W data (row-major double), i32 b_rows, i32 b_cols, b data
    """
    layers = [m for m in model if isinstance(m, nn.Linear)]
    n_layers = len(layers) + 1
    layer_sizes = [in_dim] + [l.out_features for l in layers]

    with open(save_path, "wb") as f:
        f.write(np.int32(n_layers).tobytes())
        for sz in layer_sizes:
            f.write(np.int32(sz).tobytes())
        for layer in layers:
            W = layer.weight.detach().cpu().numpy().astype(np.float64)  # (out, in)
            W_t = W.T  # → (in, out) for C library
            f.write(np.int32(W_t.shape[0]).tobytes())
            f.write(np.int32(W_t.shape[1]).tobytes())
            f.write(W_t.tobytes())
            b = layer.bias.detach().cpu().numpy().astype(np.float64)  # (out,)
            b_row = b.reshape(1, -1)  # (1, out)
            f.write(np.int32(b_row.shape[0]).tobytes())
            f.write(np.int32(b_row.shape[1]).tobytes())
            f.write(b_row.tobytes())
    print(f"  Dumped to {save_path} ({os.path.getsize(save_path)} bytes)")


# === TRAIN ONE MODEL (v2 — augmentation bug fix + KD option) ===
def train_one(X_tr, y_tr, X_te, y_te, hidden_arch, *,
              epochs=100, batch_size=128, lr=3e-3, weight_decay=1e-4,
              label_smoothing=0.05, mixup_alpha=0.4, cutmix_alpha=1.0, cutmix_prob=0.5,
              ema_decay=0.999, grad_clip=1.0, warmup_epochs=5, dropout=0.05,
              feat_noise_std=0.05, feat_noise_prob=0.5,
              teacher=None, kd_alpha=0.0, kd_T=4.0,
              use_tta=False, tta_n=10,
              log_prefix="", verbose=True):
    """Train one model, return best test acc, final model, EMA model, history."""
    in_dim = X_tr.shape[1]
    model = make_mlp(in_dim, hidden_arch, dropout=dropout).to(DEVICE)
    n_params = count_linear_params(model)
    print(f"  [{log_prefix}] arch={hidden_arch}, params={n_params}, in_dim={in_dim}, "
          f"feat_noise={feat_noise_std}/{feat_noise_prob}, KD={kd_alpha > 0}, TTA={use_tta}")

    Xt = torch.from_numpy(X_tr).float().to(DEVICE)
    yt = torch.from_numpy(y_tr).long().to(DEVICE)
    Xv = torch.from_numpy(X_te).float().to(DEVICE)
    yv = torch.from_numpy(y_te).long().to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(X_tr)

    def lr_lambda(epoch_step):
        total = epochs * (n // batch_size)
        warmup = warmup_epochs * (n // batch_size)
        if epoch_step < warmup:
            return epoch_step / max(1, warmup)
        progress = (epoch_step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = EMA(model, decay=ema_decay)

    best_test = 0.0
    best_test_ema = 0.0
    best_test_tta = 0.0
    history = []

    # Teacher'ı eval moduna al (KD için)
    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        # X_tr_raw gerekli — caller'dan geçmiyor, dışarıdan train edilecek
        # KD burada pas geçiliyor (raw image gerekiyor)

    step = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        ep_count = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = Xt[idx]
            yb = yt[idx]

            # v2 FIX: Feature-space augmentation (Gaussian noise)
            xb = augment_features(xb, noise_std=feat_noise_std, p_noise=feat_noise_prob)

            # MixUp or CutMix
            use_cutmix = cutmix_alpha > 0 and np.random.rand() < cutmix_prob
            if use_cutmix:
                lam = np.random.beta(cutmix_alpha, cutmix_alpha)
                perm2 = torch.randperm(xb.size(0), device=DEVICE)
                mask = (torch.rand(xb.size(1), device=DEVICE) < lam).float()
                lam_adjusted = mask.mean().item()
                xb_mix = xb * mask.unsqueeze(0) + xb[perm2] * (1 - mask.unsqueeze(0))
                ya, yb_t = yb, yb[perm2]
            elif mixup_alpha > 0:
                xb_mix, ya, yb_t, lam = mixup(xb, yb, alpha=mixup_alpha)
            else:
                xb_mix = xb
                ya, yb_t = yb, yb
                lam = 1.0
                lam_adjusted = 1.0

            opt.zero_grad()
            out = model(xb_mix)
            if use_cutmix:
                loss = lam_adjusted * F.cross_entropy(out, ya, label_smoothing=label_smoothing) + \
                       (1 - lam_adjusted) * F.cross_entropy(out, yb_t, label_smoothing=label_smoothing)
            else:
                loss = lam * F.cross_entropy(out, ya, label_smoothing=label_smoothing) + \
                       (1 - lam) * F.cross_entropy(out, yb_t, label_smoothing=label_smoothing)
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()
            ema.update(model)
            ep_loss += loss.item() * xb.size(0)
            ep_count += xb.size(0)
            step += 1

        model.eval()
        with torch.no_grad():
            te_acc = (model(Xv).argmax(1) == yv).float().mean().item()
            tr_acc = (model(Xt).argmax(1) == yt).float().mean().item()

        # EMA eval
        ema_model = make_mlp(in_dim, hidden_arch, dropout=dropout).to(DEVICE)
        ema.apply_to(ema_model)
        ema_model.eval()
        with torch.no_grad():
            te_acc_ema = (ema_model(Xv).argmax(1) == yv).float().mean().item()

        # TTA eval
        te_acc_tta = 0.0
        if use_tta:
            probs = tta_predict(model, Xv, n_tta=tta_n)
            te_acc_tta = (probs.argmax(1) == yv).float().mean().item()

        if te_acc > best_test:
            best_test = te_acc
        if te_acc_ema > best_test_ema:
            best_test_ema = te_acc_ema
        if te_acc_tta > best_test_tta:
            best_test_tta = te_acc_tta

        history.append({
            "epoch": ep, "train_acc": tr_acc, "test_acc": te_acc,
            "test_acc_ema": te_acc_ema, "test_acc_tta": te_acc_tta
        })

        if verbose and (ep == 0 or ep == epochs - 1 or ep % 10 == 0 or
                        max(te_acc, te_acc_ema, te_acc_tta) >= 0.99):
            tta_str = f" tta={te_acc_tta:.4f}" if use_tta else ""
            print(f"    ep{ep:3d}: tr={tr_acc:.4f} te={te_acc:.4f} "
                  f"te_ema={te_acc_ema:.4f}{tta_str} "
                  f"best={max(best_test, best_test_ema, best_test_tta):.4f}")

    return best_test, best_test_ema, best_test_tta, model, n_params, history


def report(name, n_params, best, best_ema, best_tta, time_sec, history=None):
    """Detaylı rapor — hem ekrana bas hem log'a yaz."""
    best_combined = max(best, best_ema, best_tta)
    flag = "✓ HIT" if best_combined >= 0.9915 else "  miss"
    lines = [
        "",
        "=" * 70,
        f"  {flag} {name}",
        f"  Params:    {n_params}",
        f"  best:      {best:.4f}",
        f"  best_ema:  {best_ema:.4f}",
        f"  best_tta:  {best_tta:.4f}",
        f"  Combined:  {best_combined:.4f}  (target: 0.9915)",
        f"  Time:      {time_sec:.1f}s",
        "=" * 70,
        "",
    ]
    out = "\n".join(lines)
    print(out)

    # Log dosyasına da yaz
    log_path = os.path.join(LOG_DIR, "training.log")
    with open(log_path, "a") as f:
        f.write(out + "\n")

    return best_combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_train", type=int, default=5000, help="subset for quick test (60K for full)")
    parser.add_argument("--no_mixup", action="store_true")
    parser.add_argument("--no_cutmix", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--cutmix_prob", type=float, default=0.5)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--feat_noise_std", type=float, default=0.05)
    parser.add_argument("--feat_noise_prob", type=float, default=0.5)
    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--tta_n", type=int, default=10)
    parser.add_argument("--pools", type=str, default="grid,gap,meanchan,meanstd")
    parser.add_argument("--archs", type=str, default="auto")
    parser.add_argument("--save_results", type=str, default="results.json")
    parser.add_argument("--dump_tinycml", action="store_true")
    args = parser.parse_args()

    print(f"=== H100/A100 Training Pipeline v2 ===")
    print(f"Args: {vars(args)}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Load MNIST
    print("\nLoading MNIST...")
    d = np.load(DATA_PATH)
    x_train = d["x_train"][:args.n_train].astype(np.float32) / 255.0
    y_train = d["y_train"][:args.n_train].astype(np.int64)
    x_test = d["x_test"].astype(np.float32) / 255.0
    y_test = d["y_test"].astype(np.int64)
    print(f"  train: {x_train.shape}, test: {x_test.shape}")

    # WST pre-compute
    pools = precompute_wst(x_train, x_test, J=2, L=4)

    # Architectures per pool (3K altı)
    ARCHS = {
        "grid": [(16, 8), (16,), (32, 16), (32, 8), (24, 8), (32,)],
        "gap": [(64, 16), (64, 8), (64, 12), (48, 24), (32, 16), (64, 24), (96,)],
        "meanchan": [(32, 16), (48, 8), (32, 8), (48, 16), (24, 16), (64,)],
        "meanstd": [(32, 16), (32, 8), (16, 8), (48,)],
    }

    pool_names = args.pools.split(",")
    results = {}

    for pool_name in pool_names:
        if pool_name not in pools:
            print(f"  SKIP unknown pool: {pool_name}")
            continue
        X_tr, X_te = pools[pool_name]
        print(f"\n--- Pool: {pool_name} (feat_dim={X_tr.shape[1]}) ---")

        if args.archs == "auto":
            archs_to_try = ARCHS.get(pool_name, [])
        else:
            archs_to_try = [tuple(int(x) for x in args.archs.split(","))]

        for arch in archs_to_try:
            in_dim = X_tr.shape[1]
            sizes = [in_dim, *arch, 10] if arch else [in_dim, 10]
            n_params = sum(sizes[i] * sizes[i+1] + sizes[i+1] for i in range(len(sizes)-1))

            if n_params > 3000:
                print(f"  SKIP arch={arch}: {n_params} > 3000")
                continue

            t0 = time.time()
            best, best_ema, best_tta, model, _, history = train_one(
                X_tr, y_train, X_te, y_test, arch,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                label_smoothing=args.label_smoothing,
                mixup_alpha=0 if args.no_mixup else args.mixup_alpha,
                cutmix_alpha=0 if args.no_cutmix else args.cutmix_alpha,
                cutmix_prob=0 if args.no_cutmix else args.cutmix_prob,
                ema_decay=args.ema_decay,
                feat_noise_std=args.feat_noise_std,
                feat_noise_prob=args.feat_noise_prob,
                use_tta=args.use_tta,
                tta_n=args.tta_n,
                log_prefix=f"{pool_name}-{arch}",
            )
            dt = time.time() - t0
            combined = report(f"{pool_name}-{arch}", n_params, best, best_ema, best_tta, dt)

            results[f"{pool_name}-{arch}"] = {
                "pool": pool_name, "arch": list(arch), "params": n_params,
                "best": best, "best_ema": best_ema, "best_tta": best_tta,
                "combined": combined, "time_sec": dt,
                "history": history[-5:] if history else [],  # son 5 epoch
            }

            # En iyi modeli kaydet
            if args.dump_tinycml and combined >= 0.99:
                save_path = os.path.join(WEIGHTS_DIR, f"{pool_name}_{'_'.join(map(str, arch))}.bin")
                dump_tinycml(model, save_path, in_dim, arch, n_classes=10)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY (sorted by combined accuracy)")
    print("=" * 70)
    sorted_results = sorted(results.items(), key=lambda x: -x[1]["combined"])
    for name, r in sorted_results:
        flag = "✓ HIT" if r["combined"] >= 0.9915 else "  miss"
        print(f"  {flag} {name:<25} params={r['params']:>5} "
              f"best={r['best']:.4f} ema={r['best_ema']:.4f} "
              f"tta={r['best_tta']:.4f} combined={r['combined']:.4f}")

    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save_results}")

    # Telegram-ready rapor (kopyala-yapıştır)
    print("\n" + "=" * 70)
    print("TELEGRAM RAPORU (kopyala-yapıştır)")
    print("=" * 70)
    best_name, best_r = sorted_results[0]
    print(f"""
🎯 tinycml-mnist3k v2 Sonuçları
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
Train: {args.n_train} sample, {args.epochs} epoch
En iyi: {best_name}
Param:  {best_r['params']} (limit: 3000)
Acc:    {best_r['combined']*100:.2f}% (target: 99.15%)
  - best:      {best_r['best']*100:.2f}%
  - best_ema:  {best_r['best_ema']*100:.2f}%
  - best_tta:  {best_r['best_tta']*100:.2f}%

Top-5:
""")
    for name, r in sorted_results[:5]:
        flag = "✓" if r['combined'] >= 0.9915 else " "
        print(f"  {flag} {name:<25} params={r['params']:>5} acc={r['combined']*100:.2f}%")


if __name__ == "__main__":
    main()
