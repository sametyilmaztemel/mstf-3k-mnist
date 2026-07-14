# tinycml-mnist3k — 3K-Param MNIST %99.15+ Benchmark

**Samet Yılmaz Temel — tinyrlabs/tinycml**

3,000 parametre altinda, MNIST test setinde **%99.15+** accuracy hedefi.

## Pipeline

1. **WST (Wavelet Scattering Transform)** pre-compute (0 learnable param) — Bruna & Mallat 2012
2. **Kucuk MLP** (3K altinda) heavy augmentation ile egitit
3. **Mimari search:** 4 pooling stratejisi × auto mimari = ~20 varyant
4. **En iyi mimariye EMA + TTA + Knowledge Distillation**
5. **tinycml C binary dump + C inference parity check**

## Mimari Varyantlari (3K altinda)

| Pool     | Feature dim | Mimari           | Param   |
|----------|-------------|------------------|---------|
| gap      | 25          | (64, 16)         | 2,874   |
| gap      | 25          | (48, 24)         | 2,674   |
| gap      | 25          | (96,)            | 2,506   |
| meanstd  | 50          | (32, 16)         | 2,170   |
| grid     | 100         | (16, 8)          | 1,842   |
| grid     | 100         | (24, 8)          | 2,714   |
| meanchan | 49          | (32, 16)         | 2,298   |

## Trick Kombinasyonu

- Feature-space Gaussian noise (augmentation bug fix)
- MixUp α=0.4
- CutMix α=1.0, prob=0.5
- Label smoothing 0.05
- AdamW + CosineAnnealing + 5-epoch warmup
- EMA decay=0.999
- TTA (10-20 augmentation average)
- Gradient clipping 1.0

## Kullanım

### Lokal CPU (hizli test)
```bash
python3 train_h100.py --n_train 5000 --epochs 5 --pools gap --archs 64,16
```

### Colab Pro+ (A100/H100)
1. https://colab.research.google.com/github/sametyilmaztemel/mstf-3k-mnist/blob/main/mstf_3k_search.ipynb
2. Runtime → Change runtime type → GPU (A100 veya H100)
3. Run all (~2-4 saat)

### tinycml C inference
En iyi model agirliklari `weights/` altinda `.bin` formatinda.
`mstf_3k_search.ipynb` Cell 8 ile C parity check yapilir.

## Baglantili Repolar

- **tinycml:** https://github.com/tinyrlabs/tinycml (zero-dep C11 ML library)
- **MSTF paper:** https://github.com/sametyilmaztemel/mstf-paper (ana yayin)

## Referanslar

- Bruna & Mallat 2012 — "Invariant Scattering Convolution Networks" (WST)
- Hinton 2015 — "Distilling the Knowledge in a Neural Network" (KD)
- Kymatio: https://www.kymat.io/ (PyTorch scattering)
