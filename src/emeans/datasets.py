"""Benchmark datasets for e-means and e-means++ evaluation.

Implements:
1. All 7 synthetic distributions from Appendix D.2 of arXiv:2611.09501 (each calibrated
   to unit average coordinate variance):
     - `uniform_cube`: U(-sqrt(3), sqrt(3))^d
     - `isotropic_gaussian`: N(0, I_d)
     - `stretched_gaussian` (a in {0.5, 1.0, 2.0}): lambda_i = d * i^{-a} / sum_j j^{-a}
     - `balanced_gmm` (G = k): mu_g ~ N(0, (2/3) I_d), x = mu_g + sqrt(1/3) z
     - `zipf_gmm` (G = k): same centers, cluster weights p_g proportional to 1 / g
2. `multiscale_rare_outliers`: Multi-scale density + separated rare outlier clusters
   (0 < p_r < 0.1 / k) specifically testing rare-mode discovery & Huygens protection.
3. Real deep visual feature datasets extracted locally from CIFAR-10 (`/home/catid/fffdistill/data/cifar10`)
   using cached ImageNet-pretrained ResNet-18 (`/home/catid/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`):
     - `cifar10_resnet18` (d = 512 global semantic embeddings)
     - `cifar10_resnet18_patches` (d = 128 local convolutional patch descriptors, ~200K train vectors)
   with strict image-level split isolation AND exact cross-split vector deduplication.
"""

from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CIFAR10_CANDIDATES = (
    Path("/home/catid/fffdistill/data/cifar10/cifar-10-batches-py"),
    Path("/home/catid/emeans_env/data/cifar10/cifar-10-batches-py"),
)
DEFAULT_CIFAR10_ROOT = DEFAULT_CIFAR10_CANDIDATES[0]
DEFAULT_RESNET18_CANDIDATES = (
    Path("/home/catid/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"),
    Path("/home/catid/emeans_env/resnet18-f37072fd.pth"),
)
DEFAULT_RESNET18_CKPT = DEFAULT_RESNET18_CANDIDATES[0]
DEFAULT_CACHE_DIR = _REPO_ROOT / "outputs" / "phase1" / "cache"


@dataclass
class DatasetSplit:
    """Strictly disjoint train / val / test split for clustering evaluation."""

    name: str
    train: np.ndarray          # (N_train, D) float32
    val: np.ndarray            # (N_val, D) float32
    test: np.ndarray           # (N_test, D) float32
    d: int
    k_recommended: int
    true_centers: np.ndarray | None = None
    true_probs: np.ndarray | None = None
    rare_mode_indices: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def deduplicate_splits(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deduplicate within each split and remove any val/test rows appearing in earlier splits.

    Enforces the contamination-free holdout protocol of Section 4.1 / Appendix A of arXiv:2611.09501.
    """
    train = np.ascontiguousarray(train, dtype=np.float32)
    val = np.ascontiguousarray(val, dtype=np.float32)
    test = np.ascontiguousarray(test, dtype=np.float32)

    def _row_keys(arr: np.ndarray) -> np.ndarray:
        rounded = np.round(arr, decimals=decimals)
        # Normalize -0.0 to +0.0 so byte views match identically
        rounded = np.where(rounded == 0.0, 0.0, rounded).astype(np.float32, copy=False)
        return np.ascontiguousarray(rounded).view(
            np.dtype((np.void, rounded.shape[1] * rounded.dtype.itemsize))
        ).ravel()

    k_tr = _row_keys(train)
    _, uniq_tr_idx = np.unique(k_tr, return_index=True)
    uniq_tr_idx.sort()
    train_clean = np.ascontiguousarray(train[uniq_tr_idx])
    seen_tr = k_tr[uniq_tr_idx]

    k_va = _row_keys(val)
    _, uniq_va_idx = np.unique(k_va, return_index=True)
    uniq_va_idx.sort()
    va_cand = uniq_va_idx[~np.isin(k_va[uniq_va_idx], seen_tr)]
    val_clean = np.ascontiguousarray(val[va_cand])
    seen_tr_va = np.union1d(seen_tr, k_va[va_cand])

    k_te = _row_keys(test)
    _, uniq_te_idx = np.unique(k_te, return_index=True)
    uniq_te_idx.sort()
    te_cand = uniq_te_idx[~np.isin(k_te[uniq_te_idx], seen_tr_va)]
    test_clean = np.ascontiguousarray(test[te_cand])

    return train_clean, val_clean, test_clean


def make_synthetic_dataset(
    kind: str,
    n_train: int = 100_000,
    n_val: int = 10_000,
    n_test: int = 10_000,
    d: int = 128,
    k: int = 1024,
    seed: int = 42,
    stretch_a: float = 1.0,
    zipf_s: float = 1.0,
) -> DatasetSplit:
    """Generate synthetic datasets from Appendix D.2 of arXiv:2611.09501 + rare-outlier benchmark.

    Supported `kind`:
      - 'uniform_cube'
      - 'isotropic_gaussian'
      - 'stretched_gaussian_0.5', 'stretched_gaussian_1', 'stretched_gaussian_2', or 'stretched_gaussian'
      - 'balanced_gmm'
      - 'zipf_gmm'
      - 'multiscale_rare_outliers'
    """
    rng = np.random.default_rng(seed)
    n_total = int(n_train + n_val + n_test)

    true_centers: np.ndarray | None = None
    true_probs: np.ndarray | None = None
    rare_idx: np.ndarray | None = None
    meta: dict[str, Any] = {"kind": kind, "seed": seed}

    if kind.startswith("stretched_gaussian_"):
        stretch_a = float(kind.split("_")[-1])
        base_kind = "stretched_gaussian"
    else:
        base_kind = kind

    if base_kind == "uniform_cube":
        bound = math.sqrt(3.0)
        data = rng.uniform(-bound, bound, size=(n_total, d)).astype(np.float32)

    elif base_kind == "isotropic_gaussian":
        data = rng.standard_normal(size=(n_total, d), dtype=np.float32)

    elif base_kind == "stretched_gaussian":
        idx_1based = np.arange(1, d + 1, dtype=np.float64)
        raw_var = np.power(idx_1based, -float(stretch_a))
        variances = (float(d) * raw_var / raw_var.sum()).astype(np.float32)
        stds = np.sqrt(variances)[None, :]
        data = rng.standard_normal(size=(n_total, d), dtype=np.float32) * stds
        meta["stretch_a"] = float(stretch_a)

    elif base_kind in ("balanced_gmm", "zipf_gmm"):
        G = int(k)
        sigma_mu = math.sqrt(2.0 / 3.0)
        sigma_w = math.sqrt(1.0 / 3.0)
        centers = (rng.standard_normal(size=(G, d)) * sigma_mu).astype(np.float32)
        if base_kind == "balanced_gmm":
            probs = np.full((G,), 1.0 / G, dtype=np.float64)
        else:
            ranks = np.arange(1, G + 1, dtype=np.float64)
            probs = np.power(ranks, -float(zipf_s))
            probs /= probs.sum()
            meta["zipf_s"] = float(zipf_s)

        comp_ids = rng.choice(G, size=n_total, replace=True, p=probs)
        noise = rng.standard_normal(size=(n_total, d), dtype=np.float32) * sigma_w
        data = centers[comp_ids] + noise
        true_centers = centers
        true_probs = probs.astype(np.float32)

    elif base_kind == "multiscale_rare_outliers":
        # 90% bulk clusters hold 98.5% of mass; 10% isolated rare clusters hold 1.5% total mass,
        # so each rare cluster has p_r = 0.015 / (0.1 * k) = 0.15 / k -> or 0.004 / (0.1 * k) = 0.04 / k
        # strictly below tau = 0.1 * (1 / k), directly testing Huygens protection vs revival assassination!
        n_rare = max(4, k // 10)
        n_bulk = max(1, k - n_rare)

        sigma_mu_bulk = math.sqrt(2.0 / 3.0)
        sigma_w_bulk = 0.35
        bulk_centers = (rng.standard_normal(size=(n_bulk, d)) * sigma_mu_bulk).astype(np.float32)

        # Place rare outlier modes at 3.5x radius so they are unmistakably distinct modes
        raw_dirs = rng.standard_normal(size=(n_rare, d))
        norms = np.linalg.norm(raw_dirs, axis=1, keepdims=True) + 1e-12
        rare_radius = 3.5 * math.sqrt(float(d) * (2.0 / 3.0))
        rare_centers = (raw_dirs / norms * rare_radius).astype(np.float32)
        sigma_w_rare = 0.15

        centers = np.concatenate([bulk_centers, rare_centers], axis=0)
        # Rare total share = 0.03 * (n_rare / k) so each rare mode has p_r = 0.03 / k < 0.1 / k!
        # At n_train = 100,000 and k = 256, 0.03 / 256 * 100,000 ~ 12 points per rare cluster!
        p_each_rare = 0.03 / float(k)
        rare_total = min(0.2, p_each_rare * n_rare)
        bulk_probs = np.full((n_bulk,), (1.0 - rare_total) / n_bulk, dtype=np.float64)
        rare_probs = np.full((n_rare,), rare_total / n_rare, dtype=np.float64)
        probs = np.concatenate([bulk_probs, rare_probs])
        probs /= probs.sum()

        comp_ids = rng.choice(k, size=n_total, replace=True, p=probs)
        is_rare_pt = comp_ids >= n_bulk
        scales = np.where(is_rare_pt, sigma_w_rare, sigma_w_bulk).astype(np.float32)[:, None]
        noise = rng.standard_normal(size=(n_total, d), dtype=np.float32) * scales
        data = centers[comp_ids] + noise

        true_centers = centers
        true_probs = probs.astype(np.float32)
        rare_idx = np.arange(n_bulk, k, dtype=np.int64)
        meta["n_bulk"] = n_bulk
        meta["n_rare"] = n_rare
        meta["p_each_rare_times_k"] = float(p_each_rare * k)
    else:
        raise ValueError(f"Unknown synthetic dataset kind={kind!r}")

    x_tr = np.ascontiguousarray(data[:n_train], dtype=np.float32)
    x_va = np.ascontiguousarray(data[n_train : n_train + n_val], dtype=np.float32)
    x_te = np.ascontiguousarray(data[n_train + n_val :], dtype=np.float32)

    return DatasetSplit(
        name=kind,
        train=x_tr,
        val=x_va,
        test=x_te,
        d=d,
        k_recommended=k,
        true_centers=true_centers,
        true_probs=true_probs,
        rare_mode_indices=rare_idx,
        metadata=meta,
    )


def _load_raw_cifar10(cifar_dir: Path = DEFAULT_CIFAR10_ROOT) -> tuple[np.ndarray, np.ndarray]:
    """Load raw CIFAR-10 uint8 images: returns (train_50k_3_32_32, test_10k_3_32_32)."""
    resolved_dir = cifar_dir
    if not resolved_dir.exists():
        for cand in DEFAULT_CIFAR10_CANDIDATES:
            if cand.exists():
                resolved_dir = cand
                break
    if not resolved_dir.exists():
        raise FileNotFoundError(f"CIFAR-10 directory not found at {cifar_dir}")

    train_blocks = []
    for i in range(1, 6):
        with open(resolved_dir / f"data_batch_{i}", "rb") as f:
            entry = pickle.load(f, encoding="bytes")
        train_blocks.append(entry[b"data"])
    x_train_raw = np.concatenate(train_blocks, axis=0).reshape(-1, 3, 32, 32)

    with open(resolved_dir / "test_batch", "rb") as f:
        test_entry = pickle.load(f, encoding="bytes")
    x_test_raw = test_entry[b"data"].reshape(-1, 3, 32, 32)
    return x_train_raw, x_test_raw


def _deduplicate_raw_images(
    train_imgs: np.ndarray,
    val_imgs: np.ndarray,
    test_imgs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deduplicate raw uint8 (N, 3, 32, 32) images BEFORE feature/patch extraction."""
    def _img_keys(imgs: np.ndarray) -> np.ndarray:
        flat = np.ascontiguousarray(imgs.reshape(imgs.shape[0], -1))
        return flat.view(np.dtype((np.void, flat.shape[1]))).ravel()

    k_tr = _img_keys(train_imgs)
    _, idx_tr = np.unique(k_tr, return_index=True)
    idx_tr.sort()
    tr_clean = train_imgs[idx_tr]
    seen_tr = k_tr[idx_tr]

    k_va = _img_keys(val_imgs)
    _, idx_va = np.unique(k_va, return_index=True)
    idx_va.sort()
    va_keep = idx_va[~np.isin(k_va[idx_va], seen_tr)]
    va_clean = val_imgs[va_keep]
    seen_tr_va = np.union1d(seen_tr, k_va[va_keep])

    k_te = _img_keys(test_imgs)
    _, idx_te = np.unique(k_te, return_index=True)
    idx_te.sort()
    te_keep = idx_te[~np.isin(k_te[idx_te], seen_tr_va)]
    te_clean = test_imgs[te_keep]

    return tr_clean, va_clean, te_clean


@torch.no_grad()
def load_cifar10_resnet18_embeddings(
    mode: str = "global_512",
    k_recommended: int | None = None,
    cifar_dir: Path = DEFAULT_CIFAR10_ROOT,
    ckpt_path: Path = DEFAULT_RESNET18_CKPT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    device: str | None = None,
) -> DatasetSplit:
    """Extract & cache real deep visual embeddings from CIFAR-10 using pretrained ResNet-18.

    Modes:
      - 'global_512': Penultimate avgpool 512-D embeddings (~50K train / ~5K val / ~5K test).
      - 'patches_128': Spatial `layer2` 128-D local visual descriptors on 64x64 upsampled images
        (4 spatial patches per image -> ~200K train / ~20K val / ~20K test, d=128), split strictly
        at the IMAGE level before extraction and deduplicated across splits!
    """
    if mode not in ("global_512", "patches_128"):
        raise ValueError("mode must be 'global_512' or 'patches_128'")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"cifar10_resnet18_{mode}.npz"

    if cache_file.exists():
        loaded = np.load(cache_file)
        tr = np.ascontiguousarray(loaded["train"], dtype=np.float32)
        va = np.ascontiguousarray(loaded["val"], dtype=np.float32)
        te = np.ascontiguousarray(loaded["test"], dtype=np.float32)
    else:
        import torchvision.models as models
        import torch.nn.functional as F

        tr_raw, te_full_raw = _load_raw_cifar10(cifar_dir)
        va_raw = te_full_raw[:5000]
        te_raw = te_full_raw[5000:]

        # Image-level deduplication BEFORE any feature/patch extraction
        tr_imgs, va_imgs, te_imgs = _deduplicate_raw_images(tr_raw, va_raw, te_raw)

        dev = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        net = models.resnet18(weights=None)
        resolved_ckpt = ckpt_path
        if not resolved_ckpt.exists():
            for cand_c in DEFAULT_RESNET18_CANDIDATES:
                if cand_c.exists():
                    resolved_ckpt = cand_c
                    break
        if not resolved_ckpt.exists():
            raise FileNotFoundError(f"Cached ResNet-18 checkpoint not found at {ckpt_path}")
        state = torch.load(resolved_ckpt, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.eval().to(dev)

        mean = torch.tensor([0.485, 0.456, 0.406], device=dev, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=dev, dtype=torch.float32).view(1, 3, 1, 1)

        def _extract(imgs: np.ndarray) -> np.ndarray:
            feats_list = []
            bsz = 512
            for s in range(0, imgs.shape[0], bsz):
                batch_u8 = torch.from_numpy(imgs[s : s + bsz]).to(dev, dtype=torch.float32) / 255.0
                if mode == "global_512":
                    inp = F.interpolate(batch_u8, size=(96, 96), mode="bilinear", align_corners=False)
                    inp = (inp - mean) / std
                    h = net.conv1(inp)
                    h = net.bn1(h)
                    h = net.relu(h)
                    h = net.maxpool(h)
                    h = net.layer1(h)
                    h = net.layer2(h)
                    h = net.layer3(h)
                    h = net.layer4(h)
                    h = net.avgpool(h).flatten(1)  # (B, 512)
                    feats_list.append(h.cpu().numpy().astype(np.float32))
                else:
                    # 64x64 -> conv1(/2=32) -> maxpool(/2=16) -> layer1(16) -> layer2(/2=8x8) -> avg_pool2d(2) -> 4 patches (2x2) of 128-D!
                    inp = F.interpolate(batch_u8, size=(64, 64), mode="bilinear", align_corners=False)
                    inp = (inp - mean) / std
                    h = net.conv1(inp)
                    h = net.bn1(h)
                    h = net.relu(h)
                    h = net.maxpool(h)
                    h = net.layer1(h)
                    h = net.layer2(h)  # (B, 128, 8, 8)
                    h_pooled = F.avg_pool2d(h, kernel_size=4, stride=4)  # (B, 128, 2, 2) = 4 non-overlapping quadrants!
                    patches = h_pooled.permute(0, 2, 3, 1).reshape(-1, 128)
                    feats_list.append(patches.cpu().numpy().astype(np.float32))
            return np.concatenate(feats_list, axis=0)

        tr_f = _extract(tr_imgs)
        va_f = _extract(va_imgs)
        te_f = _extract(te_imgs)

        # Vector-level exact deduplication across splits as a second safety barrier
        tr, va, te = deduplicate_splits(tr_f, va_f, te_f, decimals=5)
        tmp_file = cache_dir / f".tmp_{os.getpid()}_{mode}.npz"
        try:
            np.savez(tmp_file, train=tr, val=va, test=te)
            os.replace(tmp_file, cache_file)
        finally:
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)

    d = int(tr.shape[1])
    if k_recommended is None:
        k_recommended = 1024 if mode == "global_512" else 2048

    return DatasetSplit(
        name=f"cifar10_resnet18_{mode}",
        train=tr,
        val=va,
        test=te,
        d=d,
        k_recommended=int(k_recommended),
        metadata={"source": "CIFAR-10 + ImageNet ResNet-18", "mode": mode},
    )
