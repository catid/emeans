"""High-performance nearest-centroid assignment and accumulation kernels.

Provides:
1. Fused Triton GPU nearest-centroid kernel (`triton_assign_chunked`) that streams
   centroid tiles in registers without materializing the (B x K) distance matrix,
   preserving exact represented-norm consistency (Appendix I.1 of arXiv:2611.09501).
2. Chunked PyTorch GPU/CPU fallback (`torch_assign_chunked`) and Faiss CPU assignment.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


if HAS_TRITON:

    @triton.jit
    def _fused_assign_kernel(
        x_ptr,
        c_ptr,
        c_norm_ptr,
        out_dist_ptr,
        out_label_ptr,
        B,
        K,
        D,
        stride_xb,
        stride_xd,
        stride_ck,
        stride_cd,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        USE_BF16: tl.constexpr,
        USE_FP16: tl.constexpr,
    ):
        pid_b = tl.program_id(0).to(tl.int64)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B).to(tl.int64)
        mask_b = offs_b < B

        # Precompute ||x||^2 in FP32
        x_norm = tl.zeros((BLOCK_B,), dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
            mask_d = offs_d < D
            x_tile = tl.load(
                x_ptr + offs_b[:, None] * stride_xb + offs_d[None, :] * stride_xd,
                mask=mask_b[:, None] & mask_d[None, :],
                other=0.0,
            )
            if USE_BF16:
                x_tile = x_tile.to(tl.bfloat16)
            elif USE_FP16:
                x_tile = x_tile.to(tl.float16)
            xf = x_tile.to(tl.float32)
            x_norm += tl.sum(xf * xf, axis=1)

        best_score = tl.full((BLOCK_B,), float("inf"), dtype=tl.float32)
        best_label = tl.zeros((BLOCK_B,), dtype=tl.int32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K).to(tl.int64)
            mask_k = offs_k < K
            dots = tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32)

            for d0 in range(0, D, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D).to(tl.int64)
                mask_d = offs_d < D
                x_tile = tl.load(
                    x_ptr + offs_b[:, None] * stride_xb + offs_d[None, :] * stride_xd,
                    mask=mask_b[:, None] & mask_d[None, :],
                    other=0.0,
                )
                c_tile = tl.load(
                    c_ptr + offs_k[:, None] * stride_ck + offs_d[None, :] * stride_cd,
                    mask=mask_k[:, None] & mask_d[None, :],
                    other=0.0,
                )
                if USE_BF16:
                    x_tile = x_tile.to(tl.bfloat16)
                    c_tile = c_tile.to(tl.bfloat16)
                    dots = tl.dot(x_tile, tl.trans(c_tile), acc=dots, out_dtype=tl.float32)
                elif USE_FP16:
                    x_tile = x_tile.to(tl.float16)
                    c_tile = c_tile.to(tl.float16)
                    dots = tl.dot(x_tile, tl.trans(c_tile), acc=dots, out_dtype=tl.float32)
                else:
                    dots = tl.dot(
                        x_tile,
                        tl.trans(c_tile),
                        acc=dots,
                        input_precision="ieee",
                        out_dtype=tl.float32,
                    )

            c_norm = tl.load(c_norm_ptr + offs_k, mask=mask_k, other=float("inf"))
            scores = c_norm[None, :] - 2.0 * dots
            scores = tl.where(mask_k[None, :], scores, float("inf"))

            tile_min, tile_idx = tl.min(scores, axis=1, return_indices=True)
            improved = tile_min < best_score
            best_score = tl.where(improved, tile_min, best_score)
            best_label = tl.where(improved, k0 + tile_idx.to(tl.int32), best_label)

        dist = tl.maximum(best_score + x_norm, 0.0)
        tl.store(out_dist_ptr + offs_b, dist, mask=mask_b)
        tl.store(out_label_ptr + offs_b, best_label.to(tl.int64), mask=mask_b)


def prepare_represented_centroids(
    centroids: torch.Tensor,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert centroids to compute_dtype and compute ||c_hat||^2 from the SAME converted values.

    Follows Appendix I.1 (Eq. 10-12) of arXiv:2611.09501 so ||c_hat||^2 - 2 x_hat^T c_hat + ||x_hat||^2
    is always a true Euclidean squared distance without mixed-precision norm bias.
    """
    c_rep = centroids.to(compute_dtype).contiguous()
    c_norm = c_rep.float().square().sum(dim=1).contiguous()
    return c_rep, c_norm


def triton_assign(
    x: torch.Tensor,
    c_rep: torch.Tensor,
    c_norm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Triton nearest-centroid search on CUDA tensors."""
    B, D = x.shape
    K = c_rep.shape[0]
    out_dist = torch.empty((B,), dtype=torch.float32, device=x.device)
    out_label = torch.empty((B,), dtype=torch.int64, device=x.device)
    if B == 0:
        return out_dist, out_label

    use_bf16 = c_rep.dtype == torch.bfloat16
    use_fp16 = c_rep.dtype == torch.float16
    x_in = x.to(c_rep.dtype) if (use_bf16 or use_fp16) else x.float()
    x_in = x_in.contiguous()

    block_b = 64
    block_k = 128 if K >= 128 else 64
    block_d = 64 if (use_bf16 or use_fp16) else 32
    grid = (triton.cdiv(B, block_b),)

    _fused_assign_kernel[grid](
        x_in,
        c_rep,
        c_norm,
        out_dist,
        out_label,
        B,
        K,
        D,
        x_in.stride(0),
        x_in.stride(1),
        c_rep.stride(0),
        c_rep.stride(1),
        BLOCK_B=block_b,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        USE_BF16=use_bf16,
        USE_FP16=use_fp16,
        num_warps=4,
        num_stages=3,
    )
    return out_dist, out_label


def torch_assign_chunked(
    x: torch.Tensor,
    c_rep: torch.Tensor,
    c_norm: torch.Tensor,
    centroid_chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Memory-bounded PyTorch nearest-centroid assignment via tiled addmm."""
    B = x.shape[0]
    K = c_rep.shape[0]
    if B == 0:
        return (
            torch.empty((0,), dtype=torch.float32, device=x.device),
            torch.empty((0,), dtype=torch.int64, device=x.device),
        )
    if c_rep.dtype == torch.float64:
        x64 = x.double()
        c64_norm = c_rep.square().sum(dim=1)
        x_norm64 = x64.square().sum(dim=1)
        best_score64 = torch.full((B,), float("inf"), dtype=torch.float64, device=x.device)
        best_label = torch.zeros((B,), dtype=torch.int64, device=x.device)
        for k0 in range(0, K, centroid_chunk_size):
            k1 = min(k0 + centroid_chunk_size, K)
            c_block = c_rep[k0:k1]
            cn_block = c64_norm[k0:k1]
            scores = torch.addmm(cn_block[None, :], x64, c_block.t(), alpha=-2.0)
            block_min, block_idx = scores.min(dim=1)
            better = block_min < best_score64
            best_score64 = torch.where(better, block_min, best_score64)
            best_label = torch.where(better, block_idx + k0, best_label)
        dist = (best_score64 + x_norm64).clamp_min_(0.0).float()
        return dist, best_label

    x_rep = x.to(c_rep.dtype)
    x_norm = x_rep.float().square().sum(dim=1)

    best_score = torch.full((B,), float("inf"), dtype=torch.float32, device=x.device)
    best_label = torch.zeros((B,), dtype=torch.int64, device=x.device)

    use_lowp_addmm = x_rep.is_cuda and x_rep.dtype in (torch.bfloat16, torch.float16)
    for k0 in range(0, K, centroid_chunk_size):
        k1 = min(k0 + centroid_chunk_size, K)
        c_block = c_rep[k0:k1]
        cn_block = c_norm[k0:k1]
        if use_lowp_addmm:
            scores = torch.addmm(
                cn_block[None, :],
                x_rep,
                c_block.t(),
                alpha=-2.0,
                out_dtype=torch.float32,
            )
        else:
            scores = torch.addmm(
                cn_block[None, :],
                x_rep.float(),
                c_block.float().t(),
                alpha=-2.0,
            )
        block_min, block_idx = scores.min(dim=1)
        better = block_min < best_score
        best_score = torch.where(better, block_min, best_score)
        best_label = torch.where(better, block_idx + k0, best_label)

    dist = (best_score + x_norm).clamp_min_(0.0)
    return dist, best_label


def assign_points(
    x: torch.Tensor,
    centroids: torch.Tensor,
    compute_dtype: torch.dtype | None = None,
    backend: str = "auto",
    row_chunk_size: int = 131072,
    c_rep_cached: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign rows of x (N, D) to nearest centroids (K, D), returning (sq_dists_fp32, labels_int64)."""
    N = x.shape[0]
    if N == 0:
        return (
            torch.empty((0,), dtype=torch.float32, device=x.device),
            torch.empty((0,), dtype=torch.int64, device=x.device),
        )
    if centroids.shape[0] == 0:
        raise ValueError("centroids must be non-empty (K >= 1)")

    if compute_dtype is None:
        if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16):
            compute_dtype = x.dtype
        elif x.is_cuda:
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float32

    if c_rep_cached is not None:
        c_rep, c_norm = c_rep_cached
    else:
        c_rep, c_norm = prepare_represented_centroids(centroids, compute_dtype)

    triton_supported_dtype = c_rep.dtype in (torch.float32, torch.bfloat16, torch.float16)
    use_triton = triton_supported_dtype and (
        (backend == "triton" and x.is_cuda and HAS_TRITON)
        or (backend == "auto" and x.is_cuda and HAS_TRITON and x.shape[1] >= 16)
    )

    if N <= row_chunk_size:
        if use_triton:
            return triton_assign(x, c_rep, c_norm)
        if not x.is_cuda and HAS_FAISS and compute_dtype == torch.float32 and backend != "torch":
            x_np = np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)
            c_np = np.ascontiguousarray(c_rep.detach().cpu().numpy(), dtype=np.float32)
            d_np, l_np = faiss.knn(x_np, c_np, 1)
            return (
                torch.from_numpy(d_np.ravel()).to(dtype=torch.float32),
                torch.from_numpy(l_np.ravel()).to(dtype=torch.int64),
            )
        return torch_assign_chunked(x, c_rep, c_norm)

    dists = torch.empty((N,), dtype=torch.float32, device=x.device)
    labels = torch.empty((N,), dtype=torch.int64, device=x.device)
    for start in range(0, N, row_chunk_size):
        end = min(start + row_chunk_size, N)
        d_chunk, l_chunk = assign_points(
            x[start:end],
            centroids,
            compute_dtype=compute_dtype,
            backend=backend,
            row_chunk_size=row_chunk_size,
            c_rep_cached=(c_rep, c_norm),
        )
        dists[start:end] = d_chunk
        labels[start:end] = l_chunk
    return dists, labels


def exact_mse_fp64(
    x: torch.Tensor | np.ndarray,
    centroids: torch.Tensor | np.ndarray,
    device: str | torch.device | None = None,
    chunk_size: int = 32768,
) -> float:
    """Compute exact double-precision Mean Squared Error (paper evaluation protocol)."""
    if isinstance(x, np.ndarray):
        x_t = torch.from_numpy(x)
    else:
        x_t = x.detach()
    if isinstance(centroids, np.ndarray):
        c_t = torch.from_numpy(centroids)
    else:
        c_t = centroids.detach()

    n = x_t.shape[0]
    if n == 0:
        return 0.0

    dev = torch.device(device) if device is not None else (
        x_t.device if x_t.is_cuda else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )
    c64_raw = c_t.to(device=dev, dtype=torch.float64)
    origin = c64_raw.mean(dim=0, keepdim=True)
    c64 = (c64_raw - origin).contiguous()
    c_norm = c64.square().sum(dim=1)

    total_sq = 0.0
    for i in range(0, n, chunk_size):
        xb = x_t[i : min(i + chunk_size, n)].to(device=dev, dtype=torch.float64) - origin
        x_norm = xb.square().sum(dim=1)
        # chunk centroids too if K is huge
        best = torch.full((xb.shape[0],), float("inf"), dtype=torch.float64, device=dev)
        for k0 in range(0, c64.shape[0], 8192):
            cb = c64[k0 : min(k0 + 8192, c64.shape[0])]
            cn = c_norm[k0 : min(k0 + 8192, c64.shape[0])]
            scores = cn[None, :] - 2.0 * (xb @ cb.t())
            best = torch.minimum(best, scores.min(dim=1).values)
        d2 = (best + x_norm).clamp_min_(0.0)
        total_sq += float(d2.sum().item())
    return total_sq / n
