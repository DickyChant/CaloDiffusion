#!/usr/bin/env python3
"""Profile CSC sparse decode on CPU: PyTorch vs Numba."""
import time
import torch
import numpy as np

# Pion dimensions
L, N, E = 47, 22268, 576
MAX_NNZ = 8
B, C = 8, 1  # smaller batch for CPU

device = "cpu"

# ── Build synthetic CSC structure ───────────────────────────────────────
print("Building synthetic CSC structure...")
col_indices = torch.full((L, E, MAX_NNZ), -1, dtype=torch.int32)
col_values = torch.zeros(L, E, MAX_NNZ)
for l in range(L):
    for e_idx in range(E):
        k = min(np.random.poisson(5), MAX_NNZ)
        k = max(k, 1)
        ns = np.random.choice(N, k, replace=False)
        vs = np.random.rand(k).astype(np.float32)
        vs /= vs.sum()
        col_indices[l, e_idx, :k] = torch.from_numpy(ns.astype(np.int32))
        col_values[l, e_idx, :k] = torch.from_numpy(vs)

x = torch.randn(B, C, L, E)
print(f"CSC structure: ({L}, {E}, {MAX_NNZ}) = {col_indices.numel() * 4 / 1e6:.1f} MB")
print(f"vs dense dec_mat: ({L}, {N}, {E}) = {L * N * E * 4 / 1e6:.1f} MB")
print()

# ── PyTorch CSC (vectorized) ───────────────────────────────────────────
def pytorch_csc_per_shower(col_indices, col_values, x, L, N, E, MAX_NNZ):
    B, C = x.shape[0], x.shape[1]
    valid = col_indices >= 0
    valid_b = valid.unsqueeze(0)
    vals_b = col_values.unsqueeze(0)
    rand_vals = torch.rand(B, L, E, MAX_NNZ) * valid_b.float() + vals_b
    rand_vals.scatter_(-1, torch.argmax(rand_vals, dim=-1, keepdim=True), 1.0 + 1e-6)
    act = (rand_vals > 1.0) & valid_b
    act_f = act.float()
    weights = act_f / act_f.sum(dim=-1, keepdim=True).clamp(min=1.0) * valid_b.float()

    w = weights.unsqueeze(1)
    xv = x.unsqueeze(-1)
    contrib = w * xv

    BE = E * MAX_NNZ
    contrib_flat = contrib.reshape(B, C, L, BE)
    idx = col_indices.clamp(min=0).long().reshape(1, 1, L, BE).expand(B, C, L, BE)
    out = torch.zeros(B, C, L, N)
    out.scatter_add_(3, idx, contrib_flat)
    return out

# Warmup
for _ in range(3):
    _ = pytorch_csc_per_shower(col_indices, col_values, x, L, N, E, MAX_NNZ)

t0 = time.perf_counter()
N_ITERS = 10
for _ in range(N_ITERS):
    _ = pytorch_csc_per_shower(col_indices, col_values, x, L, N, E, MAX_NNZ)
pt_ms = (time.perf_counter() - t0) / N_ITERS * 1000
print(f"PyTorch CSC (CPU):    {pt_ms:.1f} ms  ({pt_ms/B:.1f} ms/sample)")

# ── Break down PyTorch CSC steps ────────────────────────────────────────
valid = col_indices >= 0
valid_b = valid.unsqueeze(0)
vals_b = col_values.unsqueeze(0)

t0 = time.perf_counter()
for _ in range(N_ITERS):
    rand_vals = torch.rand(B, L, E, MAX_NNZ) * valid_b.float() + vals_b
    rand_vals.scatter_(-1, torch.argmax(rand_vals, dim=-1, keepdim=True), 1.0 + 1e-6)
    act = (rand_vals > 1.0) & valid_b
    act_f = act.float()
    weights = act_f / act_f.sum(dim=-1, keepdim=True).clamp(min=1.0) * valid_b.float()
gen_ms = (time.perf_counter() - t0) / N_ITERS * 1000
print(f"  - weight generation: {gen_ms:.1f} ms")

w = weights.unsqueeze(1)
xv = x.unsqueeze(-1)
t0 = time.perf_counter()
for _ in range(N_ITERS):
    contrib = w * xv
    BE = E * MAX_NNZ
    contrib_flat = contrib.reshape(B, C, L, BE)
    idx = col_indices.clamp(min=0).long().reshape(1, 1, L, BE).expand(B, C, L, BE)
    out = torch.zeros(B, C, L, N)
    out.scatter_add_(3, idx, contrib_flat)
scatter_ms = (time.perf_counter() - t0) / N_ITERS * 1000
print(f"  - multiply+scatter:  {scatter_ms:.1f} ms")

# ── Numba CSC (explicit loops) ─────────────────────────────────────────
try:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def numba_csc_per_shower(col_indices_np, col_values_np, x_np, rand_np, out_np):
        """Pre-generated random numbers, parallel over B."""
        B, C, L, E = x_np.shape
        MAX_NNZ = col_indices_np.shape[2]

        for b in prange(B):
            for l in range(L):
                for e in range(E):
                    n_active = 0
                    max_val = -1.0
                    max_k = 0

                    # Pass 1: generate activations from pre-generated randoms
                    for k in range(MAX_NNZ):
                        n_idx = col_indices_np[l, e, k]
                        if n_idx < 0:
                            break
                        rv = rand_np[b, l, e, k] + col_values_np[l, e, k]
                        if rv > max_val:
                            max_val = rv
                            max_k = n_active
                        n_active += 1

                    # Recompute with argmax guarantee
                    norm = 0.0
                    for k in range(n_active):
                        rv = rand_np[b, l, e, k] + col_values_np[l, e, k]
                        if k == max_k:
                            rv = 1.0 + 1e-6
                        if rv > 1.0:
                            norm += 1.0
                    if norm < 1.0:
                        norm = 1.0
                    inv_norm = 1.0 / norm

                    # Pass 2: scatter weighted x
                    for k in range(n_active):
                        rv = rand_np[b, l, e, k] + col_values_np[l, e, k]
                        if k == max_k:
                            rv = 1.0 + 1e-6
                        if rv > 1.0:
                            n_idx = col_indices_np[l, e, k]
                            for c_idx in range(C):
                                out_np[b, c_idx, l, n_idx] += inv_norm * x_np[b, c_idx, l, e]

    ci_np = col_indices.numpy()
    cv_np = col_values.numpy()
    x_np = x.numpy()

    # Warmup / compile (include rand generation in timing)
    rand_np = np.random.rand(B, L, E, MAX_NNZ).astype(np.float32)
    out_np = np.zeros((B, C, L, N), dtype=np.float32)
    numba_csc_per_shower(ci_np, cv_np, x_np, rand_np, out_np)
    out_np = np.zeros((B, C, L, N), dtype=np.float32)
    numba_csc_per_shower(ci_np, cv_np, x_np, rand_np, out_np)

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        rand_np = np.random.rand(B, L, E, MAX_NNZ).astype(np.float32)
        out_np = np.zeros((B, C, L, N), dtype=np.float32)
        numba_csc_per_shower(ci_np, cv_np, x_np, rand_np, out_np)
    nb_ms = (time.perf_counter() - t0) / N_ITERS * 1000
    print(f"\nNumba CSC (CPU):      {nb_ms:.1f} ms  ({nb_ms/B:.1f} ms/sample)")
    print(f"Speedup vs PyTorch:   {pt_ms/nb_ms:.1f}x")

    # Breakdown: rand gen vs kernel
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        rand_np = np.random.rand(B, L, E, MAX_NNZ).astype(np.float32)
    rand_ms = (time.perf_counter() - t0) / N_ITERS * 1000
    print(f"  - np.random.rand:    {rand_ms:.1f} ms")
    print(f"  - numba kernel:      {nb_ms - rand_ms:.1f} ms")

except ImportError:
    print("\nNumba not installed — pip install numba")
