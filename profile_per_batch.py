#!/usr/bin/env python3
"""Profile where time is spent in per_batch sparse decode."""
import os
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(os.getcwd(), ".triton_cache"))

import torch
from einops import rearrange

L, N, E = 47, 22268, 576
ALPHA, R = 12, 48
B, C = 64, 1

device = "cuda"
dec_mat = torch.rand(L, N, E, device=device) * (torch.rand(L, N, E, device=device) < 5.0/N).float()

x = torch.randn(B, C, L, ALPHA, R, device=device)
x_flat = rearrange(x, "b c l a r -> b c l (a r)", a=ALPHA, r=R)

# ── Profile PyTorch generate_sparse_mat ──
def gen_sparse(mat):
    eps = 1e-6
    mask = mat > eps
    rand_mat = torch.rand_like(mat) * mask + mat
    maxs = torch.argmax(rand_mat, dim=-2, keepdim=True)
    rand_mat = rand_mat.scatter(-2, maxs, 1.0 + eps)
    sparse_mat = (rand_mat > 1.0).to(torch.float32)
    sparse_mat /= torch.sum(sparse_mat, dim=-2, keepdim=True)
    sparse_mat *= mask
    return sparse_mat

# Warmup
for _ in range(5):
    sm = gen_sparse(dec_mat)
    _ = torch.einsum("l n e, ... l e -> ... l n", sm, x_flat)
torch.cuda.synchronize()

# Time generation
torch.cuda.synchronize()
s = torch.cuda.Event(enable_timing=True)
e = torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(20):
    sm = gen_sparse(dec_mat)
e.record()
torch.cuda.synchronize()
gen_ms = s.elapsed_time(e) / 20
print(f"generate_sparse_mat:  {gen_ms:.2f} ms")

# Time einsum
torch.cuda.synchronize()
s.record()
for _ in range(20):
    _ = torch.einsum("l n e, ... l e -> ... l n", sm, x_flat)
e.record()
torch.cuda.synchronize()
ein_ms = s.elapsed_time(e) / 20
print(f"einsum:               {ein_ms:.2f} ms")
print(f"total:                {gen_ms + ein_ms:.2f} ms")

# ── Now try CSC-based generation for per_batch ──
from calodiffusion.utils.triton_sparse import precompute_csc
col_indices, col_values, max_nnz = precompute_csc(dec_mat)

def gen_sparse_from_csc(col_indices, col_values, L, N, E, max_nnz):
    """Generate sparse mat using precomputed CSC — only touch nonzero entries."""
    # For each (l, e), we have max_nnz candidate entries
    # Generate random for those, threshold, normalize, scatter into dense mat
    valid = col_indices >= 0  # (L, E, max_nnz)
    rand_vals = torch.rand_like(col_values) * valid.float() + col_values

    # Argmax per column
    maxs = torch.argmax(rand_vals, dim=-1, keepdim=True)  # (L, E, 1)
    rand_vals.scatter_(-1, maxs, 1.0 + 1e-6)

    # Threshold
    act = (rand_vals > 1.0) & valid
    act_f = act.float()
    norm = act_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
    weights = act_f / norm  # (L, E, max_nnz)

    # Scatter into dense (L, N, E) matrix
    sparse_mat = torch.zeros(L, N, E, device=col_indices.device)
    # col_indices is (L, E, max_nnz), need to scatter to dim=1 (N)
    idx = col_indices.clamp(min=0).long()  # replace -1 with 0, masked out below
    # sparse_mat[l, idx[l,e,k], e] = weights[l,e,k]
    # Reshape for scatter: need (L, E, max_nnz) -> scatter into (L, N, E)
    # Transpose to (L, max_nnz, E) for scatter on dim=1
    idx_t = idx.permute(0, 2, 1)          # (L, max_nnz, E)
    weights_t = (weights * valid.float()).permute(0, 2, 1)  # (L, max_nnz, E)
    sparse_mat.scatter_add_(1, idx_t, weights_t)
    return sparse_mat

# Warmup
for _ in range(5):
    sm2 = gen_sparse_from_csc(col_indices, col_values, L, N, E, max_nnz)
torch.cuda.synchronize()

s.record()
for _ in range(20):
    sm2 = gen_sparse_from_csc(col_indices, col_values, L, N, E, max_nnz)
e.record()
torch.cuda.synchronize()
csc_gen_ms = s.elapsed_time(e) / 20
print(f"\nCSC generate_sparse:  {csc_gen_ms:.2f} ms")

s.record()
for _ in range(20):
    _ = torch.einsum("l n e, ... l e -> ... l n", sm2, x_flat)
e.record()
torch.cuda.synchronize()
print(f"einsum (same):        {ein_ms:.2f} ms")
print(f"CSC total:            {csc_gen_ms + ein_ms:.2f} ms")

# ── Or skip dense mat entirely: CSC scatter-matmul ──
# Instead of building dense (L,N,E) then einsum, directly compute output
# out[b,c,l,n] = sum over active (e,k) of weight[l,e,k] * x[b,c,l,e]  where n = col_indices[l,e,k]
def csc_direct_matmul(col_indices, col_values, x_flat, L, N, E, max_nnz):
    B, C = x_flat.shape[0], x_flat.shape[1]
    valid = col_indices >= 0
    rand_vals = torch.rand_like(col_values) * valid.float() + col_values
    maxs = torch.argmax(rand_vals, dim=-1, keepdim=True)
    rand_vals.scatter_(-1, maxs, 1.0 + 1e-6)
    act = (rand_vals > 1.0) & valid
    act_f = act.float()
    norm = act_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
    weights = act_f / norm  # (L, E, max_nnz)

    # x_flat: (B, C, L, E) -> gather x values for each (l, e)
    # weights: (L, E, max_nnz) -> multiply with x[..., l, e] and scatter to n
    # Expand weights: (1, 1, L, E, max_nnz)
    w = (weights * valid.float()).unsqueeze(0).unsqueeze(0)  # (1,1,L,E,max_nnz)
    xv = x_flat.unsqueeze(-1)  # (B, C, L, E, 1)
    contrib = w * xv  # (B, C, L, E, max_nnz)

    # Now scatter-add: for each (b,c,l,e,k), add contrib to out[b,c,l,n_indices[l,e,k]]
    idx = col_indices.clamp(min=0).long()  # (L, E, max_nnz)
    # Reshape to (B, C, L, E*max_nnz) and scatter to (B, C, L, N)
    BE = E * max_nnz
    contrib_flat = contrib.reshape(B, C, L, BE)
    idx_flat = idx.reshape(1, 1, L, BE).expand(B, C, L, BE)

    out = torch.zeros(B, C, L, N, device=x_flat.device)
    out.scatter_add_(3, idx_flat, contrib_flat)
    return out

for _ in range(5):
    out3 = csc_direct_matmul(col_indices, col_values, x_flat, L, N, E, max_nnz)
torch.cuda.synchronize()

s.record()
for _ in range(20):
    out3 = csc_direct_matmul(col_indices, col_values, x_flat, L, N, E, max_nnz)
e.record()
torch.cuda.synchronize()
direct_ms = s.elapsed_time(e) / 20
print(f"\nCSC direct scatter:   {direct_ms:.2f} ms  (no dense mat, no einsum)")
