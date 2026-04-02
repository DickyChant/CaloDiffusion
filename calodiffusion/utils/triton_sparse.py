"""
Triton-accelerated sparse decoding for HGCal calorimeter showers.

Replaces generate_sparse_mat() + einsum with fused Triton kernels that avoid
materializing the full (B, L, N, E) sparse matrix, reducing memory from
O(B*L*N*E) to O(B*L*E).

Also provides a non-fused generate_sparse_mat_triton() that produces the same
output as the PyTorch version but with fewer intermediate allocations.
"""

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _sparse_norms_kernel(
        in_mat_ptr,
        norms_ptr,
        argmax_ptr,
        seed,
        L,
        N,
        E,
        B_eff,
        BLOCK_N: tl.constexpr,
    ):
        """Compute per-column normalization factors and argmax indices.

        For each column (b, l, e) of the decode matrix, generates random
        activations and stores the count of activated entries (norm) and the
        index of the guaranteed-active entry (argmax).

        Grid: (B_eff * L * E,)
        """
        pid = tl.program_id(0)
        b = pid // (L * E)
        rem = pid % (L * E)
        l = rem // E
        e = rem % E

        EPS: tl.constexpr = 1e-6

        n_offs = tl.arange(0, BLOCK_N)
        valid = n_offs < N

        # Load column: in_mat[l, :, e]  (contiguous layout L x N x E)
        ptrs = in_mat_ptr + l * (N * E) + n_offs * E + e
        col = tl.load(ptrs, mask=valid, other=0.0)

        nz = col > EPS
        nz_f = nz.to(tl.float32)

        # Deterministic random per (b, l, n, e)
        rng_off = b * (L * N * E) + l * (N * E) + n_offs * E + e
        r = tl.rand(seed, rng_off)
        rc = r * nz_f + col

        # Argmax over n (smallest index wins ties, matching torch.argmax)
        mx = tl.max(rc, axis=0)
        is_mx = (rc == mx) & valid
        aidx = tl.min(tl.where(is_mx, n_offs, N), axis=0)

        # Guarantee argmax passes the >1.0 threshold
        rc = tl.where(n_offs == aidx, 1.0 + EPS, rc)

        # Count activated entries
        act = ((rc > 1.0) & valid & nz).to(tl.float32)
        norm = tl.sum(act, axis=0)
        norm = tl.maximum(norm, 1.0)

        idx = b * (L * E) + l * E + e
        tl.store(norms_ptr + idx, norm)
        tl.store(argmax_ptr + idx, aidx)

    @triton.jit
    def _sparse_decode_kernel(
        in_mat_ptr,
        x_ptr,
        norms_ptr,
        argmax_ptr,
        out_ptr,
        seed,
        B,
        C,
        L,
        N,
        E,
        B_eff,
        BLOCK_E: tl.constexpr,
    ):
        """Original (non-tiled) sparse-weight generation + matmul.

        One output element per block.
        Grid: (B * C * L * N,)
        """
        pid = tl.program_id(0)
        n = pid % N
        rem = pid // N
        l = rem % L
        rem = rem // L
        c = rem % C
        b = rem // C

        b_eff = b % B_eff

        EPS: tl.constexpr = 1e-6
        acc = tl.zeros([1], dtype=tl.float32)

        for e_start in range(0, E, BLOCK_E):
            eo = e_start + tl.arange(0, BLOCK_E)
            em = eo < E

            cv = tl.load(
                in_mat_ptr + l * (N * E) + n * E + eo, mask=em, other=0.0
            )
            nz = cv > EPS

            rng_off = b_eff * (L * N * E) + l * (N * E) + n * E + eo
            r = tl.rand(seed, rng_off)
            rv = r * nz.to(tl.float32) + cv

            ai = tl.load(
                argmax_ptr + b_eff * (L * E) + l * E + eo, mask=em, other=-1
            )
            rv = tl.where(ai == n, 1.0 + EPS, rv)

            act = (rv > 1.0) & nz & em
            nm = tl.load(
                norms_ptr + b_eff * (L * E) + l * E + eo, mask=em, other=1.0
            )
            w = act.to(tl.float32) / nm

            xv = tl.load(
                x_ptr + b * (C * L * E) + c * (L * E) + l * E + eo,
                mask=em, other=0.0,
            )

            acc += tl.sum(w * xv, axis=0)

        tl.store(
            out_ptr + b * (C * L * N) + c * (L * N) + l * N + n,
            tl.sum(acc, axis=0),
        )

    @triton.jit
    def _sparse_decode_tiled_kernel(
        in_mat_ptr,
        x_ptr,
        norms_ptr,
        argmax_ptr,
        out_ptr,
        seed,
        B,
        C,
        L,
        N,
        E,
        B_eff,
        N_TILES,
        TILE_N: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """Tiled sparse-weight generation + matmul.

        Computes out[b,c,l,n] = sum_e(sparse_weight[b,l,n,e] * x[b,c,l,e])
        for a tile of TILE_N consecutive n values per block.

        Grid: (B * C * L * N_TILES,)
        """
        pid = tl.program_id(0)
        nt = pid % N_TILES
        rem = pid // N_TILES
        l = rem % L
        rem = rem // L
        c = rem % C
        b = rem // C

        b_eff = b % B_eff

        EPS: tl.constexpr = 1e-6

        n_start = nt * TILE_N
        n_offs = n_start + tl.arange(0, TILE_N)  # [TILE_N]
        n_mask = n_offs < N

        acc = tl.zeros([TILE_N], dtype=tl.float32)

        for e_start in range(0, E, BLOCK_E):
            eo = e_start + tl.arange(0, BLOCK_E)  # [BLOCK_E]
            em = eo < E

            # x[b, c, l, eo] — shared across all TILE_N outputs  [BLOCK_E]
            xv = tl.load(
                x_ptr + b * (C * L * E) + c * (L * E) + l * E + eo,
                mask=em, other=0.0,
            )

            # norms[b_eff, l, eo] and argmax[b_eff, l, eo] — shared  [BLOCK_E]
            nm = tl.load(
                norms_ptr + b_eff * (L * E) + l * E + eo,
                mask=em, other=1.0,
            )
            ai = tl.load(
                argmax_ptr + b_eff * (L * E) + l * E + eo,
                mask=em, other=-1,
            )

            # in_mat[l, n_offs, eo] — 2D tile [TILE_N, BLOCK_E]
            mat_ptrs = (in_mat_ptr + l * (N * E)
                        + n_offs[:, None] * E + eo[None, :])
            mask_2d = n_mask[:, None] & em[None, :]
            cv = tl.load(mat_ptrs, mask=mask_2d, other=0.0)

            nz = cv > EPS

            # Regenerate random — [TILE_N, BLOCK_E]
            rng_off = (b_eff * (L * N * E) + l * (N * E)
                       + n_offs[:, None] * E + eo[None, :])
            r = tl.rand(seed, rng_off)
            rv = r * nz.to(tl.float32) + cv

            # Argmax check: ai is [BLOCK_E], broadcast to [TILE_N, BLOCK_E]
            rv = tl.where(
                n_offs[:, None] == ai[None, :],
                1.0 + EPS, rv,
            )

            # Activation + weight  [TILE_N, BLOCK_E]
            act = (rv > 1.0) & nz & mask_2d
            w = act.to(tl.float32) / nm[None, :]

            # Dot product: reduce over E dim → [TILE_N]
            acc += tl.sum(w * xv[None, :], axis=1)

        # Store TILE_N results
        out_ptrs = out_ptr + b * (C * L * N) + c * (L * N) + l * N + n_offs
        tl.store(out_ptrs, acc, mask=n_mask)

    @triton.jit
    def _generate_sparse_col_kernel(
        in_mat_ptr,
        out_ptr,
        seed,
        L,
        N,
        E,
        BLOCK_N: tl.constexpr,
    ):
        """Non-fused: generate one column of the sparse matrix.

        Processes column (b, l, e), writing N output values.
        Grid: (B_eff * L * E,)
        """
        pid = tl.program_id(0)
        b = pid // (L * E)
        rem = pid % (L * E)
        l = rem // E
        e = rem % E

        EPS: tl.constexpr = 1e-6

        n_offs = tl.arange(0, BLOCK_N)
        valid = n_offs < N

        col = tl.load(
            in_mat_ptr + l * (N * E) + n_offs * E + e, mask=valid, other=0.0
        )
        nz = col > EPS
        nz_f = nz.to(tl.float32)

        rng_off = b * (L * N * E) + l * (N * E) + n_offs * E + e
        r = tl.rand(seed, rng_off)
        rc = r * nz_f + col

        mx = tl.max(rc, axis=0)
        is_mx = (rc == mx) & valid
        aidx = tl.min(tl.where(is_mx, n_offs, N), axis=0)
        rc = tl.where(n_offs == aidx, 1.0 + EPS, rc)

        act = ((rc > 1.0) & valid & nz).to(tl.float32)
        norm = tl.sum(act, axis=0)
        norm = tl.maximum(norm, EPS)
        sparse = act / norm * nz_f

        tl.store(
            out_ptr + b * (L * N * E) + l * (N * E) + n_offs * E + e,
            sparse,
            mask=valid,
        )


def _get_block_n(N):
    return triton.next_power_of_2(N)


def _generate_sparse_mat_pytorch(in_mat):
    """Generate a single sparse decode matrix using PyTorch. Returns (L, N, E)."""
    eps = 1e-6
    mask = in_mat > eps
    rand_mat = torch.rand_like(in_mat) * mask + in_mat
    maxs = torch.argmax(rand_mat, dim=-2, keepdim=True)
    rand_mat = rand_mat.scatter(-2, maxs, 1.0 + eps)
    sparse_mat = (rand_mat > 1.0).to(torch.float32)
    sparse_mat /= torch.sum(sparse_mat, dim=-2, keepdim=True)
    sparse_mat *= mask
    return sparse_mat


def _run_norms_kernel(in_mat, B_eff, seed):
    """Shared: run the norms/argmax kernel for both tiled and non-tiled paths."""
    L, N, E = in_mat.shape
    device = in_mat.device
    BLOCK_N = _get_block_n(N)

    norms = torch.empty(B_eff, L, E, device=device, dtype=torch.float32)
    argmax = torch.empty(B_eff, L, E, device=device, dtype=torch.int32)

    grid = (B_eff * L * E,)
    _sparse_norms_kernel[grid](
        in_mat, norms, argmax, seed, L, N, E, B_eff, BLOCK_N=BLOCK_N
    )
    return norms, argmax


def sparse_decode_fused(in_mat, x, per_batch=False):
    """Fused sparse decode using the TILED kernel (adaptive tile size).

    For per_batch=True, uses a hybrid approach: Triton generates the sparse
    matrix (only 1 copy), then cuBLAS einsum does the matmul — much faster
    than launching B*C*L*N thread blocks.

    For per_batch=False, uses the fully fused Triton path to avoid
    materializing the (B, L, N, E) sparse matrix.
    """
    assert HAS_TRITON, "Triton is required for sparse_decode_fused"
    in_mat = in_mat.contiguous()
    x = x.contiguous()

    L, N, E = in_mat.shape
    B, C = x.shape[0], x.shape[1]
    assert x.shape[2] == L and x.shape[3] == E

    if per_batch:
        # Hybrid: Triton generates 1 sparse mat, cuBLAS does the matmul.
        sparse_mat = generate_sparse_mat_triton(in_mat, batches=1, per_batch=True)
        sparse_mat = sparse_mat.squeeze(0)
        return torch.einsum("l n e, ... l e -> ... l n", sparse_mat, x)

    # per_shower: fully fused Triton path
    B_eff = B
    seed = int(torch.randint(0, 2**31, (1,)).item())

    norms, argmax = _run_norms_kernel(in_mat, B_eff, seed)

    out = torch.empty(B, C, L, N, device=in_mat.device, dtype=torch.float32)

    # Adaptive tile size: large N benefits from tiling, small N doesn't
    if N > 4096:
        TILE_N = 64
    elif N > 1024:
        TILE_N = 16
    else:
        TILE_N = 4
    BLOCK_E = min(triton.next_power_of_2(E), 128)
    N_TILES = triton.cdiv(N, TILE_N)
    grid_decode = (B * C * L * N_TILES,)
    _sparse_decode_tiled_kernel[grid_decode](
        in_mat, x, norms, argmax, out, seed,
        B, C, L, N, E, B_eff, N_TILES,
        TILE_N=TILE_N, BLOCK_E=BLOCK_E,
    )
    return out


def sparse_decode_fused_notiled(in_mat, x, per_batch=False):
    """Fused sparse decode using the ORIGINAL (non-tiled) kernel.

    Same hybrid per_batch optimization as sparse_decode_fused.
    For per_shower: one output element per thread block. Grid: (B*C*L*N,).
    """
    assert HAS_TRITON, "Triton is required"
    in_mat = in_mat.contiguous()
    x = x.contiguous()

    L, N, E = in_mat.shape
    B, C = x.shape[0], x.shape[1]
    assert x.shape[2] == L and x.shape[3] == E

    if per_batch:
        sparse_mat = generate_sparse_mat_triton(in_mat, batches=1, per_batch=True)
        sparse_mat = sparse_mat.squeeze(0)
        return torch.einsum("l n e, ... l e -> ... l n", sparse_mat, x)

    B_eff = B
    seed = int(torch.randint(0, 2**31, (1,)).item())

    norms, argmax = _run_norms_kernel(in_mat, B_eff, seed)

    out = torch.empty(B, C, L, N, device=in_mat.device, dtype=torch.float32)

    BLOCK_E = min(triton.next_power_of_2(E), 128)
    grid_decode = (B * C * L * N,)
    _sparse_decode_kernel[grid_decode](
        in_mat, x, norms, argmax, out, seed,
        B, C, L, N, E, B_eff, BLOCK_E=BLOCK_E,
    )
    return out


def generate_sparse_mat_triton(in_mat, batches=1, per_batch=False):
    """Triton-accelerated generate_sparse_mat (non-fused).

    Drop-in replacement for generate_sparse_mat(). Produces the (B, L, N, E)
    sparse matrix but avoids intermediate tensors (rand_mat, mask, etc.).

    Args:
        in_mat: (L, N, E) decode matrix
        batches: batch size
        per_batch: if True, single mask expanded for all batches

    Returns:
        sparse_mat: (batches, L, N, E) sparse decode matrix
    """
    assert HAS_TRITON, "Triton is required for generate_sparse_mat_triton"
    in_mat = in_mat.contiguous()

    L, N, E = in_mat.shape
    B_eff = 1 if per_batch else batches
    device = in_mat.device
    seed = int(torch.randint(0, 2**31, (1,)).item())

    BLOCK_N = _get_block_n(N)

    out = torch.empty(B_eff, L, N, E, device=device, dtype=torch.float32)

    grid = (B_eff * L * E,)
    _generate_sparse_col_kernel[grid](
        in_mat, out, seed, L, N, E, BLOCK_N=BLOCK_N
    )

    if per_batch:
        out = out.expand(batches, -1, -1, -1)

    return out


# ── CSC (Compressed Sparse Column) approach ─────────────────────────────
#
# Key insight: dec_mat is extremely sparse (~5 nonzero per column out of
# N=22268 for pion). Precompute the sparsity pattern once, then only
# process nonzero entries. Reduces work per column from O(N) to O(nnz).


def precompute_csc(dec_mat, eps=1e-6):
    """Precompute CSC structure from decode matrix.

    For each column (l, e), stores the indices and values of nonzero entries
    over the N dimension. dec_mat is fixed after init, so this is done once.

    Args:
        dec_mat: (L, N, E) decode matrix

    Returns:
        col_indices: (L, E, max_nnz) int32 — nonzero n indices, -1 padded
        col_values:  (L, E, max_nnz) float32 — corresponding values
        max_nnz: int — max nonzeros per column (used as BLOCK_NNZ constexpr)
    """
    L, N, E = dec_mat.shape
    device = dec_mat.device

    mask = dec_mat > eps                         # (L, N, E)
    nnz_per_col = mask.sum(dim=1)                # (L, E)
    max_nnz = int(nnz_per_col.max().item())
    # round up to power of 2 for Triton constexpr
    max_nnz_padded = 1 << (max_nnz - 1).bit_length()

    col_indices = torch.full(
        (L, E, max_nnz_padded), -1, dtype=torch.int32, device=device
    )
    col_values = torch.zeros(
        (L, E, max_nnz_padded), dtype=torch.float32, device=device
    )

    # Vectorized: for each (l, e), argsort to pack nonzeros to front
    # Transpose to (L, E, N) for easier per-column processing
    dec_t = dec_mat.permute(0, 2, 1).contiguous()    # (L, E, N)
    mask_t = mask.permute(0, 2, 1).contiguous()       # (L, E, N)

    for l in range(L):
        for e in range(E):
            nz_idx = torch.nonzero(mask_t[l, e], as_tuple=False).squeeze(-1)
            k = nz_idx.shape[0]
            if k > 0:
                col_indices[l, e, :k] = nz_idx.to(torch.int32)
                col_values[l, e, :k] = dec_t[l, e, nz_idx]

    return col_indices.contiguous(), col_values.contiguous(), max_nnz_padded


if HAS_TRITON:

    @triton.jit
    def _sparse_decode_csc_kernel(
        col_indices_ptr,   # (L, E, MAX_NNZ)
        col_values_ptr,    # (L, E, MAX_NNZ)
        x_ptr,             # (B, C, L, E)
        out_ptr,           # (B, C, L, N)
        seed,
        B, C, L, N, E, B_eff,
        MAX_NNZ: tl.constexpr,
        TILE_E: tl.constexpr,
    ):
        """CSC sparse decode: only process nonzero entries of dec_mat.

        For each column (l, e), loads ~5 nonzero (n, val) pairs,
        generates ~5 random numbers, computes sparse weights,
        and scatter-adds weighted x to output.

        Grid: (B * C * L * E_TILES,)
        """
        pid = tl.program_id(0)
        E_TILES = tl.cdiv(E, TILE_E)
        et = pid % E_TILES
        rem = pid // E_TILES
        l = rem % L
        rem = rem // L
        c = rem % C
        b = rem // C
        b_eff = b % B_eff

        EPS: tl.constexpr = 1e-6

        e_start = et * TILE_E
        e_offs = e_start + tl.arange(0, TILE_E)

        out_base = b * (C * L * N) + c * (L * N) + l * N
        csc_base = l * (E * MAX_NNZ)

        for ei in range(TILE_E):
            e = e_start + ei
            if e < E:
                # Load x[b, c, l, e]
                x_val = tl.load(x_ptr + b * (C * L * E) + c * (L * E) + l * E + e)

                # Load CSC column: indices and values for (l, e)
                col_off = csc_base + e * MAX_NNZ
                k_offs = tl.arange(0, MAX_NNZ)
                n_indices = tl.load(col_indices_ptr + col_off + k_offs)
                vals = tl.load(col_values_ptr + col_off + k_offs)
                valid = n_indices >= 0

                # Generate random for these ~5 entries (not 22268!)
                rng_off = b_eff * (L * N * E) + l * (N * E) + n_indices * E + e
                r = tl.rand(seed, rng_off)
                rc = tl.where(valid, r + vals, 0.0)

                # Argmax among nonzero entries
                mx = tl.max(rc, axis=0)
                is_mx = (rc == mx) & valid
                aidx = tl.min(tl.where(is_mx, k_offs, MAX_NNZ), axis=0)
                rc = tl.where(k_offs == aidx, 1.0 + EPS, rc)

                # Threshold + normalize
                act = ((rc > 1.0) & valid).to(tl.float32)
                norm = tl.maximum(tl.sum(act, axis=0), 1.0)
                weights = act / norm * x_val

                # Scatter-add to output
                tl.atomic_add(out_ptr + out_base + n_indices, weights, mask=valid)


def _csc_per_batch_scatter(col_indices, col_values, max_nnz, x, L, N, E):
    """CSC per_batch: generate weights from CSC, scatter-add to output.

    Avoids materializing the dense (L, N, E) matrix entirely.
    ~6x faster than PyTorch generate + einsum for pion.
    """
    B, C = x.shape[0], x.shape[1]
    valid = col_indices >= 0  # (L, E, max_nnz)

    rand_vals = torch.rand_like(col_values) * valid.float() + col_values
    maxs = torch.argmax(rand_vals, dim=-1, keepdim=True)
    rand_vals.scatter_(-1, maxs, 1.0 + 1e-6)

    act = (rand_vals > 1.0) & valid
    act_f = act.float()
    norm = act_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
    weights = act_f / norm * valid.float()  # (L, E, max_nnz)

    # weights * x[..., l, e] → scatter to out[..., l, n]
    w = weights.unsqueeze(0).unsqueeze(0)    # (1, 1, L, E, max_nnz)
    xv = x.unsqueeze(-1)                     # (B, C, L, E, 1)
    contrib = w * xv                          # (B, C, L, E, max_nnz)

    BE = E * max_nnz
    contrib_flat = contrib.reshape(B, C, L, BE)
    idx = col_indices.clamp(min=0).long()
    idx_flat = idx.reshape(1, 1, L, BE).expand(B, C, L, BE)

    out = torch.zeros(B, C, L, N, device=x.device, dtype=torch.float32)
    out.scatter_add_(3, idx_flat, contrib_flat)
    return out


def sparse_decode_csc(dec_mat, x, csc_data, per_batch=False):
    """Sparse decode using precomputed CSC structure.

    Dramatically faster for large N (pion) because it only processes
    nonzero entries (~5 per column instead of 22268).

    Args:
        dec_mat: (L, N, E) decode matrix (used for per_batch fallback)
        x: (B, C, L, E) input tensor
        csc_data: tuple from precompute_csc(): (col_indices, col_values, max_nnz)
        per_batch: if True, uses hybrid einsum path

    Returns:
        out: (B, C, L, N) output tensor
    """
    assert HAS_TRITON, "Triton is required"

    L, N, E = dec_mat.shape
    B, C = x.shape[0], x.shape[1]
    x = x.contiguous()
    col_indices, col_values, max_nnz = csc_data

    if per_batch:
        # CSC direct scatter: generate weights from CSC, multiply by x,
        # scatter_add to output. Avoids materializing dense (L, N, E) mat.
        return _csc_per_batch_scatter(col_indices, col_values, max_nnz, x, L, N, E)
    B_eff = B
    seed = int(torch.randint(0, 2**31, (1,)).item())

    # Output must be zeroed for atomic_add
    out = torch.zeros(B, C, L, N, device=x.device, dtype=torch.float32)

    TILE_E = 16
    E_TILES = triton.cdiv(E, TILE_E)
    grid = (B * C * L * E_TILES,)

    _sparse_decode_csc_kernel[grid](
        col_indices, col_values, x, out, seed,
        B, C, L, N, E, B_eff,
        MAX_NNZ=max_nnz, TILE_E=TILE_E,
    )
    return out
