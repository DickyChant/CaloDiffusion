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
        """Fused sparse-weight generation + matmul.

        Computes out[b,c,l,n] = sum_e(sparse_weight[b,l,n,e] * x[b,c,l,e])
        without materializing the full sparse weight matrix.

        Grid: (B * C * L * N,)
        """
        pid = tl.program_id(0)
        n = pid % N
        rem = pid // N
        l = rem % L
        rem = rem // L
        c = rem % C
        b = rem // C

        # per_batch: B_eff=1 → b_eff=0; per_shower: B_eff=B → b_eff=b
        b_eff = b % B_eff

        EPS: tl.constexpr = 1e-6
        acc = tl.zeros([1], dtype=tl.float32)

        for e_start in range(0, E, BLOCK_E):
            eo = e_start + tl.arange(0, BLOCK_E)
            em = eo < E

            # Decode matrix: in_mat[l, n, e]
            cv = tl.load(
                in_mat_ptr + l * (N * E) + n * E + eo, mask=em, other=0.0
            )
            nz = cv > EPS

            # Regenerate same random as norms kernel
            rng_off = b_eff * (L * N * E) + l * (N * E) + n * E + eo
            r = tl.rand(seed, rng_off)
            rv = r * nz.to(tl.float32) + cv

            # Check if n is the argmax for this column
            ai = tl.load(
                argmax_ptr + b_eff * (L * E) + l * E + eo, mask=em, other=-1
            )
            rv = tl.where(ai == n, 1.0 + EPS, rv)

            # Activation + weight
            act = (rv > 1.0) & nz & em
            nm = tl.load(
                norms_ptr + b_eff * (L * E) + l * E + eo, mask=em, other=1.0
            )
            w = act.to(tl.float32) / nm

            # Input: x[b, c, l, e]
            xv = tl.load(
                x_ptr + b * (C * L * E) + c * (L * E) + l * E + eo,
                mask=em,
                other=0.0,
            )

            acc += tl.sum(w * xv, axis=0)

        tl.store(
            out_ptr + b * (C * L * N) + c * (L * N) + l * N + n,
            tl.sum(acc, axis=0),
        )

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


def sparse_decode_fused(in_mat, x, per_batch=False):
    """Fused sparse matrix generation + matmul using Triton.

    Replaces:
        sparse_mat = generate_sparse_mat(in_mat, batches, per_batch)
        out = einsum("b l n e, b c l e -> b c l n", sparse_mat, x)

    Memory: O(B*L*E) instead of O(B*L*N*E).

    Args:
        in_mat: (L, N, E) decode matrix (contiguous)
        x: (B, C, L, E) input tensor (contiguous)
        per_batch: if True, single sparse mask for entire batch

    Returns:
        out: (B, C, L, N) output tensor
    """
    assert HAS_TRITON, "Triton is required for sparse_decode_fused"
    in_mat = in_mat.contiguous()
    x = x.contiguous()

    L, N, E = in_mat.shape
    B, C = x.shape[0], x.shape[1]
    assert x.shape[2] == L and x.shape[3] == E

    B_eff = 1 if per_batch else B
    device = in_mat.device
    seed = int(torch.randint(0, 2**31, (1,)).item())

    BLOCK_N = _get_block_n(N)

    # Kernel 1: compute norms and argmax indices
    norms = torch.empty(B_eff, L, E, device=device, dtype=torch.float32)
    argmax = torch.empty(B_eff, L, E, device=device, dtype=torch.int32)

    grid_norms = (B_eff * L * E,)
    _sparse_norms_kernel[grid_norms](
        in_mat, norms, argmax, seed, L, N, E, B_eff, BLOCK_N=BLOCK_N
    )

    # Kernel 2: fused sparse decode matmul
    out = torch.empty(B, C, L, N, device=device, dtype=torch.float32)

    BLOCK_E = min(triton.next_power_of_2(E), 128)
    grid_decode = (B * C * L * N,)
    _sparse_decode_kernel[grid_decode](
        in_mat, x, norms, argmax, out, seed,
        B, C, L, N, E, B_eff, BLOCK_E=BLOCK_E
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
