#!/usr/bin/env python3
"""Benchmark: PyTorch vs Triton sparse decoding for HGCal showers.

Runs two HGCal configurations:
  - Photon: L=47, N=2076,  E=12x21=252
  - Pion:   L=47, N=22268, E=12x48=576

Usage:
    python benchmark_sparse_decoding.py                          # both configs, both impls
    python benchmark_sparse_decoding.py --config photon          # photon only
    python benchmark_sparse_decoding.py --config pion            # pion only
    python benchmark_sparse_decoding.py --skip-triton            # PyTorch only
    python benchmark_sparse_decoding.py --batch-sizes 32 128     # custom batch sizes
"""

import argparse
import json
import os
from datetime import datetime

# Redirect caches to $PWD to avoid home dir quota issues
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(os.getcwd(), ".triton_cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))

import torch
from einops import rearrange

# ── HGCal configurations ────────────────────────────────────────────────

CONFIGS = {
    "photon": {
        "L": 47,            # layers
        "N": 2076,          # max cells per layer
        "ALPHA_BINS": 12,
        "R_BINS": 21,       # E = 12*21 = 252
        "C": 1,
        "default_batches": [1, 32, 64, 128, 256],
    },
    "pion": {
        "L": 47,
        "N": 22268,         # ~10x more cells than photon
        "ALPHA_BINS": 12,
        "R_BINS": 48,       # E = 12*48 = 576
        "C": 1,
        # pion is much larger; keep default batch sizes conservative
        "default_batches": [1, 8, 16, 32, 64],
    },
}


# ── Synthetic data ──────────────────────────────────────────────────────

def create_synthetic_dec_mat(L, N, E, device="cuda"):
    """Create a realistic sparse decode matrix (L, N, E).

    Mimics the real HGCal geometry: each column (over N) has a handful of
    nonzero entries that sum to ~1.
    """
    mask = torch.rand(L, N, E, device=device) < (5.0 / N)
    forced = torch.randint(0, N, (L, E), device=device)
    mask[
        torch.arange(L, device=device).unsqueeze(1),
        forced,
        torch.arange(E, device=device).unsqueeze(0),
    ] = True
    vals = torch.rand(L, N, E, device=device) * mask.float()
    col_sums = vals.sum(dim=1, keepdim=True).clamp(min=1e-8)
    dec_mat = vals / col_sums * mask.float()
    return dec_mat


# ── PyTorch baseline (from HGCal_utils.py) ─────────────────────────────

def pytorch_sparse_decode(dec_mat, x, alpha_bins, r_bins, per_batch=False):
    """generate_sparse_mat + einsum (the existing PyTorch path)."""
    B = x.shape[0]
    out = rearrange(x, "b c l a r -> b c l (a r)", a=alpha_bins, r=r_bins)

    batch_size = 1 if per_batch else B
    eps = 1e-6

    in_mat = dec_mat.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    mask = in_mat > eps
    rand_mat = torch.rand_like(in_mat) * mask + in_mat
    maxs = torch.argmax(rand_mat, dim=-2, keepdim=True)
    rand_mat = rand_mat.scatter(-2, maxs, 1.0 + eps)
    sparse_mat = (rand_mat > 1.0).to(torch.float32)
    sparse_mat_norm = torch.sum(sparse_mat, dim=-2, keepdim=True)
    sparse_mat /= sparse_mat_norm
    sparse_mat *= mask
    del in_mat, rand_mat, mask

    if per_batch:
        sparse_mat = sparse_mat.repeat(B, 1, 1, 1)

    result = torch.einsum("b l n e, b c l e -> b c l n", sparse_mat, out)
    return result


# ── Chunked PyTorch (from upstream/chunked_sparse branch) ──────────────

def _generate_sparse_mat_single(in_mat, batch_size=1):
    """generate_sparse_mat from chunked_sparse branch (no per_batch flag)."""
    eps = 1e-6
    if batch_size > 1:
        in_mat = in_mat.unsqueeze(0).expand(batch_size, -1, -1, -1)
    mask = in_mat > eps
    rand_mat = torch.rand_like(in_mat) * mask + in_mat
    maxs = torch.argmax(rand_mat, dim=-2, keepdim=True)
    rand_mat = rand_mat.scatter(-2, maxs, 1.0 + eps)
    sparse_mat = (rand_mat > 1.0).to(torch.float32)
    sparse_mat /= torch.sum(sparse_mat, dim=-2, keepdim=True)
    sparse_mat *= mask
    return sparse_mat


def chunked_sparse_decode(dec_mat, x, alpha_bins, r_bins, per_batch=False,
                          shower_chunk_size=10):
    """Chunked PyTorch sparse decode (from upstream/chunked_sparse)."""
    B = x.shape[0]
    out = rearrange(x, "b c l a r -> b c l (a r)", a=alpha_bins, r=r_bins)

    if per_batch:
        sparse_mat = _generate_sparse_mat_single(dec_mat, batch_size=1)
        return torch.einsum("l n e, ... l e -> ... l n", sparse_mat, out)

    results = []
    for i in range(0, B, shower_chunk_size):
        out_chunk = out[i:i + shower_chunk_size]
        k = out_chunk.shape[0]
        sparse_mat_k = _generate_sparse_mat_single(dec_mat, batch_size=k)
        if k == 1:
            sparse_mat_k = sparse_mat_k.unsqueeze(0)
        results.append(torch.einsum("b l n e, b c l e -> b c l n",
                                    sparse_mat_k, out_chunk))
    return torch.cat(results, dim=0)


# ── Triton path ─────────────────────────────────────────────────────────

def triton_sparse_decode(dec_mat, x, alpha_bins, r_bins, per_batch=False):
    """Fused Triton sparse decode — original non-tiled kernel."""
    from calodiffusion.utils.triton_sparse import sparse_decode_fused_notiled

    out = rearrange(x, "b c l a r -> b c l (a r)", a=alpha_bins, r=r_bins)
    return sparse_decode_fused_notiled(dec_mat, out, per_batch=per_batch)


def triton_tiled_sparse_decode(dec_mat, x, alpha_bins, r_bins, per_batch=False):
    """Fused Triton sparse decode — tiled kernel (adaptive TILE_N)."""
    from calodiffusion.utils.triton_sparse import sparse_decode_fused

    out = rearrange(x, "b c l a r -> b c l (a r)", a=alpha_bins, r=r_bins)
    return sparse_decode_fused(dec_mat, out, per_batch=per_batch)


def triton_csc_sparse_decode(dec_mat, x, alpha_bins, r_bins, per_batch=False,
                             _csc_cache={}):
    """Triton CSC sparse decode — only processes nonzero entries."""
    from calodiffusion.utils.triton_sparse import sparse_decode_csc, precompute_csc

    out = rearrange(x, "b c l a r -> b c l (a r)", a=alpha_bins, r=r_bins)
    key = id(dec_mat)
    if key not in _csc_cache:
        _csc_cache[key] = precompute_csc(dec_mat)
    return sparse_decode_csc(dec_mat, out, _csc_cache[key], per_batch=per_batch)


# ── Pure PyTorch CSC (no Triton/Numba needed) ──────────────────────────

def _ensure_pytorch_csc(dec_mat, cache):
    """Precompute CSC using pure PyTorch (same structure as triton precompute)."""
    key = id(dec_mat)
    if key not in cache:
        from calodiffusion.utils.triton_sparse import precompute_csc
        cache[key] = precompute_csc(dec_mat)
    return cache[key]


def pytorch_csc_sparse_decode(dec_mat, x, alpha_bins, r_bins, per_batch=False,
                              _csc_cache={}):
    """Pure PyTorch CSC sparse decode — works on any GPU, no Triton needed."""
    B = x.shape[0]
    C = x.shape[1]
    out = rearrange(x, "b c l a r -> b c l (a r)", a=alpha_bins, r=r_bins)
    col_indices, col_values, max_nnz = _ensure_pytorch_csc(dec_mat, _csc_cache)

    L, N, E = dec_mat.shape
    valid = col_indices >= 0  # (L, E, max_nnz)

    if per_batch:
        # Single random mask — same as _csc_per_batch_scatter
        rand_vals = torch.rand_like(col_values) * valid.float() + col_values
        rand_vals.scatter_(-1, torch.argmax(rand_vals, dim=-1, keepdim=True), 1.0 + 1e-6)
        act = (rand_vals > 1.0) & valid
        act_f = act.float()
        weights = act_f / act_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        weights = weights * valid.float()

        w = weights.unsqueeze(0).unsqueeze(0)       # (1, 1, L, E, nnz)
        xv = out.unsqueeze(-1)                       # (B, C, L, E, 1)
        contrib = w * xv                              # (B, C, L, E, nnz)
    else:
        # Per-shower random mask — (B, L, E, max_nnz), only ~33MB for pion B=64
        valid_b = valid.unsqueeze(0)                  # (1, L, E, nnz)
        vals_b = col_values.unsqueeze(0)              # (1, L, E, nnz)
        rand_vals = torch.rand(B, L, E, max_nnz, device=out.device) * valid_b.float() + vals_b
        rand_vals.scatter_(-1, torch.argmax(rand_vals, dim=-1, keepdim=True), 1.0 + 1e-6)
        act = (rand_vals > 1.0) & valid_b
        act_f = act.float()
        weights = act_f / act_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        weights = weights * valid_b.float()

        w = weights.unsqueeze(1)                      # (B, 1, L, E, nnz)
        xv = out.unsqueeze(-1)                        # (B, C, L, E, 1)
        contrib = w * xv                              # (B, C, L, E, nnz)

    BE = E * max_nnz
    contrib_flat = contrib.reshape(B, C, L, BE)
    idx = col_indices.clamp(min=0).long()
    idx_flat = idx.reshape(1, 1, L, BE).expand(B, C, L, BE)

    result = torch.zeros(B, C, L, N, device=out.device, dtype=torch.float32)
    result.scatter_add_(3, idx_flat, contrib_flat)
    return result


# ── Benchmarking utilities ──────────────────────────────────────────────

def benchmark_fn(fn, num_warmup, num_iters):
    """Returns (mean_ms, std_ms, peak_mem_MB)."""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    for i in range(num_iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    peak_mb = (torch.cuda.max_memory_allocated() - mem_before) / 1e6
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return mean, std, peak_mb


def print_header(cfg_name, cfg):
    E = cfg["ALPHA_BINS"] * cfg["R_BINS"]
    print(f"\n{'='*72}")
    print(f"  {cfg_name.upper()}:  L={cfg['L']}, N={cfg['N']}, "
          f"E={cfg['ALPHA_BINS']}x{cfg['R_BINS']}={E}, C={cfg['C']}")
    est_bytes = cfg["L"] * cfg["N"] * E * 4
    print(f"  dec_mat size: {est_bytes / 1e6:.1f} MB  |  "
          f"per-shower sparse mat (B=1): {est_bytes / 1e6:.1f} MB")
    print(f"{'='*72}\n")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Benchmark sparse decoding: PyTorch vs Triton (photon & pion)"
    )
    p.add_argument(
        "--config", choices=["photon", "pion", "both"], default="both",
        help="Which HGCal configuration to benchmark",
    )
    p.add_argument("--batch-sizes", nargs="+", type=int, default=None,
                   help="Override batch sizes (default: config-specific)")
    p.add_argument("--num-warmup", type=int, default=5)
    p.add_argument("--num-iters", type=int, default=20)
    p.add_argument("--skip-triton", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("-o", "--output", default=None,
                   help="Output JSON file (default: benchmark_results_<timestamp>.json)")
    args = p.parse_args()

    device = args.device
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    has_triton = False
    triton_version = None
    if not args.skip_triton:
        try:
            from calodiffusion.utils.triton_sparse import HAS_TRITON
            has_triton = HAS_TRITON
            if has_triton:
                import triton
                triton_version = triton.__version__
        except ImportError:
            pass
    if not has_triton and not args.skip_triton:
        print("WARNING: Triton not available, running PyTorch only")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "device": torch.cuda.get_device_name(),
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        "torch_version": torch.__version__,
        "triton_version": triton_version,
        "num_warmup": args.num_warmup,
        "num_iters": args.num_iters,
    }
    all_results = []

    cfgs = ["photon", "pion"] if args.config == "both" else [args.config]

    for cfg_name in cfgs:
        cfg = CONFIGS[cfg_name]
        L = cfg["L"]
        N = cfg["N"]
        alpha = cfg["ALPHA_BINS"]
        r = cfg["R_BINS"]
        E = alpha * r
        C = cfg["C"]
        batch_sizes = args.batch_sizes or cfg["default_batches"]

        print_header(cfg_name, cfg)

        dec_mat = create_synthetic_dec_mat(L, N, E, device)

        hdr = (f"{'impl':<10} {'B':>4} {'mode':>11} "
               f"{'mean_ms':>10} {'std_ms':>8} {'ms/sample':>10} {'peak_MB':>10}")
        print(hdr)
        print("-" * len(hdr))

        for B in batch_sizes:
            x = torch.randn(B, C, L, alpha, r, device=device)

            for per_batch in [False, True]:
                mode = "per_batch" if per_batch else "per_shower"

                # — PyTorch —
                try:
                    mean, std, mem = benchmark_fn(
                        lambda: pytorch_sparse_decode(
                            dec_mat, x, alpha, r, per_batch
                        ),
                        args.num_warmup, args.num_iters,
                    )
                    per = mean / B
                    print(f"{'pytorch':<10} {B:>4} {mode:>11} "
                          f"{mean:>10.2f} {std:>8.2f} {per:>10.3f} {mem:>10.1f}")
                    all_results.append({
                        "config": cfg_name, "impl": "pytorch",
                        "batch": B, "mode": mode,
                        "mean_ms": round(mean, 3), "std_ms": round(std, 3),
                        "ms_per_sample": round(per, 3),
                        "peak_MB": round(mem, 1),
                    })
                except torch.cuda.OutOfMemoryError:
                    print(f"{'pytorch':<10} {B:>4} {mode:>11} "
                          f"{'OOM':>10} {'':>8} {'':>10} {'':>10}")
                    all_results.append({
                        "config": cfg_name, "impl": "pytorch",
                        "batch": B, "mode": mode, "OOM": True,
                    })
                    torch.cuda.empty_cache()

                # — Chunked PyTorch (upstream/chunked_sparse) —
                try:
                    mean, std, mem = benchmark_fn(
                        lambda: chunked_sparse_decode(
                            dec_mat, x, alpha, r, per_batch
                        ),
                        args.num_warmup, args.num_iters,
                    )
                    per = mean / B
                    print(f"{'chunked':<10} {B:>4} {mode:>11} "
                          f"{mean:>10.2f} {std:>8.2f} {per:>10.3f} {mem:>10.1f}")
                    all_results.append({
                        "config": cfg_name, "impl": "chunked",
                        "batch": B, "mode": mode,
                        "mean_ms": round(mean, 3), "std_ms": round(std, 3),
                        "ms_per_sample": round(per, 3),
                        "peak_MB": round(mem, 1),
                    })
                except torch.cuda.OutOfMemoryError:
                    print(f"{'chunked':<10} {B:>4} {mode:>11} "
                          f"{'OOM':>10} {'':>8} {'':>10} {'':>10}")
                    all_results.append({
                        "config": cfg_name, "impl": "chunked",
                        "batch": B, "mode": mode, "OOM": True,
                    })
                    torch.cuda.empty_cache()

                # — PyTorch CSC (no Triton needed) —
                try:
                    mean, std, mem = benchmark_fn(
                        lambda: pytorch_csc_sparse_decode(
                            dec_mat, x, alpha, r, per_batch
                        ),
                        args.num_warmup, args.num_iters,
                    )
                    per = mean / B
                    print(f"{'pt_csc':<10} {B:>4} {mode:>11} "
                          f"{mean:>10.2f} {std:>8.2f} {per:>10.3f} {mem:>10.1f}")
                    all_results.append({
                        "config": cfg_name, "impl": "pytorch_csc",
                        "batch": B, "mode": mode,
                        "mean_ms": round(mean, 3), "std_ms": round(std, 3),
                        "ms_per_sample": round(per, 3),
                        "peak_MB": round(mem, 1),
                    })
                except torch.cuda.OutOfMemoryError:
                    print(f"{'pt_csc':<10} {B:>4} {mode:>11} "
                          f"{'OOM':>10} {'':>8} {'':>10} {'':>10}")
                    all_results.append({
                        "config": cfg_name, "impl": "pytorch_csc",
                        "batch": B, "mode": mode, "OOM": True,
                    })
                    torch.cuda.empty_cache()

                # — Triton —
                if has_triton:
                    try:
                        mean, std, mem = benchmark_fn(
                            lambda: triton_sparse_decode(
                                dec_mat, x, alpha, r, per_batch
                            ),
                            args.num_warmup, args.num_iters,
                        )
                        per = mean / B
                        print(f"{'triton':<10} {B:>4} {mode:>11} "
                              f"{mean:>10.2f} {std:>8.2f} {per:>10.3f} {mem:>10.1f}")
                        all_results.append({
                            "config": cfg_name, "impl": "triton",
                            "batch": B, "mode": mode,
                            "mean_ms": round(mean, 3), "std_ms": round(std, 3),
                            "ms_per_sample": round(per, 3),
                            "peak_MB": round(mem, 1),
                        })
                    except torch.cuda.OutOfMemoryError:
                        print(f"{'triton':<10} {B:>4} {mode:>11} "
                              f"{'OOM':>10} {'':>8} {'':>10} {'':>10}")
                        all_results.append({
                            "config": cfg_name, "impl": "triton",
                            "batch": B, "mode": mode, "OOM": True,
                        })
                        torch.cuda.empty_cache()

                # — Triton tiled —
                if has_triton:
                    try:
                        mean, std, mem = benchmark_fn(
                            lambda: triton_tiled_sparse_decode(
                                dec_mat, x, alpha, r, per_batch
                            ),
                            args.num_warmup, args.num_iters,
                        )
                        per = mean / B
                        print(f"{'triton_til':<10} {B:>4} {mode:>11} "
                              f"{mean:>10.2f} {std:>8.2f} {per:>10.3f} {mem:>10.1f}")
                        all_results.append({
                            "config": cfg_name, "impl": "triton_tiled",
                            "batch": B, "mode": mode,
                            "mean_ms": round(mean, 3), "std_ms": round(std, 3),
                            "ms_per_sample": round(per, 3),
                            "peak_MB": round(mem, 1),
                        })
                    except torch.cuda.OutOfMemoryError:
                        print(f"{'triton_til':<10} {B:>4} {mode:>11} "
                              f"{'OOM':>10} {'':>8} {'':>10} {'':>10}")
                        all_results.append({
                            "config": cfg_name, "impl": "triton_tiled",
                            "batch": B, "mode": mode, "OOM": True,
                        })
                        torch.cuda.empty_cache()

                # — Triton CSC —
                if has_triton:
                    try:
                        mean, std, mem = benchmark_fn(
                            lambda: triton_csc_sparse_decode(
                                dec_mat, x, alpha, r, per_batch
                            ),
                            args.num_warmup, args.num_iters,
                        )
                        per = mean / B
                        print(f"{'triton_csc':<10} {B:>4} {mode:>11} "
                              f"{mean:>10.2f} {std:>8.2f} {per:>10.3f} {mem:>10.1f}")
                        all_results.append({
                            "config": cfg_name, "impl": "triton_csc",
                            "batch": B, "mode": mode,
                            "mean_ms": round(mean, 3), "std_ms": round(std, 3),
                            "ms_per_sample": round(per, 3),
                            "peak_MB": round(mem, 1),
                        })
                    except torch.cuda.OutOfMemoryError:
                        print(f"{'triton_csc':<10} {B:>4} {mode:>11} "
                              f"{'OOM':>10} {'':>8} {'':>10} {'':>10}")
                        all_results.append({
                            "config": cfg_name, "impl": "triton_csc",
                            "batch": B, "mode": mode, "OOM": True,
                        })
                        torch.cuda.empty_cache()

            del x
            torch.cuda.empty_cache()
            print()

        del dec_mat
        torch.cuda.empty_cache()

    # ── Save results ────────────────────────────────────────────────────
    outfile = args.output or f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output = {"metadata": metadata, "results": all_results}
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
