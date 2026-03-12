"""
tune_glam.py

Tune GLaM hidden space dimensions (num_alpha_bins, num_r_bins) using Ray Tune.
For each trial: truth sparse cells → GLaM encode → sparse decode → compare vs original.

Usage:
  python tune_glam.py --config configs/tune_glam_photon.json --results-dir ./results --n-trials 100
"""

import argparse
import glob
import json
import os

import h5py as h5
import numpy as np
import ray
import ray.tune
import torch

from calodiffusion.utils.HGCal_utils import HGCalConverter, HighLevelFeatures

try:
    import jetnet
except ImportError:
    jetnet = None


def load_truth_showers(data_dir, max_files, nevts_per_file):
    """Load truth showers from h5 files.

    Returns (showers [N, num_layers, max_ncell], energies [N, 1]).
    """
    input_files = sorted(glob.glob(os.path.join(data_dir, "HGCal_showers*.h5")))
    if not input_files:
        raise FileNotFoundError(f"No HGCal_showers*.h5 found in {data_dir}")
    if max_files > 0:
        input_files = input_files[:max_files]

    all_showers = []
    all_energies = []
    for fpath in input_files:
        with h5.File(fpath, "r") as f:
            n = f["showers"].shape[0]
            end = n if nevts_per_file <= 0 else min(nevts_per_file, n)
            all_showers.append(f["showers"][:end])
            all_energies.append(f["gen_info"][:end, 0:1])

    showers = np.concatenate(all_showers, axis=0)
    energies = np.concatenate(all_energies, axis=0)
    print(f"Loaded {showers.shape[0]} events from {len(input_files)} files, "
          f"shower shape {showers.shape}")
    return showers, energies


def compute_metrics(truth, decoded, energies, hlf):
    """Compute FPD and sparsity difference between truth and decoded showers."""
    truth_features = hlf(truth, energies)
    decoded_features = hlf(decoded, energies)

    # FPD
    if jetnet is None:
        raise ImportError("jetnet is required for FPD. Install with: pip install jetnet")
    fpd_val, _ = jetnet.evaluation.fpd(
        np.nan_to_num(decoded_features), np.nan_to_num(truth_features)
    )

    # Sparsity difference per layer
    eps = 1e-6
    truth_sparsity = np.mean(truth > eps, axis=2)       # (N, L)
    decoded_sparsity = np.mean(decoded > eps, axis=2)    # (N, L)
    sparsity_diff = np.mean(np.abs(truth_sparsity - decoded_sparsity))

    return {"FPD": float(fpd_val), "sparsity_diff": float(sparsity_diff)}


class GLaMTuneTrainable(ray.tune.Trainable):
    """Ray Tune Trainable that measures GLaM encode→sparse_decode round-trip quality."""

    def setup(self, config):
        self.num_alpha_bins = config["num_alpha_bins"]
        self.num_r_bins = config["num_r_bins"]
        num_layers = config["num_layers"]
        geom_file = config["geom_file"]
        self.shower_scale = config.get("shower_scale", 200.0)
        batch_size = config.get("batch_size", 256)
        self.batch_size = batch_size

        bins = [num_layers, self.num_alpha_bins, self.num_r_bins]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        print(f"Trial: alpha={self.num_alpha_bins}, r={self.num_r_bins}, "
              f"grid_size={self.num_alpha_bins * self.num_r_bins}")

        # Build converter
        self.converter = HGCalConverter(
            bins=bins, geom_file=geom_file, trainable=False, device=device
        ).to(device=device)
        self.converter.init(norm=False)

        # Feature extractor
        self.hlf = HighLevelFeatures(geom_file)

        # Load data
        self.truth_showers, self.energies = load_truth_showers(
            config["data_dir"], config.get("max_files", 20),
            config.get("nevts_per_file", 500),
        )
        self.current_step = 0

    def step(self):
        self.current_step += 1

        # Encode truth → GLaM grid
        scaled = self.truth_showers * self.shower_scale
        encoded = self.converter.enc_batches(scaled, batch_size=self.batch_size)

        # Sparse decode back to cell space
        decoded = self.converter.dec_batches(
            encoded, batch_size=self.batch_size, sparse_decoding=True
        )
        decoded = np.clip(decoded / self.shower_scale, 0.0, None)

        # Compute metrics
        metrics = compute_metrics(
            self.truth_showers, decoded, self.energies, self.hlf
        )
        metrics["num_alpha_bins"] = self.num_alpha_bins
        metrics["num_r_bins"] = self.num_r_bins
        metrics["grid_size"] = self.num_alpha_bins * self.num_r_bins
        metrics["step"] = self.current_step

        print(f"  alpha={self.num_alpha_bins}, r={self.num_r_bins} → "
              f"FPD={metrics['FPD']:.4f}, sparsity_diff={metrics['sparsity_diff']:.4f}")

        return metrics

    def save_checkpoint(self, checkpoint_dir):
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Tune GLaM grid dimensions with Ray Tune")
    parser.add_argument("--config", required=True, help="Config JSON file")
    parser.add_argument("--results-dir", default=None,
                        help="Directory for Ray Tune results (default: ~/glam_tune_results/)")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Number of trials (overrides config)")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = json.load(f)

    results_dir = args.results_dir or os.path.join(
        os.environ.get("HOME", "."), "glam_tune_results", config.get("particle", "unknown")
    )
    os.makedirs(results_dir, exist_ok=True)

    n_trials = args.n_trials or config.get("n_trials", 100)
    search_space = config["search_space"]

    # Resolve geom_file relative to repo root if not absolute
    geom_file = config["geom_file"]
    if not os.path.isabs(geom_file):
        geom_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), geom_file)

    # Build param space: tunable params + fixed base config
    param_space = {
        "num_alpha_bins": ray.tune.randint(*search_space["num_alpha_bins"]),
        "num_r_bins": ray.tune.randint(*search_space["num_r_bins"]),
        "num_layers": config["num_layers"],
        "geom_file": geom_file,
        "data_dir": config["data_dir"],
        "max_files": config.get("max_files", 20),
        "nevts_per_file": config.get("nevts_per_file", 500),
        "batch_size": config.get("batch_size", 256),
        "shower_scale": config.get("shower_scale", 200.0),
    }

    # SLURM-aware resource detection
    ray.init()
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        try:
            total_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK",
                             os.environ.get("SLURM_JOB_CPUS_PER_NODE", "16")))
        except (ValueError, KeyError):
            total_cpus = 16
        cpu_per_trial = max(1, total_cpus // n_gpu)
        gpu_per_trial = 1
    else:
        cpu_per_trial = 1
        gpu_per_trial = 0

    resources = {"cpu": cpu_per_trial, "gpu": gpu_per_trial}
    print(f"Resources per trial: {resources} ({n_gpu} GPUs detected)")

    experiment_name = f"glam_tune_{config.get('particle', 'unknown')}"

    tuner = ray.tune.Tuner(
        ray.tune.with_resources(GLaMTuneTrainable, resources=resources),
        tune_config=ray.tune.TuneConfig(
            metric="FPD",
            mode="min",
            num_samples=n_trials,
        ),
        run_config=ray.tune.RunConfig(
            name=experiment_name,
            storage_path=results_dir,
            stop={"training_iteration": 1},
        ),
        param_space=param_space,
    )

    result_grid = tuner.fit()

    # Save results
    results_df = result_grid.get_dataframe()
    results_path = os.path.join(results_dir, f"{experiment_name}_results.json")
    results_df.to_json(results_path)
    print(f"\nResults saved to {results_path}")

    # Print best result
    try:
        best = result_grid.get_best_result(metric="FPD", mode="min")
        print(f"\nBest trial:")
        print(f"  num_alpha_bins = {best.config['num_alpha_bins']}")
        print(f"  num_r_bins     = {best.config['num_r_bins']}")
        print(f"  grid_size      = {best.config['num_alpha_bins'] * best.config['num_r_bins']}")
        print(f"  FPD            = {best.metrics['FPD']:.6f}")
        print(f"  sparsity_diff  = {best.metrics['sparsity_diff']:.6f}")
    except RuntimeError as e:
        print(f"\nNo successful trials found: {e}")
        print("Check individual trial error logs above for details.")


if __name__ == "__main__":
    main()
