import argparse
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


def q(value: str) -> str:
    return shlex.quote(str(value))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    slurm_dir = Path(__file__).resolve().parent

    user = os.environ.get("USER", "unknown")
    default_output_root = Path("/pscratch") / user / "calodiff_measurements"

    parser = argparse.ArgumentParser(
        description="Create and submit a SLURM job for measurement.sh inference timing."
    )
    parser.add_argument("-n", "--name", default="infer_measure", help="Job label")
    parser.add_argument("-d", "--data-dir", required=True, help="Input data directory for measurement.sh")
    parser.add_argument("--repo-dir", default=str(repo_root), help="Repository root path")
    parser.add_argument("--output-root", default=str(default_output_root), help="Base output directory (should be on /pscratch)")
    parser.add_argument("--n-events", type=int, default=1000)
    parser.add_argument("--sample-steps", type=int, default=200)
    parser.add_argument("--batch-sizes", default="1 10 100 1000")
    parser.add_argument("--auto-prepare-dummy", type=int, choices=[0, 1], default=1)
    parser.add_argument("--photon-config", default="", help="Optional override")
    parser.add_argument("--pion-config", default="", help="Optional override")
    parser.add_argument("--photon-model", default="", help="Optional override")
    parser.add_argument("--pion-model", default="", help="Optional override")
    parser.add_argument("--partition", default="regular")
    parser.add_argument("--constraint", default="gpu")
    parser.add_argument("--memory", default="64G")
    parser.add_argument("--time-limit", default="02:00:00")
    parser.add_argument("--account", default="", help="Optional SLURM account")
    parser.add_argument("--conda-env", default="calodiff-env", help="Set to 'none' to skip conda activation")
    parser.add_argument("--dry-run", action="store_true", help="Only generate script, do not submit")
    args = parser.parse_args()

    template_path = slurm_dir / "measurement_template.sh"
    template = template_path.read_text(encoding="ascii")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = Path(args.output_root).expanduser() / args.name / stamp
    job_dir.mkdir(parents=True, exist_ok=True)

    account_line = f"#SBATCH --account={args.account}" if args.account else ""

    replacements = {
        "__JOB_NAME__": args.name,
        "__JOB_OUT__": str(job_dir),
        "__PARTITION__": args.partition,
        "__MEMORY__": str(args.memory),
        "__CONSTRAINT__": args.constraint,
        "__TIME_LIMIT__": args.time_limit,
        "__ACCOUNT_LINE__": account_line,
        "__REPO_DIR__": q(Path(args.repo_dir).expanduser()),
        "__DATA_DIR__": q(Path(args.data_dir).expanduser()),
        "__N_EVENTS__": str(args.n_events),
        "__SAMPLE_STEPS__": str(args.sample_steps),
        "__BATCH_SIZES__": q(args.batch_sizes),
        "__AUTO_PREPARE_DUMMY__": str(args.auto_prepare_dummy),
        "__PHOTON_CONFIG__": q(args.photon_config),
        "__PION_CONFIG__": q(args.pion_config),
        "__PHOTON_MODEL__": q(args.photon_model),
        "__PION_MODEL__": q(args.pion_model),
        "__CONDA_ENV__": q(args.conda_env),
        "__OUTPUT_ROOT__": q(Path(args.output_root).expanduser()),
    }

    script = template
    for key, value in replacements.items():
        script = script.replace(key, value)

    script_path = job_dir / "run_measurement.slurm"
    script_path.write_text(script, encoding="ascii")
    script_path.chmod(0o755)

    print(f"Created: {script_path}")
    print(f"Output directory: {job_dir}")

    if args.dry_run:
        print("Dry run requested; skipping sbatch.")
        return 0

    cmd = ["sbatch", str(script_path)]
    print("Submitting:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
