#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  ./measurement.sh DATA_DIR [N_EVENTS] [SAMPLE_STEPS]

Purpose:
  Benchmark inference wall time + GPU memory for both dataset1 photon and pion
  showers across batch sizes.

Environment overrides:
  PHOTON_CONFIG        (default: calodiffusion/configs/config_dataset1_photon.json)
  PION_CONFIG          (default: calodiffusion/configs/config_dataset1_pion.json)
  PHOTON_MODEL         (default: auto-create dummy if missing and AUTO_PREPARE_DUMMY=1)
  PION_MODEL           (default: auto-create dummy if missing and AUTO_PREPARE_DUMMY=1)
  BATCH_SIZES          (default: "1 10 100 1000")
  AUTO_PREPARE_DUMMY   (default: 1; create synthetic data/binning/dummy models as needed)
  WORK_DIR             (default: /pscratch/$USER/calodiff_measurements if available, else ./measurement_artifacts)
  LOGFILE              (default: $WORK_DIR/inference_measurement_<timestamp>.log)
EOF
    exit 0
fi

DATA_DIR="${1:-$ROOT_DIR/data}"
N_EVENTS="${2:-1000}"
SAMPLE_STEPS="${3:-200}"

DEFAULT_WORK_DIR="$ROOT_DIR/measurement_artifacts"
if [[ -d "/pscratch" && -n "${USER:-}" ]]; then
    DEFAULT_WORK_DIR="/pscratch/${USER}/calodiff_measurements"
fi
WORK_DIR="${WORK_DIR:-$DEFAULT_WORK_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGFILE="${LOGFILE:-$WORK_DIR/inference_measurement_${TIMESTAMP}.log}"
BATCH_SIZES="${BATCH_SIZES:-1 10 100 1000}"
AUTO_PREPARE_DUMMY="${AUTO_PREPARE_DUMMY:-1}"

PHOTON_CONFIG="${PHOTON_CONFIG:-$ROOT_DIR/calodiffusion/configs/config_dataset1_photon.json}"
PION_CONFIG="${PION_CONFIG:-$ROOT_DIR/calodiffusion/configs/config_dataset1_pion.json}"
PHOTON_MODEL="${PHOTON_MODEL:-}"
PION_MODEL="${PION_MODEL:-}"

mkdir -p "$WORK_DIR"
mkdir -p "$DATA_DIR"

PYTHON_CMD=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "ERROR: neither python3 nor python is available in PATH." >&2
    exit 1
fi

INFER_CMD=()
if command -v calodif-inference >/dev/null 2>&1; then
    INFER_CMD=(calodif-inference)
else
    INFER_CMD=(python -m calodiffusion.inference)
fi

PREP_VARS="$(
"$PYTHON_CMD" - "$ROOT_DIR" "$DATA_DIR" "$WORK_DIR" "$N_EVENTS" "$AUTO_PREPARE_DUMMY" \
    "$PHOTON_CONFIG" "$PION_CONFIG" "$PHOTON_MODEL" "$PION_MODEL" <<'PY'
import os
import shlex
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import h5py
import numpy as np
import torch
import yaml

from calodiffusion.models.calodiffusion import CaloDiffusion


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def create_synthetic_h5(path: Path, n_events: int, n_bins: int):
    rng = np.random.default_rng(7)
    showers = rng.uniform(1e-6, 1.0, size=(n_events, n_bins)).astype(np.float32)
    energies = rng.uniform(1.0, 4000.0, size=(n_events, 1)).astype(np.float32)
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("showers", data=showers)
        h5f.create_dataset("incident_energies", data=energies)


def create_dummy_binning(path: Path, particle: str, total_bins: int, n_layers: int):
    layer_bins = [total_bins // n_layers] * n_layers
    for i in range(total_bins % n_layers):
        layer_bins[i] += 1

    lines = ["<binning>", f'  <particle name="{particle}">']
    for i, nbin in enumerate(layer_bins):
        r_edges = ",".join(str(j) for j in range(nbin + 1))
        lines.append(f'    <layer id="{i}" r_edges="{r_edges}" n_bin_alpha="1" />')
    lines.append("  </particle>")
    lines.append("</binning>")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def create_dummy_model(path: Path, cfg: dict):
    model = CaloDiffusion(cfg, n_steps=cfg["NSTEPS"], loss_type=cfg["LOSS_TYPE"])
    checkpoint = {"model_state_dict": model.state_dict()}
    torch.save(checkpoint, path)


def prepare_case(
    particle: str,
    base_config: Path,
    model_override: str,
    data_dir: Path,
    work_dir: Path,
    n_events: int,
    auto_prepare: bool,
):
    if not base_config.exists():
        raise FileNotFoundError(f"{particle} config not found: {base_config}")

    cfg = yaml.safe_load(base_config.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"{particle} config could not be parsed as dictionary: {base_config}")

    shape_orig = as_list(cfg.get("SHAPE_ORIG"))
    shape_final = as_list(cfg.get("SHAPE_FINAL"))
    if len(shape_orig) < 2:
        raise ValueError(f"{particle} config missing SHAPE_ORIG: {base_config}")
    if len(shape_final) < 3:
        raise ValueError(f"{particle} config missing SHAPE_FINAL: {base_config}")

    n_bins = int(shape_orig[-1])
    n_layers = int(shape_final[2])

    eval_entries = as_list(cfg.get("EVAL"))
    eval_name = eval_entries[0] if eval_entries else f"dataset_1_{particle}_measurement.hdf5"
    if not str(eval_name).endswith((".h5", ".hdf5")):
        eval_name = f"dataset_1_{particle}_measurement.hdf5"
    eval_name = os.path.basename(eval_name)
    eval_path = data_dir / eval_name

    if not eval_path.exists():
        if not auto_prepare:
            raise FileNotFoundError(
                f"{particle} eval data not found at {eval_path}. "
                "Set AUTO_PREPARE_DUMMY=1 or provide data."
            )
        create_synthetic_h5(eval_path, n_events=n_events, n_bins=n_bins)

    cfg["EVAL"] = [eval_name]
    cfg["FILES"] = [eval_name]
    cfg["PART_TYPE"] = particle

    bin_file = cfg.get("BIN_FILE")
    bin_file_ok = isinstance(bin_file, str) and os.path.exists(bin_file)
    if not bin_file_ok:
        if not auto_prepare:
            raise FileNotFoundError(
                f"{particle} BIN_FILE is missing/unreadable ({bin_file}). "
                "Set AUTO_PREPARE_DUMMY=1 or provide a valid binning XML."
            )
        bin_dir = work_dir / "binning"
        bin_dir.mkdir(parents=True, exist_ok=True)
        dummy_bin = bin_dir / f"binning_dataset1_{particle}_dummy.xml"
        create_dummy_binning(dummy_bin, particle=particle, total_bins=n_bins, n_layers=n_layers)
        cfg["BIN_FILE"] = str(dummy_bin)

    cfg_dir = work_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    prepared_cfg = cfg_dir / f"{particle}_measurement.yaml"
    prepared_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="ascii")

    model_path = Path(model_override).expanduser() if model_override else None
    if model_path is not None and model_path.exists():
        final_model = model_path
    else:
        if model_path is not None and not model_path.exists() and not auto_prepare:
            raise FileNotFoundError(
                f"{particle} model checkpoint not found: {model_path}"
            )
        if not auto_prepare:
            raise ValueError(
                f"{particle} model missing and AUTO_PREPARE_DUMMY=0."
            )
        model_dir = work_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        final_model = model_dir / f"{particle}_dummy_model.pth"
        if not final_model.exists():
            create_dummy_model(final_model, cfg)

    return prepared_cfg, final_model, eval_path, cfg["BIN_FILE"]


def emit(name: str, value: str):
    print(f"{name}={shlex.quote(str(value))}")


root_dir = Path(sys.argv[1])
data_dir = Path(sys.argv[2])
work_dir = Path(sys.argv[3])
n_events = int(sys.argv[4])
auto_prepare = bool(int(sys.argv[5]))

photon_cfg = Path(sys.argv[6]).expanduser()
pion_cfg = Path(sys.argv[7]).expanduser()
photon_model_override = sys.argv[8]
pion_model_override = sys.argv[9]

prepared_photon = prepare_case(
    particle="photon",
    base_config=photon_cfg,
    model_override=photon_model_override,
    data_dir=data_dir,
    work_dir=work_dir,
    n_events=n_events,
    auto_prepare=auto_prepare,
)
prepared_pion = prepare_case(
    particle="pion",
    base_config=pion_cfg,
    model_override=pion_model_override,
    data_dir=data_dir,
    work_dir=work_dir,
    n_events=n_events,
    auto_prepare=auto_prepare,
)

emit("PHOTON_CONFIG_PREP", prepared_photon[0])
emit("PHOTON_MODEL_PREP", prepared_photon[1])
emit("PHOTON_DATA_USED", prepared_photon[2])
emit("PHOTON_BIN_USED", prepared_photon[3])
emit("PION_CONFIG_PREP", prepared_pion[0])
emit("PION_MODEL_PREP", prepared_pion[1])
emit("PION_DATA_USED", prepared_pion[2])
emit("PION_BIN_USED", prepared_pion[3])
PY
)"

eval "$PREP_VARS"

HAS_GPU_MONITOR=0
if command -v nvidia-smi >/dev/null 2>&1; then
    HAS_GPU_MONITOR=1
fi

get_gpu_mem_mib() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d '[:space:]'
}

log() {
    echo "$1" | tee -a "$LOGFILE"
}

log "Inference Measurement"
log "Timestamp: $TIMESTAMP"
log "Data dir: $DATA_DIR"
log "N events: $N_EVENTS"
log "Sample steps: $SAMPLE_STEPS"
log "Batch sizes: $BATCH_SIZES"
log "Auto dummy prep: $AUTO_PREPARE_DUMMY"
log "Inference command: ${INFER_CMD[*]}"
log "Photon config: $PHOTON_CONFIG_PREP"
log "Photon model: $PHOTON_MODEL_PREP"
log "Photon data: $PHOTON_DATA_USED"
log "Photon binning: $PHOTON_BIN_USED"
log "Pion config: $PION_CONFIG_PREP"
log "Pion model: $PION_MODEL_PREP"
log "Pion data: $PION_DATA_USED"
log "Pion binning: $PION_BIN_USED"
log "========================================"

measure_particle() {
    local particle="$1"
    local config="$2"
    local model="$3"

    log ""
    log "########## $particle ##########"

    for BS in $BATCH_SIZES; do
        local run_log="$WORK_DIR/${particle}_bs${BS}_${TIMESTAMP}.cmd.log"
        local generated_h5="$WORK_DIR/generated_${particle}_bs${BS}_${TIMESTAMP}.h5"

        log ""
        log "--- $particle | batch size: $BS ---"

        local baseline="N/A"
        local peak="N/A"
        if [[ "$HAS_GPU_MONITOR" -eq 1 ]]; then
            baseline="$(get_gpu_mem_mib || true)"
            if [[ -z "$baseline" || ! "$baseline" =~ ^[0-9]+$ ]]; then
                baseline="N/A"
            fi
            peak="$baseline"
            log "Baseline GPU memory: ${baseline} MiB"
        else
            log "Baseline GPU memory: unavailable (nvidia-smi not found)"
        fi

        local start_ns
        local end_ns
        start_ns="$(date +%s%N)"

        "${INFER_CMD[@]}" \
            -c "$config" \
            -d "$DATA_DIR" \
            -n "$N_EVENTS" \
            sample \
            -g "$generated_h5" \
            --model-loc "$model" \
            --batch-size "$BS" \
            --sample-steps "$SAMPLE_STEPS" \
            diffusion >"$run_log" 2>&1 &
        local inf_pid=$!

        if [[ "$HAS_GPU_MONITOR" -eq 1 ]]; then
            while kill -0 "$inf_pid" 2>/dev/null; do
                local mem
                mem="$(get_gpu_mem_mib || true)"
                if [[ -n "$mem" && "$mem" =~ ^[0-9]+$ ]]; then
                    if [[ "$peak" =~ ^[0-9]+$ ]]; then
                        if (( mem > peak )); then
                            peak="$mem"
                        fi
                    else
                        peak="$mem"
                    fi
                fi
                sleep 0.5
            done
        fi

        local exit_code=0
        if wait "$inf_pid"; then
            exit_code=0
        else
            exit_code=$?
        fi
        end_ns="$(date +%s%N)"

        local elapsed_sec
        elapsed_sec="$("$PYTHON_CMD" - <<PY
start_ns = int("$start_ns")
end_ns = int("$end_ns")
print(f"{(end_ns - start_ns) / 1e9:.6f}")
PY
)"
        local ev_per_s
        ev_per_s="$("$PYTHON_CMD" - <<PY
n_events = float("$N_EVENTS")
elapsed = float("$elapsed_sec")
if elapsed <= 0:
    print("inf")
else:
    print(f"{n_events / elapsed:.3f}")
PY
)"

        log "Elapsed wall time: ${elapsed_sec} s"
        log "Throughput: ${ev_per_s} events/s"
        if [[ "$HAS_GPU_MONITOR" -eq 1 ]]; then
            log "Peak GPU memory: ${peak} MiB"
        else
            log "Peak GPU memory: unavailable (nvidia-smi not found)"
        fi
        log "Exit code: $exit_code"
        log "Command log: $run_log"

        if [[ "$exit_code" -ne 0 ]]; then
            log "Last 30 lines from failed run:"
            tail -n 30 "$run_log" | tee -a "$LOGFILE"
        fi
    done
}

measure_particle "photon" "$PHOTON_CONFIG_PREP" "$PHOTON_MODEL_PREP"
measure_particle "pion" "$PION_CONFIG_PREP" "$PION_MODEL_PREP"

log ""
log "========================================"
log "Results saved to $LOGFILE"
