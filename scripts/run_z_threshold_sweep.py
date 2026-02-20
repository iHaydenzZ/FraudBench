"""Run z-threshold sweep experiments for input validation analysis.

GPU configs = neural models (require CUDA for training + CAPGD gradients).
CPU configs = tree models (XGBoost trains on CPU, CAPGD is a no-op on trees).

Usage:
    uv run python scripts/run_z_threshold_sweep.py              # run all
    uv run python scripts/run_z_threshold_sweep.py --gpu-only   # neural only
    uv run python scripts/run_z_threshold_sweep.py --cpu-only   # tree only
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

# Only seed 42 for z-threshold sweep (as specified in todo)
SEEDS = [42]

# Neural models — require GPU for training + CAPGD
GPU_CONFIGS = [
    "configs/ccfd_input_val_z5.yaml",
    "configs/ccfd_input_val_z10.yaml",
    "configs/sparkov_input_val_z5.yaml",
    "configs/sparkov_input_val_z10.yaml",
]

# Tree models — CPU only (XGBoost + CAPGD no-op)
CPU_CONFIGS = [
    "configs/ccfd_tree_input_val_z5.yaml",
    "configs/ccfd_tree_input_val_z10.yaml",
    "configs/sparkov_tree_input_val_z5.yaml",
    "configs/sparkov_tree_input_val_z10.yaml",
]


def _short_name(config_path):
    return os.path.splitext(os.path.basename(config_path))[0]


def _run_one(config, seed):
    cmd = [sys.executable, "-m", "runner.run", "--config", config, "--seed", str(seed)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return config, seed, result.returncode, result.stderr


def main():
    parser = argparse.ArgumentParser(description="Z-threshold sweep batch runner")
    parser.add_argument("--cpu-only", action="store_true", help="Tree (CPU) configs only")
    parser.add_argument("--gpu-only", action="store_true", help="Neural (GPU) configs only")
    parser.add_argument("--workers", type=int, default=None,
                        help="Max parallel workers (default: 1 for GPU, 2 for CPU)")
    args = parser.parse_args()

    if args.cpu_only and args.gpu_only:
        parser.error("--cpu-only and --gpu-only are mutually exclusive")

    if args.cpu_only:
        configs = CPU_CONFIGS
        default_workers = 2
        label = "CPU (tree)"
    elif args.gpu_only:
        configs = GPU_CONFIGS
        default_workers = 1
        label = "GPU (neural)"
    else:
        configs = GPU_CONFIGS + CPU_CONFIGS
        default_workers = 1
        label = "All"

    max_workers = args.workers or default_workers
    experiments = [(config, seed) for config in configs for seed in SEEDS]
    total = len(experiments)
    failed = []
    start = time.time()

    print(f"Running {total} z-threshold experiments ({label}) with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one, c, s): (c, s) for c, s in experiments}
        pbar = tqdm(as_completed(futures), total=total, desc=label, unit="exp", dynamic_ncols=True)
        for future in pbar:
            config, seed, rc, stderr = future.result()
            pbar.set_postfix_str(f"{_short_name(config)} s{seed}")
            if rc != 0:
                print(f"\nFAILED: {_short_name(config)} s{seed}")
                if stderr:
                    print(stderr[-300:])
                failed.append((config, seed))

    elapsed = time.time() - start
    print(f"\nDone: {total - len(failed)}/{total} succeeded in {elapsed:.0f}s")
    if failed:
        print(f"Failed ({len(failed)}):")
        for config, seed in failed:
            print(f"  {config} --seed {seed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
