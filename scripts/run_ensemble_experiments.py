"""Run ensemble experiments (4 datasets x 3 seeds x 2 attack types = 24 runs).

All ensemble experiments require GPU for MLP training. The GPU/CPU split here
separates CAPGD (fast, gradient-based) from Square Attack (slower, black-box)
so they can be scheduled independently.

Usage:
    uv run python scripts/run_ensemble_experiments.py              # run all
    uv run python scripts/run_ensemble_experiments.py --gpu-only   # CAPGD only
    uv run python scripts/run_ensemble_experiments.py --cpu-only   # Square only
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

SEEDS = [42, 123, 456]

# CAPGD configs — white-box attack via MLP gradients (fast, requires GPU)
GPU_CONFIGS = [
    "configs/ccfd_ensemble.yaml",
    "configs/ieee_cis_ensemble.yaml",
    "configs/lcld_ensemble.yaml",
    "configs/sparkov_ensemble.yaml",
]

# Square Attack configs — black-box attack (slower, still needs GPU for training)
CPU_CONFIGS = [
    "configs/ccfd_ensemble_square.yaml",
    "configs/ieee_cis_ensemble_square.yaml",
    "configs/lcld_ensemble_square.yaml",
    "configs/sparkov_ensemble_square.yaml",
]


def _short_name(config_path):
    return os.path.splitext(os.path.basename(config_path))[0]


def _run_one(config, seed):
    cmd = [sys.executable, "-m", "runner.run", "--config", config, "--seed", str(seed)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return config, seed, result.returncode, result.stderr


def main():
    parser = argparse.ArgumentParser(description="Ensemble experiment batch runner")
    parser.add_argument("--cpu-only", action="store_true", help="Square Attack configs only")
    parser.add_argument("--gpu-only", action="store_true", help="CAPGD configs only")
    parser.add_argument("--workers", type=int, default=None, help="Max parallel workers (default: 1)")
    args = parser.parse_args()

    if args.cpu_only and args.gpu_only:
        parser.error("--cpu-only and --gpu-only are mutually exclusive")

    if args.cpu_only:
        configs = CPU_CONFIGS
        label = "Square"
    elif args.gpu_only:
        configs = GPU_CONFIGS
        label = "CAPGD"
    else:
        configs = GPU_CONFIGS + CPU_CONFIGS
        label = "All"

    max_workers = args.workers or 1  # default sequential (GPU memory)
    experiments = [(config, seed) for config in configs for seed in SEEDS]
    total = len(experiments)
    failed = []
    start = time.time()

    print(f"Running {total} ensemble experiments ({label}) with {max_workers} workers...")

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
