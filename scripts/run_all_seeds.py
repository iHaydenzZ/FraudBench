"""Run the full experiment matrix across 3 seeds.

Usage:
    uv run python scripts/run_all_seeds.py              # run all
    uv run python scripts/run_all_seeds.py --cpu-only   # tree models only
    uv run python scripts/run_all_seeds.py --gpu-only   # neural models only
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

SEEDS = [42, 123, 456]

GPU_CONFIGS = [
    # Neural models (require GPU for training + CAPGD)
    "configs/ccfd.yaml",
    "configs/ccfd_adv_train.yaml",
    "configs/ccfd_input_val.yaml",
    "configs/ccfd_eps_sweep.yaml",
    "configs/ieee_cis_neural.yaml",
    "configs/ieee_cis_adv_train.yaml",
    "configs/ieee_cis_input_val.yaml",
    "configs/ieee_cis_eps_sweep.yaml",
    "configs/lcld.yaml",
    "configs/lcld_adv_train.yaml",
    "configs/lcld_input_val.yaml",
    "configs/lcld_eps_sweep.yaml",
    "configs/sparkov.yaml",
    "configs/sparkov_adv_train.yaml",
    "configs/sparkov_input_val.yaml",
    "configs/sparkov_eps_sweep.yaml",
]

CPU_CONFIGS = [
    # Tree models (CPU only -- XGBoost + black-box attacks)
    "configs/ccfd_tree.yaml",
    "configs/ccfd_tree_input_val.yaml",
    "configs/ccfd_tree_hsj.yaml",
    "configs/ccfd_tree_square.yaml",
    "configs/ieee_cis.yaml",
    "configs/ieee_cis_tree_input_val.yaml",
    "configs/ieee_cis_tree_hsj.yaml",
    "configs/ieee_cis_tree_square.yaml",
    "configs/lcld_tree.yaml",
    "configs/lcld_tree_input_val.yaml",
    "configs/lcld_tree_hsj.yaml",
    "configs/lcld_tree_square.yaml",
    "configs/sparkov_tree.yaml",
    "configs/sparkov_tree_input_val.yaml",
    "configs/sparkov_tree_hsj.yaml",
    "configs/sparkov_tree_square.yaml",
]


def _short_name(config_path):
    """configs/ccfd_tree_hsj.yaml -> ccfd_tree_hsj"""
    return os.path.splitext(os.path.basename(config_path))[0]


def _run_one(config, seed):
    """Run a single experiment subprocess. Returns (config, seed, returncode, stderr)."""
    cmd = [sys.executable, "-m", "runner.run", "--config", config, "--seed", str(seed)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return config, seed, result.returncode, result.stderr


def main():
    parser = argparse.ArgumentParser(description="Batch experiment runner")
    parser.add_argument("--cpu-only", action="store_true", help="Run only CPU (tree) experiments")
    parser.add_argument("--gpu-only", action="store_true", help="Run only GPU (neural) experiments")
    parser.add_argument(
        "--workers", type=int, default=None, help="Max parallel workers (default: 2 for CPU, 4 for GPU)"
    )
    args = parser.parse_args()

    if args.cpu_only:
        configs = CPU_CONFIGS
        default_workers = 2  # each XGBoost uses n_jobs=-1 (all cores)
        label = "CPU"
    elif args.gpu_only:
        configs = GPU_CONFIGS
        default_workers = 4
        label = "GPU"
    else:
        configs = GPU_CONFIGS + CPU_CONFIGS
        default_workers = 2
        label = "All"

    max_workers = args.workers or default_workers
    experiments = [(config, seed) for config in configs for seed in SEEDS]
    total = len(experiments)
    failed = []
    start = time.time()

    print(f"Running {total} experiments ({label}) with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one, config, seed): (config, seed) for config, seed in experiments}
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


if __name__ == "__main__":
    main()
