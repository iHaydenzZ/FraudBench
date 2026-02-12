"""Run all tree model (CPU-only) experiments across 3 seeds.

Usage:
    uv run python scripts/run_local_trees.py
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

SEEDS = [42, 123, 456]

CONFIGS = [
    # CCFD tree
    "configs/ccfd_tree.yaml",
    "configs/ccfd_tree_input_val.yaml",
    "configs/ccfd_tree_hsj.yaml",
    "configs/ccfd_tree_square.yaml",
    # IEEE-CIS tree
    "configs/ieee_cis.yaml",
    "configs/ieee_cis_tree_input_val.yaml",
    "configs/ieee_cis_tree_hsj.yaml",
    "configs/ieee_cis_tree_square.yaml",
    # LCLD tree
    "configs/lcld_tree.yaml",
    "configs/lcld_tree_input_val.yaml",
    "configs/lcld_tree_hsj.yaml",
    "configs/lcld_tree_square.yaml",
    # Sparkov tree
    "configs/sparkov_tree.yaml",
    "configs/sparkov_tree_input_val.yaml",
    "configs/sparkov_tree_hsj.yaml",
    "configs/sparkov_tree_square.yaml",
]

MAX_WORKERS = 8  # half of 16 cores — leaves headroom for OS


def _short_name(config_path):
    """configs/ccfd_tree_hsj.yaml -> ccfd_tree_hsj"""
    return os.path.splitext(os.path.basename(config_path))[0]


def _run_one(config, seed):
    """Run a single experiment subprocess. Returns (config, seed, returncode)."""
    cmd = [sys.executable, "-m", "runner.run", "--config", config, "--seed", str(seed)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return config, seed, result.returncode


def main():
    experiments = [(config, seed) for config in CONFIGS for seed in SEEDS]
    total = len(experiments)
    failed = []
    start = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_run_one, config, seed): (config, seed)
            for config, seed in experiments
        }
        pbar = tqdm(as_completed(futures), total=total, desc="Trees", unit="exp", dynamic_ncols=True)
        for future in pbar:
            config, seed, rc = future.result()
            pbar.set_postfix_str(f"{_short_name(config)} s{seed}")
            if rc != 0:
                failed.append((config, seed))

    elapsed = time.time() - start
    print(f"\nDone: {total - len(failed)}/{total} succeeded in {elapsed:.0f}s")
    if failed:
        print(f"Failed ({len(failed)}):")
        for config, seed in failed:
            print(f"  {config} --seed {seed}")


if __name__ == "__main__":
    main()
