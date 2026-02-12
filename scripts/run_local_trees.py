"""Run all tree model (CPU-only) experiments across 3 seeds.

Usage:
    uv run python scripts/run_local_trees.py
"""
import os
import subprocess
import sys
import time

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


def _short_name(config_path):
    """configs/ccfd_tree_hsj.yaml -> ccfd_tree_hsj"""
    return os.path.splitext(os.path.basename(config_path))[0]


def main():
    total = len(CONFIGS) * len(SEEDS)
    failed = []
    start = time.time()

    experiments = [(config, seed) for config in CONFIGS for seed in SEEDS]
    pbar = tqdm(experiments, desc="Trees", unit="exp", dynamic_ncols=True)

    for config, seed in pbar:
        pbar.set_postfix_str(f"{_short_name(config)} s{seed}")

        cmd = [sys.executable, "-m", "runner.run", "--config", config, "--seed", str(seed)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            failed.append((config, seed))

    elapsed = time.time() - start
    print(f"\nDone: {total - len(failed)}/{total} succeeded in {elapsed:.0f}s")
    if failed:
        print(f"Failed ({len(failed)}):")
        for config, seed in failed:
            print(f"  {config} --seed {seed}")


if __name__ == "__main__":
    main()
