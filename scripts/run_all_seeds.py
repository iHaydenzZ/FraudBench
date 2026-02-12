"""Run the full experiment matrix across 3 seeds."""
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

SEEDS = [42, 123, 456]

# All valid configs (excluding special/utility configs)
CONFIGS = [
    # CCFD: neural baselines + defences
    "configs/ccfd.yaml",
    "configs/ccfd_adv_train.yaml",
    "configs/ccfd_input_val.yaml",
    # CCFD: tree baselines + defences
    "configs/ccfd_tree.yaml",
    "configs/ccfd_tree_input_val.yaml",
    # IEEE-CIS: neural + defences
    "configs/ieee_cis_neural.yaml",
    "configs/ieee_cis_adv_train.yaml",
    "configs/ieee_cis_input_val.yaml",
    # IEEE-CIS: tree + defences
    "configs/ieee_cis.yaml",
    "configs/ieee_cis_tree_input_val.yaml",
    # LCLD: neural + defences
    "configs/lcld.yaml",
    "configs/lcld_adv_train.yaml",
    "configs/lcld_input_val.yaml",
    # LCLD: tree + defences
    "configs/lcld_tree.yaml",
    "configs/lcld_tree_input_val.yaml",
    # Sparkov: neural + defences
    "configs/sparkov.yaml",
    "configs/sparkov_adv_train.yaml",
    "configs/sparkov_input_val.yaml",
    # Sparkov: tree + defences
    "configs/sparkov_tree.yaml",
    "configs/sparkov_tree_input_val.yaml",
    # Phase 4: Black-box attacks (tree models)
    "configs/ccfd_tree_hsj.yaml",
    "configs/ccfd_tree_square.yaml",
    "configs/ieee_cis_tree_hsj.yaml",
    "configs/ieee_cis_tree_square.yaml",
    "configs/lcld_tree_hsj.yaml",
    "configs/lcld_tree_square.yaml",
    "configs/sparkov_tree_hsj.yaml",
    "configs/sparkov_tree_square.yaml",
    # Epsilon sweeps
    "configs/ccfd_eps_sweep.yaml",
    "configs/ieee_cis_eps_sweep.yaml",
    "configs/lcld_eps_sweep.yaml",
    "configs/sparkov_eps_sweep.yaml",
]

MAX_WORKERS = 8


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
        pbar = tqdm(as_completed(futures), total=total, desc="All", unit="exp", dynamic_ncols=True)
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
