"""Run the remaining HSJ experiments with 2 parallel workers.

Uses reduced HSJ parameters (max_iter=5, max_eval=200, init_eval=50)
vs the original CCFD runs (max_iter=20, max_eval=1000, init_eval=100).

Runs 2 experiments in parallel, each pinned to 8 cores (via OMP/MKL env vars
and XGBoost nthread override) to avoid CPU contention on Ryzen 9 7940HX.
"""

import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

EXPERIMENTS = [
    # 3 fast non-HSJ experiments (already completed)
    # ("configs/ieee_cis_tree_input_val.yaml", 42),
    # ("configs/ieee_cis_tree_input_val.yaml", 123),
    # ("configs/lcld_tree_square.yaml", 123),
    # ieee_cis s42 already completed
    # ("configs/ieee_cis_tree_hsj.yaml", 42),
    ("configs/ieee_cis_tree_hsj.yaml", 123),
    ("configs/ieee_cis_tree_hsj.yaml", 456),
    ("configs/lcld_tree_hsj.yaml", 42),
    ("configs/lcld_tree_hsj.yaml", 123),
    ("configs/lcld_tree_hsj.yaml", 456),
    ("configs/sparkov_tree_hsj.yaml", 42),
    ("configs/sparkov_tree_hsj.yaml", 123),
    ("configs/sparkov_tree_hsj.yaml", 456),
]

MAX_WORKERS = 3
CORES_PER_WORKER = 8  # 16 cores, 3 workers x 8 threads (24/32 logical threads)


def _short_name(config_path):
    return os.path.splitext(os.path.basename(config_path))[0]


def _run_one(config, seed):
    """Run a single experiment with capped core usage."""
    env = os.environ.copy()
    env["XGBOOST_NTHREADS"] = str(CORES_PER_WORKER)
    env["OMP_NUM_THREADS"] = str(CORES_PER_WORKER)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "runner.run", "--config", config, "--seed", str(seed)],
        capture_output=True,
        text=True,
        env=env,
    )
    return config, seed, result.returncode, time.time() - t0, result.stderr


def main():
    total = len(EXPERIMENTS)
    failed = []
    start = time.time()

    print(f"Running {total} experiments with {MAX_WORKERS} workers x {CORES_PER_WORKER} cores each...")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_one, config, seed): (config, seed) for config, seed in EXPERIMENTS}
        done_count = 0
        for future in as_completed(futures):
            config, seed, rc, elapsed, stderr = future.result()
            done_count += 1
            name = _short_name(config)
            if rc != 0:
                print(f"[{done_count}/{total}] FAILED  {name} s{seed} ({elapsed:.0f}s)")
                if stderr:
                    print(stderr[-300:])
                failed.append((config, seed))
            else:
                print(f"[{done_count}/{total}] OK      {name} s{seed} ({elapsed:.0f}s)")

    total_time = time.time() - start
    print(f"\nDone: {total - len(failed)}/{total} succeeded in {total_time / 3600:.1f}h")
    if failed:
        print("Failed:")
        for config, seed in failed:
            print(f"  {config} --seed {seed}")


if __name__ == "__main__":
    main()
