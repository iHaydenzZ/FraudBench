"""Run the 9 remaining HopSkipJump experiments sequentially."""

import subprocess
import sys
import time

EXPERIMENTS = [
    ("configs/ieee_cis_tree_hsj.yaml", 42),
    ("configs/ieee_cis_tree_hsj.yaml", 123),
    ("configs/ieee_cis_tree_hsj.yaml", 456),
    ("configs/lcld_tree_hsj.yaml", 42),
    ("configs/lcld_tree_hsj.yaml", 123),
    ("configs/lcld_tree_hsj.yaml", 456),
    ("configs/sparkov_tree_hsj.yaml", 42),
    ("configs/sparkov_tree_hsj.yaml", 123),
    ("configs/sparkov_tree_hsj.yaml", 456),
]


def main():
    total = len(EXPERIMENTS)
    failed = []
    start = time.time()

    for i, (config, seed) in enumerate(EXPERIMENTS, 1):
        name = config.split("/")[-1].replace(".yaml", "")
        print(f"\n[{i}/{total}] {name} seed={seed}")
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, "-m", "runner.run", "--config", config, "--seed", str(seed)],
            text=True,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"  FAILED after {elapsed:.0f}s")
            failed.append((config, seed))
        else:
            print(f"  OK in {elapsed:.0f}s")

    total_time = time.time() - start
    print(f"\nDone: {total - len(failed)}/{total} succeeded in {total_time / 3600:.1f}h")
    if failed:
        print("Failed:")
        for config, seed in failed:
            print(f"  {config} --seed {seed}")


if __name__ == "__main__":
    main()
