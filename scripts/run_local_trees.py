"""Run all tree model (CPU-only) experiments across 3 seeds.

Usage:
    uv run python scripts/run_local_trees.py
"""
import subprocess
import sys
import time

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


def main():
    total = len(CONFIGS) * len(SEEDS)
    done = 0
    failed = []
    start = time.time()

    print(f"Running {total} tree experiments ({len(CONFIGS)} configs x {len(SEEDS)} seeds)")
    print("=" * 60)

    for config in CONFIGS:
        for seed in SEEDS:
            done += 1
            print(f"\n[{done}/{total}] {config} --seed {seed}")
            print("-" * 60)

            cmd = [sys.executable, "-m", "runner.run", "--config", config, "--seed", str(seed)]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"  FAILED (exit code {result.returncode})")
                print(result.stderr[-500:] if result.stderr else "no stderr")
                failed.append((config, seed))
            else:
                lines = result.stdout.strip().split("\n")
                for line in lines[-4:]:
                    print(f"  {line.strip()}")

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"Done: {done - len(failed)}/{total} succeeded in {elapsed:.0f}s")
    if failed:
        print(f"Failed ({len(failed)}):")
        for config, seed in failed:
            print(f"  {config} --seed {seed}")


if __name__ == "__main__":
    main()
