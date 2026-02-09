"""Run the full experiment matrix across 3 seeds."""
import subprocess
import sys
import time

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
]

def main():
    total = len(CONFIGS) * len(SEEDS)
    done = 0
    failed = []
    start = time.time()

    print(f"Running {total} experiments ({len(CONFIGS)} configs x {len(SEEDS)} seeds)")
    print("=" * 60)

    for config in CONFIGS:
        for seed in SEEDS:
            done += 1
            label = f"[{done}/{total}] {config} --seed {seed}"
            print(f"\n{label}")
            print("-" * 60)

            cmd = [sys.executable, "-m", "runner.run", "--config", config, "--seed", str(seed)]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"  FAILED (exit code {result.returncode})")
                print(result.stderr[-500:] if result.stderr else "no stderr")
                failed.append((config, seed))
            else:
                # Extract last few lines for summary
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
