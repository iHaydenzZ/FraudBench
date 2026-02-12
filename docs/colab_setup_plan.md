# FraudBench: Google Colab Integration Setup Plan

## Context

This plan is for **Claude Code** to execute on a local machine. The goal is to restructure the existing FraudBench (FRBS) capstone project so that:

1. All development happens locally with proper Python package structure.
2. Google Colab is used **only** as a GPU runtime to run experiments.
3. Results are persisted to Google Drive so they survive Colab session disconnects.
4. The student has **100 Colab compute units** — efficiency is critical.

The existing project is a **Capstone MVP** repository called `Capstone_MVP` (or similar). It already has a working structure under an `MVP/` directory. **Do not rewrite existing code** — restructure and add Colab integration layers on top.

---

## Current Repository Structure (DO NOT DELETE)

The existing project looks like this:

```
Capstone_MVP/   (or root of the repo)
└── MVP/
    ├── attacks/          # CAPGD attack implementation
    ├── configs/          # Experiment configs (mvp.yaml, ccfd.yaml, ieee_cis.yaml)
    ├── constraints/      # Feature constraint schema and validation
    ├── datasets/         # Dataset loaders (CCFD, IEEE-CIS, LCLD, Sparkov)
    │   └── cards/        # Dataset documentation
    ├── defences/         # Adversarial training, input validation
    ├── evaluation/       # Metrics (PR-AUC, recall, F1) and results registry
    ├── models/           # Tree (XGBoost) and Neural (MLP) wrappers
    ├── preprocessing/    # Data preprocessing pipeline with caching
    ├── results/          # Experiment results, cached splits/preprocessors
    ├── runner/           # CLI entrypoint
    ├── tests/            # Pytest test suite (27 tests)
    ├── CLAUDE.md         # Claude Code guidelines
    ├── AGENTS.md         # Repository guidelines
    ├── README.md
    ├── pyproject.toml    # (or setup.py)
    └── ...
```

The project uses **`uv`** as its package manager. Existing commands:

```bash
uv sync                                              # Install dependencies
uv run python -m runner.run --config configs/mvp.yaml  # Run experiment
uv run pytest tests/ -v                              # Run all 27 tests
```

---

## Task 1: Verify and Fix Package Installability

**Goal:** Ensure the project can be installed as an editable Python package via `pip install -e .` (required for Colab, which does not have `uv`).

### Steps

1. **Check `pyproject.toml` (or `setup.py`)** at the repo root. If it does not exist, create one. It must:
   - Define `name = "fraudbench"` (or the current package name).
   - List all dependencies that are currently in `pyproject.toml` or `requirements.txt`.
   - Set `packages` to find the `MVP` directory or its sub-packages.

2. **Ensure the MVP directory is importable as a Python package.** Check that:
   - `MVP/__init__.py` exists (create if missing — can be empty).
   - All subdirectories (`attacks/`, `configs/`, `constraints/`, `datasets/`, `defences/`, `evaluation/`, `models/`, `preprocessing/`, `runner/`) have `__init__.py` files.

3. **Test local editable install:**
   ```bash
   pip install -e . --break-system-packages
   # Then verify imports work:
   python -c "from MVP.datasets.loader import load_dataset; print('Import OK')"
   python -c "from MVP.runner import run; print('Runner OK')"
   ```
   If the import path is different (e.g., `from runner.run import ...` instead of `from MVP.runner.run import ...`), document the actual import path and adjust the Colab notebooks accordingly.

4. **Create `requirements-colab.txt`** — a flat requirements file for Colab (since Colab cannot use `uv`). Extract all dependencies from `pyproject.toml`:
   ```
   # requirements-colab.txt
   # Core ML
   torch>=2.0
   scikit-learn>=1.3
   xgboost>=2.0
   
   # Adversarial Robustness
   adversarial-robustness-toolbox>=1.17
   
   # Data
   pandas>=2.0
   numpy>=1.24
   
   # Config
   pyyaml>=6.0
   
   # Evaluation
   matplotlib>=3.7
   seaborn>=0.12
   ```
   > Adjust version numbers based on what is currently in the project's dependency spec.

---

## Task 2: Create the `notebooks/` Directory with Colab Notebooks

**Goal:** Create ready-to-use Colab notebooks that clone the repo, install dependencies, mount Google Drive, and run experiments.

### 2.1 Create directory

```bash
mkdir -p notebooks
```

### 2.2 Create `notebooks/colab_runner.ipynb`

This is the **main experiment runner notebook**. Create it as a Jupyter notebook (`.ipynb` JSON format) with the following cells:

---

**Cell 1 — Header (Markdown)**

```markdown
# FraudBench Experiment Runner (Colab)

This notebook runs FraudBench experiments using Colab GPU.

**Workflow:**
1. Mount Google Drive for data and results persistence
2. Clone/update the repo
3. Install dependencies
4. Run experiments via config files
5. Results auto-save to Google Drive
```

---

**Cell 2 — GPU Check (Code)**

```python
# Verify GPU is available (disconnect and switch runtime if not)
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    print("WARNING: No GPU detected. Go to Runtime > Change runtime type > GPU")
```

---

**Cell 3 — Mount Google Drive (Code)**

```python
# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Create project directories on Drive (only needed once)
import os
DRIVE_ROOT = "/content/drive/MyDrive/FraudBench"
for subdir in ["data", "results", "models", "logs"]:
    os.makedirs(os.path.join(DRIVE_ROOT, subdir), exist_ok=True)
    print(f"  {DRIVE_ROOT}/{subdir}/")

print("\nGoogle Drive mounted and directories ready.")
```

---

**Cell 4 — Clone or Update Repo (Code)**

```python
# Clone repo (first time) or pull latest changes
import os

REPO_URL = "https://github.com/YOUR_USERNAME/Capstone_MVP.git"  # UPDATE THIS
REPO_DIR = "/content/Capstone_MVP"

if os.path.exists(REPO_DIR):
    print("Repo exists. Pulling latest changes...")
    os.chdir(REPO_DIR)
    !git pull
else:
    print("Cloning repo...")
    !git clone {REPO_URL} {REPO_DIR}
    os.chdir(REPO_DIR)

print(f"\nWorking directory: {os.getcwd()}")
!git log --oneline -3
```

---

**Cell 5 — Install Dependencies (Code)**

```python
# Install project dependencies (only needs to run once per session)
!pip install -e . -q 2>&1 | tail -5

# If the above fails, use the flat requirements file:
# !pip install -r requirements-colab.txt -q
```

---

**Cell 6 — Symlink Data from Google Drive (Code)**

```python
# Link datasets from Google Drive to the project's expected data path
# This avoids re-uploading data every session
import os

DRIVE_DATA = "/content/drive/MyDrive/FraudBench/data"
# Adjust LOCAL_DATA to match the path in your dataset loader (datasets/loader.py DEFAULT_DATA_ROOT)
LOCAL_DATA = "/content/Capstone_MVP/data"  # UPDATE if different

# Remove existing symlink/directory if present
if os.path.islink(LOCAL_DATA):
    os.unlink(LOCAL_DATA)
elif os.path.isdir(LOCAL_DATA):
    import shutil
    shutil.rmtree(LOCAL_DATA)

os.symlink(DRIVE_DATA, LOCAL_DATA)
print(f"Linked: {LOCAL_DATA} -> {DRIVE_DATA}")

# Verify datasets are accessible
for d in os.listdir(DRIVE_DATA):
    full = os.path.join(DRIVE_DATA, d)
    if os.path.isdir(full):
        files = os.listdir(full)
        print(f"  {d}/ ({len(files)} files)")
```

---

**Cell 7 — Run Experiment (Code)**

```python
# Run a single experiment using a config file
# Change the config path to run different experiments

CONFIG = "MVP/configs/mvp.yaml"  # UPDATE to your target config

!python -m MVP.runner.run --config {CONFIG}

# If the above import path does not work, try:
# !cd MVP && python -m runner.run --config configs/mvp.yaml
```

---

**Cell 8 — Copy Results to Google Drive (Code)**

```python
# Save experiment results to Google Drive for persistence
import shutil
import glob

# Source: local results directory
RESULTS_SRC = "/content/Capstone_MVP/MVP/results"  # UPDATE if different
# Destination: Google Drive
RESULTS_DST = "/content/drive/MyDrive/FraudBench/results"

# Copy registry and all result files
for f in glob.glob(os.path.join(RESULTS_SRC, "*")):
    dst = os.path.join(RESULTS_DST, os.path.basename(f))
    if os.path.isfile(f):
        shutil.copy2(f, dst)
        print(f"Saved: {os.path.basename(f)}")

print(f"\nAll results backed up to {RESULTS_DST}")
```

---

**Cell 9 — Batch Runner (Code)**

```python
# Run multiple experiments sequentially
# Useful for running all MVP experiments in one session

configs = [
    "MVP/configs/ccfd.yaml",
    "MVP/configs/ccfd_input_val.yaml",
    "MVP/configs/ccfd_adv_train.yaml",
    "MVP/configs/ccfd_tree.yaml",
    # Add more configs as needed
]

import subprocess
import time

for i, config in enumerate(configs):
    print(f"\n{'='*60}")
    print(f"Experiment {i+1}/{len(configs)}: {config}")
    print(f"{'='*60}")
    
    start = time.time()
    result = subprocess.run(
        ["python", "-m", "MVP.runner.run", "--config", config],
        capture_output=True, text=True
    )
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"SUCCESS ({elapsed:.1f}s)")
        # Print last 5 lines of output for key metrics
        for line in result.stdout.strip().split('\n')[-5:]:
            print(f"  {line}")
    else:
        print(f"FAILED ({elapsed:.1f}s)")
        print(result.stderr[-500:] if result.stderr else "No error output")

print(f"\n{'='*60}")
print("All experiments complete. Run Cell 8 to save results to Drive.")
```

---

**Cell 10 — Session Cleanup (Code)**

```python
# IMPORTANT: Run this when you're done to stop consuming compute units!
# This disconnects the runtime and releases the GPU.

# First, make sure results are saved (run Cell 8 first!)
print("Make sure you've saved results to Google Drive (Cell 8) before disconnecting!")
print("To disconnect: Runtime > Disconnect and delete runtime")
print("Or uncomment the line below:")
# from google.colab import runtime; runtime.unassign()
```

---

### 2.3 Create `notebooks/colab_debug.ipynb`

A lightweight notebook for debugging WITHOUT GPU (saves compute units). Create with these cells:

**Cell 1 (Markdown):**
```markdown
# FraudBench Debug Notebook (No GPU)
Use this notebook for debugging with a small data sample. 
**Set runtime to CPU** to save compute units.
```

**Cell 2 (Code):**
```python
# Quick setup (same as runner but no GPU check)
from google.colab import drive
drive.mount('/content/drive')

import os
REPO_DIR = "/content/Capstone_MVP"
if os.path.exists(REPO_DIR):
    os.chdir(REPO_DIR)
    !git pull
else:
    !git clone https://github.com/YOUR_USERNAME/Capstone_MVP.git {REPO_DIR}
    os.chdir(REPO_DIR)

!pip install -e . -q
```

**Cell 3 (Code):**
```python
# Debug: test imports and data loading with small sample
import sys
sys.path.insert(0, '/content/Capstone_MVP')

from MVP.datasets.loader import load_dataset

# Load a small sample to verify everything works
# Adjust the function call based on your actual loader API
data = load_dataset("ccfd", config={"sample_frac": 0.01})
print(f"Loaded {len(data.X)} samples, {data.X.shape[1]} features")
print(f"Fraud rate: {data.y.mean():.4%}")
```

---

## Task 3: Create the Google Drive Data Upload Script

**Goal:** Provide a helper script the student can run locally to prepare datasets for Google Drive.

### Create `scripts/prepare_drive_data.py`

```python
"""
Prepare datasets for Google Drive upload.

Usage:
    python scripts/prepare_drive_data.py --data-dir /path/to/local/datasets --output-dir ./drive_upload

This creates a clean directory structure that can be drag-and-dropped into Google Drive.
"""
import argparse
import os
import shutil


# Expected dataset structure (must match datasets/loader.py)
DATASET_STRUCTURE = {
    "CCFD": ["creditcard.csv"],
    "IEEE-CIS": ["train_transaction.csv", "train_identity.csv"],
    "LCLD": [],   # Add expected files when known
    "Sparkov": ["fraudTrain.csv", "fraudTest.csv"],
}


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for Google Drive")
    parser.add_argument("--data-dir", required=True, help="Local directory containing raw datasets")
    parser.add_argument("--output-dir", default="./drive_upload", help="Output directory to upload to Drive")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for dataset_name, expected_files in DATASET_STRUCTURE.items():
        src_dir = os.path.join(args.data_dir, dataset_name)
        dst_dir = os.path.join(args.output_dir, dataset_name)

        if not os.path.exists(src_dir):
            print(f"  SKIP  {dataset_name}/ (not found at {src_dir})")
            continue

        os.makedirs(dst_dir, exist_ok=True)

        # Copy all CSV files from the source
        copied = 0
        for root, dirs, files in os.walk(src_dir):
            for f in files:
                if f.endswith('.csv'):
                    src_file = os.path.join(root, f)
                    dst_file = os.path.join(dst_dir, f)
                    shutil.copy2(src_file, dst_file)
                    size_mb = os.path.getsize(dst_file) / 1e6
                    print(f"  COPY  {dataset_name}/{f} ({size_mb:.1f} MB)")
                    copied += 1

        if copied == 0:
            print(f"  WARN  {dataset_name}/ — no CSV files found")

        # Check for expected files
        for ef in expected_files:
            if not os.path.exists(os.path.join(dst_dir, ef)):
                print(f"  MISS  {dataset_name}/{ef} — expected but not found")

    print(f"\nDone. Upload '{args.output_dir}/' to Google Drive > MyDrive > FraudBench > data")


if __name__ == "__main__":
    main()
```

---

## Task 4: Create the `.colab` Configuration Shortcut

### Create `notebooks/open_in_colab.md`

```markdown
# Opening Notebooks in Google Colab

## Quick Links (update after pushing to GitHub)

- **Experiment Runner**: 
  `https://colab.research.google.com/github/YOUR_USERNAME/Capstone_MVP/blob/main/notebooks/colab_runner.ipynb`

- **Debug Notebook**: 
  `https://colab.research.google.com/github/YOUR_USERNAME/Capstone_MVP/blob/main/notebooks/colab_debug.ipynb`

## Manual Steps

1. Go to [Google Colab](https://colab.research.google.com)
2. File > Open notebook > GitHub tab
3. Enter your repo URL
4. Select the notebook to open

## Google Drive Setup (One-Time)

Upload your datasets to Google Drive with this structure:

```
MyDrive/
└── FraudBench/
    ├── data/
    │   ├── CCFD/
    │   │   └── creditcard.csv
    │   ├── IEEE-CIS/
    │   │   ├── train_transaction.csv
    │   │   └── train_identity.csv
    │   ├── LCLD/
    │   │   └── ...
    │   └── Sparkov/
    │       ├── fraudTrain.csv
    │       └── fraudTest.csv
    ├── results/       (auto-created by notebook)
    ├── models/        (auto-created by notebook)
    └── logs/          (auto-created by notebook)
```

Use `scripts/prepare_drive_data.py` to create this structure locally, then upload.
```

---

## Task 5: Update `.gitignore`

Add the following entries to `.gitignore` (do not remove existing entries):

```gitignore
# Colab checkpoints
.ipynb_checkpoints/

# Data (should be on Google Drive, not in repo)
data/
*.csv

# Results and cached artifacts (should be on Google Drive)
results/*.joblib
results/*.json
results/split_indices_*
results/preprocessor_*

# Model weights
*.pt
*.pth
*.pkl
*.joblib

# OS
.DS_Store
__pycache__/
*.pyc
```

---

## Task 6: Update `CLAUDE.md` with Colab Workflow

Append the following section to the existing `CLAUDE.md` (or `MVP/CLAUDE.md`):

```markdown
## Colab Integration

### Development Workflow

1. **Local development**: Write and debug code locally using `uv run` and `uv run pytest`.
2. **Push to GitHub**: `git add . && git commit -m "..." && git push`
3. **Run on Colab**: Open `notebooks/colab_runner.ipynb`, pull latest code, run experiments.
4. **Results on Drive**: All experiment outputs are saved to Google Drive automatically.

### Important Notes

- Colab does NOT have `uv`. Use `pip install -e .` instead.
- The runner command on Colab may differ from local:
  - Local:  `uv run python -m runner.run --config configs/mvp.yaml`
  - Colab:  `python -m MVP.runner.run --config MVP/configs/mvp.yaml`
  - Or:     `cd MVP && python -m runner.run --config configs/mvp.yaml`
- Datasets live on Google Drive at `/content/drive/MyDrive/FraudBench/data/`
- Symlink from the local data path to Drive data (handled by notebook Cell 6).

### Compute Unit Budget

The student has ~100 Colab compute units. Approximate burn rates:
- T4 GPU: ~1.96 units/hour
- A100 GPU: ~12.46 units/hour
- CPU: Free (no unit consumption)

**Always use T4 unless A100 is specifically needed.** Debug on CPU runtime.
```

---

## Task 7: Verify Everything Works Locally

Run these checks to confirm the setup is correct before pushing:

```bash
# 1. Check all __init__.py files exist
find MVP -type d -exec sh -c 'test -f "$1/__init__.py" || echo "MISSING: $1/__init__.py"' _ {} \;

# 2. Verify editable install works
pip install -e . --break-system-packages
python -c "from MVP.datasets.loader import load_dataset; print('OK')"

# 3. Run existing tests (should still pass — nothing should break)
uv run pytest tests/ -v

# 4. Verify notebooks are valid JSON
python -c "import json; json.load(open('notebooks/colab_runner.ipynb')); print('Notebook OK')"

# 5. Check git status
git status
git diff --stat
```

---

## Task 8: Commit and Push

```bash
git add notebooks/ scripts/prepare_drive_data.py requirements-colab.txt .gitignore
git add -u  # Stage modifications to existing files (CLAUDE.md, pyproject.toml, __init__.py)
git commit -m "feat: add Colab integration notebooks and Drive workflow

- Add colab_runner.ipynb for GPU experiment execution
- Add colab_debug.ipynb for CPU debugging
- Add prepare_drive_data.py for Drive upload preparation
- Add requirements-colab.txt for Colab pip install
- Update CLAUDE.md with Colab workflow documentation
- Ensure all packages have __init__.py for pip install -e ."

git push origin main
```

---

## Summary of Files to Create/Modify

| Action | File | Purpose |
|--------|------|---------|
| CREATE | `notebooks/colab_runner.ipynb` | Main experiment runner for Colab |
| CREATE | `notebooks/colab_debug.ipynb` | Lightweight debug notebook (CPU) |
| CREATE | `notebooks/open_in_colab.md` | Quick-start guide with Colab links |
| CREATE | `scripts/prepare_drive_data.py` | Prepare datasets for Drive upload |
| CREATE | `requirements-colab.txt` | Flat deps file for Colab pip install |
| MODIFY | `pyproject.toml` (or `setup.py`) | Ensure editable install works |
| MODIFY | `MVP/__init__.py` | Create if missing |
| MODIFY | `MVP/*/__init__.py` | Create in all subdirectories if missing |
| MODIFY | `.gitignore` | Add Colab and data exclusions |
| MODIFY | `CLAUDE.md` | Add Colab workflow section |

---

## Critical Reminders

1. **DO NOT rewrite existing code.** Only add new files and make minimal modifications.
2. **Test that `uv run pytest tests/ -v` still passes** after all changes.
3. **The notebook import paths depend on the actual package structure.** If `runner.run` is invoked as `python -m runner.run` locally (from inside `MVP/`), then on Colab it might need `cd MVP && python -m runner.run` or `python -m MVP.runner.run` from the repo root. **Test both and document whichever works.**
4. **The `DEFAULT_DATA_ROOT` in `datasets/loader.py`** determines where the symlink should point. Check this value and adjust Cell 6 accordingly.
5. **Notebooks must be valid `.ipynb` JSON files**, not Python scripts. Use the `nbformat` library or create them as proper Jupyter notebook JSON.
