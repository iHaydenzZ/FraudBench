# MVP Supplement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the defence comparison gap by adding an Ensemble defence, deepening Input Validation analysis, adding Adversarial Training trade-off analysis, upgrading statistical tests, and generating final figures/docs.

**Architecture:** `EnsembleModel` (LogisticRegression + XGBoost + MLP with soft voting) extends `BaseModel`. Runner dispatches on `model.type == "ensemble"`. Analysis scripts read from `results/registry_clean.csv`. All new experiments log to the standard registry via `ExperimentRegistry`.

**Tech Stack:** PyTorch, scikit-learn, XGBoost, scipy.stats, pandas, matplotlib, seaborn

---

## Task 1: Write Failing Tests for EnsembleModel

**Files:**
- Create: `tests/test_ensemble.py`

**Step 1: Write test file with all EnsembleModel tests**

```python
"""Tests for ensemble defence (heterogeneous model ensemble)."""

import numpy as np
import pandas as pd
import os


def _make_dummy_data(n_samples=200, n_features=10, seed=42):
    """Create a small dummy dataset for testing."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.randint(0, 2, size=n_samples))
    return X, y


class TestEnsembleModel:
    """Tests for EnsembleModel (LR + XGBoost + MLP ensemble)."""

    def test_fit_and_predict_proba(self):
        """Ensemble trains all sub-models and returns valid probabilities."""
        from defences.ensemble import EnsembleModel

        X, y = _make_dummy_data()
        model = EnsembleModel({"epochs": 2, "hidden_dim": 8, "batch_size": 64})
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert probs.shape == (200,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_exposes_pytorch_model(self):
        """EnsembleModel.model is an nn.Module for CAPGD compatibility."""
        import torch.nn as nn
        from defences.ensemble import EnsembleModel

        X, y = _make_dummy_data()
        model = EnsembleModel({"epochs": 2, "hidden_dim": 8, "batch_size": 64})
        model.fit(X, y)

        assert hasattr(model, "model")
        assert isinstance(model.model, nn.Module)
        assert hasattr(model, "device")
        assert hasattr(model, "_use_logits")

    def test_save_load_roundtrip(self, tmp_path):
        """Save/load preserves predictions."""
        from defences.ensemble import EnsembleModel

        X, y = _make_dummy_data()
        model = EnsembleModel({"epochs": 2, "hidden_dim": 8, "batch_size": 64})
        model.fit(X, y)

        path = str(tmp_path / "ensemble.joblib")
        model.save(path)
        assert os.path.exists(path)

        loaded = EnsembleModel.load(path)
        preds_original = model.predict_proba(X)
        preds_loaded = loaded.predict_proba(X)
        np.testing.assert_allclose(preds_original, preds_loaded, atol=1e-5)

    def test_load_returns_ensemble_instance(self, tmp_path):
        """Loaded model is an EnsembleModel."""
        from defences.ensemble import EnsembleModel

        X, y = _make_dummy_data()
        model = EnsembleModel({"epochs": 1, "hidden_dim": 8, "batch_size": 64})
        model.fit(X, y)

        path = str(tmp_path / "ensemble.joblib")
        model.save(path)

        loaded = EnsembleModel.load(path)
        assert isinstance(loaded, EnsembleModel)

    def test_soft_voting_averages_three_models(self):
        """Ensemble output is the average of three sub-model predictions."""
        from defences.ensemble import EnsembleModel

        X, y = _make_dummy_data(n_samples=50, n_features=5)
        model = EnsembleModel({"epochs": 2, "hidden_dim": 8, "batch_size": 32})
        model.fit(X, y)

        # Get individual sub-model predictions
        import torch
        lr_probs = model.lr_model.predict_proba(X)[:, 1]
        xgb_probs = model.xgb_model.predict_proba(X)[:, 1]

        model.mlp.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(model.device)
        with torch.no_grad():
            mlp_out = model.mlp(X_tensor)
            mlp_probs = torch.sigmoid(mlp_out).cpu().numpy().flatten()

        expected = (lr_probs + xgb_probs + mlp_probs) / 3.0
        actual = model.predict_proba(X)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_extends_base_model(self):
        """EnsembleModel is a BaseModel subclass."""
        from defences.ensemble import EnsembleModel
        from models.base import BaseModel

        assert issubclass(EnsembleModel, BaseModel)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ensemble.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'defences.ensemble'`

---

## Task 2: Implement EnsembleModel

**Files:**
- Create: `defences/ensemble.py`

**Step 1: Write the EnsembleModel class**

```python
"""Ensemble defence: heterogeneous model ensemble with soft voting.

The ensemble combines LogisticRegression, XGBoost, and a 3-layer MLP.
Diversity across model families is the defensive mechanism — attacking
one component's decision boundary does not necessarily fool the others.

CAPGD compatibility: The MLP component is exposed via .model so CAPGD
can compute gradients. The full ensemble evaluates robustness via predict_proba.
"""

import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from models.base import BaseModel
from models.neural import SimpleMLP


class EnsembleModel(BaseModel):
    """Heterogeneous ensemble: LogisticRegression + XGBoost + MLP with soft voting.

    The MLP component is exposed via .model for CAPGD gradient-based attacks.
    predict_proba() averages P(fraud) from all three sub-models.
    """

    def __init__(self, params=None):
        super().__init__(params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # MLP hyperparams
        self.mlp_epochs = self.params.get("epochs", 15)
        self.mlp_hidden_dim = self.params.get("hidden_dim", 128)
        self.mlp_batch_size = self.params.get("batch_size", 256)
        self.mlp_lr = self.params.get("lr", 0.001)

        # Sub-models (initialized during fit)
        self.lr_model = None
        self.xgb_model = None
        self.mlp = None
        self.model = None  # alias for self.mlp — CAPGD compatibility
        self._use_logits = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        print("  Training Ensemble (LR + XGBoost + MLP)...")

        neg_count = int((y == 0).sum())
        pos_count = max(int((y == 1).sum()), 1)

        # 1. Logistic Regression
        print("    [1/3] LogisticRegression...")
        self.lr_model = LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="lbfgs"
        )
        self.lr_model.fit(X, y)

        # 2. XGBoost
        print("    [2/3] XGBClassifier...")
        self.xgb_model = XGBClassifier(
            max_depth=6,
            n_estimators=100,
            learning_rate=0.1,
            scale_pos_weight=neg_count / pos_count,
            tree_method="hist",
            objective="binary:logistic",
            eval_metric="aucpr",
            n_jobs=int(os.environ.get("XGBOOST_NTHREADS", 0)) or -1,
        )
        self.xgb_model.fit(X, y)

        # 3. MLP (class-weighted loss, same approach as NeuralModel)
        print("    [3/3] SimpleMLP...")
        input_dim = X.shape[1]
        pos_weight_tensor = torch.tensor(
            [neg_count / pos_count], dtype=torch.float32
        ).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self._use_logits = True

        self.mlp = SimpleMLP(input_dim, self.mlp_hidden_dim, use_sigmoid=False).to(
            self.device
        )
        self.model = self.mlp  # CAPGD reads model.model
        self.mlp.train()
        optimizer = optim.Adam(self.mlp.parameters(), lr=self.mlp_lr)

        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        y_tensor = (
            torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(self.device)
        )
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.mlp_batch_size, shuffle=True)

        for epoch in range(self.mlp_epochs):
            total_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.mlp(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(
                    f"      MLP Epoch {epoch + 1}/{self.mlp_epochs}, "
                    f"Loss: {total_loss / len(loader):.4f}"
                )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Soft voting: average P(fraud) from all three sub-models."""
        # LR
        lr_probs = self.lr_model.predict_proba(X)[:, 1]

        # XGBoost
        xgb_probs = self.xgb_model.predict_proba(X)[:, 1]

        # MLP
        self.mlp.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.mlp(X_tensor)
            if self._use_logits:
                outputs = torch.sigmoid(outputs)
        mlp_probs = outputs.cpu().numpy().flatten()

        return (lr_probs + xgb_probs + mlp_probs) / 3.0

    def save(self, path: str) -> None:
        """Save all three sub-models to a single joblib file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint = {
            "lr_model": self.lr_model,
            "xgb_model": self.xgb_model,
            "mlp_state_dict": self.mlp.state_dict(),
            "mlp_config": {
                "input_dim": self.mlp.fc1.in_features,
                "hidden_dim": self.mlp_hidden_dim,
                "use_sigmoid": self.mlp.use_sigmoid,
            },
            "params": self.params,
            "_use_logits": self._use_logits,
        }
        joblib.dump(checkpoint, path)

    @classmethod
    def load(cls, path: str) -> "EnsembleModel":
        """Load an EnsembleModel from a joblib checkpoint."""
        data = joblib.load(path)
        instance = cls(data.get("params", {}))
        instance.lr_model = data["lr_model"]
        instance.xgb_model = data["xgb_model"]

        cfg = data["mlp_config"]
        instance.mlp = SimpleMLP(
            cfg["input_dim"], cfg["hidden_dim"], cfg["use_sigmoid"]
        ).to(instance.device)
        instance.mlp.load_state_dict(data["mlp_state_dict"])
        instance.mlp.eval()
        instance.model = instance.mlp
        instance._use_logits = data.get("_use_logits", False)

        return instance
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_ensemble.py -v`
Expected: All 6 tests PASS

**Step 3: Run full test suite to check for regressions**

Run: `uv run pytest tests/ -v --ignore=tests/test_dataset.py`
Expected: All existing tests still PASS

**Step 4: Commit**

```bash
git add defences/ensemble.py tests/test_ensemble.py
git commit -m "feat: implement EnsembleModel defence (LR + XGBoost + MLP soft voting)"
```

---

## Task 3: Integrate EnsembleModel into Runner

**Files:**
- Modify: `runner/run.py` (lines 112-121 — model dispatch; line 132 — save extension)

**Step 1: Add ensemble model dispatch to runner**

In `runner/run.py`, replace the model dispatch block (lines 112-121):

```python
    if model_type == "tree":
        from models.tree import TreeModel

        model = TreeModel(model_params)
    elif model_type == "neural":
        from models.neural import NeuralModel

        model = NeuralModel(model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

With:

```python
    if model_type == "tree":
        from models.tree import TreeModel

        model = TreeModel(model_params)
    elif model_type == "neural":
        from models.neural import NeuralModel

        model = NeuralModel(model_params)
    elif model_type == "ensemble":
        from defences.ensemble import EnsembleModel

        model = EnsembleModel(model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

Also update the save extension line (currently line 132):

```python
    ext = ".pt" if model_type == "neural" else ".joblib"
```

This already handles ensemble correctly (falls into `.joblib`), so no change needed here.

**Step 2: Run tests**

Run: `uv run pytest tests/test_runner.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add runner/run.py
git commit -m "feat: add ensemble model dispatch to runner"
```

---

## Task 4: Create Ensemble Config Files (CAPGD + Square Attack)

**Files — CAPGD (GPU, white-box via MLP gradients):**
- Create: `configs/ccfd_ensemble.yaml`
- Create: `configs/ieee_cis_ensemble.yaml`
- Create: `configs/lcld_ensemble.yaml`
- Create: `configs/sparkov_ensemble.yaml`

**Files — Square Attack (GPU for training, black-box attack):**
- Create: `configs/ccfd_ensemble_square.yaml`
- Create: `configs/ieee_cis_ensemble_square.yaml`
- Create: `configs/lcld_ensemble_square.yaml`
- Create: `configs/sparkov_ensemble_square.yaml`

**Step 1: Create CAPGD ensemble configs (4 files)**

`configs/ccfd_ensemble.yaml`:
```yaml
experiment_name: "ccfd_ensemble"
seed: 42

dataset:
  name: "ccfd"
  split_strategy: "stratified"
  test_size: 0.2
  val_size: 0.2

model:
  type: "ensemble"
  params:
    epochs: 15
    hidden_dim: 128
    batch_size: 256
    lr: 0.001

attack:
  type: "capgd"
  epsilon: 0.1
  steps: 10

defence:
  type: "ensemble"
```

`configs/ieee_cis_ensemble.yaml`, `configs/lcld_ensemble.yaml`, `configs/sparkov_ensemble.yaml` — same structure, change `dataset.name` and `experiment_name` accordingly.

**Step 2: Create Square Attack ensemble configs (4 files)**

`configs/ccfd_ensemble_square.yaml`:
```yaml
experiment_name: "ccfd_ensemble_square"
seed: 42

dataset:
  name: "ccfd"
  split_strategy: "stratified"
  test_size: 0.2
  val_size: 0.2
  sample_frac: 0.1

model:
  type: "ensemble"
  params:
    epochs: 15
    hidden_dim: 128
    batch_size: 256
    lr: 0.001

attack:
  type: "square"
  epsilon: 0.1
  max_iter: 100
  norm: "inf"

defence:
  type: "ensemble"
```

Note: `sample_frac: 0.1` matches existing Square Attack configs (e.g. `ccfd_tree_square.yaml`) since Square Attack is slower and uses a subsample.

`configs/ieee_cis_ensemble_square.yaml`, `configs/lcld_ensemble_square.yaml`, `configs/sparkov_ensemble_square.yaml` — same structure, change `dataset.name` and `experiment_name`.

**Step 3: Verify config loading**

Run: `uv run python -c "import yaml; print(yaml.safe_load(open('configs/ccfd_ensemble.yaml'))['model']['type'])"`
Expected: `ensemble`

Run: `uv run python -c "import yaml; c = yaml.safe_load(open('configs/ccfd_ensemble_square.yaml')); print(c['attack']['type'], c['dataset'].get('sample_frac'))"`
Expected: `square 0.1`

**Step 4: Commit**

```bash
git add configs/ccfd_ensemble.yaml configs/ieee_cis_ensemble.yaml \
        configs/lcld_ensemble.yaml configs/sparkov_ensemble.yaml \
        configs/ccfd_ensemble_square.yaml configs/ieee_cis_ensemble_square.yaml \
        configs/lcld_ensemble_square.yaml configs/sparkov_ensemble_square.yaml
git commit -m "feat: add ensemble config files (CAPGD + Square) for all 4 datasets"
```

---

## Task 5: Run Ensemble Experiments (GPU + CPU separated)

**Files:**
- Create: `scripts/run_ensemble_experiments.py`

All ensemble experiments require GPU (MLP component trains on CUDA). The runner follows the same `run_all_seeds.py` pattern with `ProcessPoolExecutor`, `tqdm`, and `--gpu-only`/`--cpu-only` flags — but here GPU means "CAPGD configs" (white-box, fast) and CPU means "Square Attack configs" (black-box, slower, can run with fewer GPU resources).

**Step 1: Write ensemble batch runner with GPU/CPU separation**

```python
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


if __name__ == "__main__":
    main()
```

**Step 2: Run CAPGD ensemble experiments (GPU, ~12 runs)**

Run: `uv run python scripts/run_ensemble_experiments.py --gpu-only`
Expected: 12 experiments complete (4 datasets x 3 seeds), results logged to `results/registry.csv`

**Step 3: Run Square Attack ensemble experiments (GPU for train, ~12 runs)**

Run: `uv run python scripts/run_ensemble_experiments.py --cpu-only`
Expected: 12 experiments complete, results appended to `results/registry.csv`

**Step 4: Verify results in registry**

Run: `uv run python -c "import pandas as pd; df = pd.read_csv('results/registry.csv'); print(df[df['defence_type']=='ensemble'][['dataset','seed','attack_type','defence_type','clean_pr_auc','robust_pr_auc']].to_string())"`
Expected: 24 rows with `defence_type == "ensemble"` (12 CAPGD + 12 Square)

**Step 5: Commit**

```bash
git add scripts/run_ensemble_experiments.py
git commit -m "feat: add ensemble batch runner (GPU/CPU split) + run 24 experiments"
```

---

## Task 6: Create Z-Threshold Sweep Config Files

**Files — GPU (neural, require CUDA for training + CAPGD):**
- Create: `configs/ccfd_input_val_z5.yaml`
- Create: `configs/ccfd_input_val_z10.yaml`
- Create: `configs/sparkov_input_val_z5.yaml`
- Create: `configs/sparkov_input_val_z10.yaml`

**Files — CPU (tree, XGBoost trains on CPU, CAPGD is no-op):**
- Create: `configs/ccfd_tree_input_val_z5.yaml`
- Create: `configs/ccfd_tree_input_val_z10.yaml`
- Create: `configs/sparkov_tree_input_val_z5.yaml`
- Create: `configs/sparkov_tree_input_val_z10.yaml`

These configs mirror the existing `*_input_val.yaml` configs but with modified `z_threshold` values.

**Step 1: Create CCFD neural configs (z=5.0 and z=10.0)**

`configs/ccfd_input_val_z5.yaml`:
```yaml
experiment_name: "ccfd_neural_input_val_z5"
seed: 42

dataset:
  name: "ccfd"
  split_strategy: "stratified"
  test_size: 0.2
  val_size: 0.2

model:
  type: "neural"
  params:
    epochs: 20
    hidden_dim: 128
    batch_size: 256
    lr: 0.001

attack:
  type: "capgd"
  epsilon: 0.1
  steps: 10

defence:
  type: "input_validation"
  params:
    mode: "sanitise"
    z_threshold: 5.0
```

`configs/ccfd_input_val_z10.yaml` — same but `z_threshold: 10.0` and `experiment_name: "ccfd_neural_input_val_z10"`.

**Step 2: Create CCFD tree configs, Sparkov neural configs, Sparkov tree configs**

Follow the same pattern. For tree configs, use:
- `model.type: "tree"`
- `model.params: {max_depth: 6, n_estimators: 100, learning_rate: 0.1}`
- Tree configs use the `ccfd_tree_input_val.yaml` / `sparkov_tree_input_val.yaml` patterns

For Sparkov configs, use `dataset.name: "sparkov"`.

Total: 8 new config files.

**Step 3: Commit**

```bash
git add configs/ccfd_input_val_z5.yaml configs/ccfd_input_val_z10.yaml \
        configs/ccfd_tree_input_val_z5.yaml configs/ccfd_tree_input_val_z10.yaml \
        configs/sparkov_input_val_z5.yaml configs/sparkov_input_val_z10.yaml \
        configs/sparkov_tree_input_val_z5.yaml configs/sparkov_tree_input_val_z10.yaml
git commit -m "feat: add z-threshold sweep configs for CCFD and Sparkov"
```

---

## Task 7: Run Z-Threshold Sweep and Analyse

**Files:**
- Create: `scripts/run_z_threshold_sweep.py`
- Modify: `scripts/analyse_input_validation.py` (extend to compare across thresholds)

**Step 1: Write sweep runner with GPU/CPU separation**

```python
"""Run z-threshold sweep experiments for input validation analysis.

GPU configs = neural models (require CUDA for training + CAPGD gradients).
CPU configs = tree models (XGBoost trains on CPU, CAPGD is a no-op on trees).

Usage:
    uv run python scripts/run_z_threshold_sweep.py              # run all
    uv run python scripts/run_z_threshold_sweep.py --gpu-only   # neural only
    uv run python scripts/run_z_threshold_sweep.py --cpu-only   # tree only
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

# Only seed 42 for z-threshold sweep (as specified in todo)
SEEDS = [42]

# Neural models — require GPU for training + CAPGD
GPU_CONFIGS = [
    "configs/ccfd_input_val_z5.yaml",
    "configs/ccfd_input_val_z10.yaml",
    "configs/sparkov_input_val_z5.yaml",
    "configs/sparkov_input_val_z10.yaml",
]

# Tree models — CPU only (XGBoost + CAPGD no-op)
CPU_CONFIGS = [
    "configs/ccfd_tree_input_val_z5.yaml",
    "configs/ccfd_tree_input_val_z10.yaml",
    "configs/sparkov_tree_input_val_z5.yaml",
    "configs/sparkov_tree_input_val_z10.yaml",
]


def _short_name(config_path):
    return os.path.splitext(os.path.basename(config_path))[0]


def _run_one(config, seed):
    cmd = [sys.executable, "-m", "runner.run", "--config", config, "--seed", str(seed)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return config, seed, result.returncode, result.stderr


def main():
    parser = argparse.ArgumentParser(description="Z-threshold sweep batch runner")
    parser.add_argument("--cpu-only", action="store_true", help="Tree (CPU) configs only")
    parser.add_argument("--gpu-only", action="store_true", help="Neural (GPU) configs only")
    parser.add_argument("--workers", type=int, default=None,
                        help="Max parallel workers (default: 1 for GPU, 2 for CPU)")
    args = parser.parse_args()

    if args.cpu_only:
        configs = CPU_CONFIGS
        default_workers = 2
        label = "CPU (tree)"
    elif args.gpu_only:
        configs = GPU_CONFIGS
        default_workers = 1
        label = "GPU (neural)"
    else:
        configs = GPU_CONFIGS + CPU_CONFIGS
        default_workers = 1
        label = "All"

    max_workers = args.workers or default_workers
    experiments = [(config, seed) for config in configs for seed in SEEDS]
    total = len(experiments)
    failed = []
    start = time.time()

    print(f"Running {total} z-threshold experiments ({label}) with {max_workers} workers...")

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


if __name__ == "__main__":
    main()
```

**Step 2: Run GPU (neural) z-threshold experiments**

Run: `uv run python scripts/run_z_threshold_sweep.py --gpu-only`
Expected: 4 neural experiments complete

**Step 3: Run CPU (tree) z-threshold experiments**

Run: `uv run python scripts/run_z_threshold_sweep.py --cpu-only`
Expected: 4 tree experiments complete (can run in parallel with Step 2 on a separate terminal)

**Step 3: Write z-threshold comparison analysis script**

Create `scripts/analyse_z_threshold.py`:

```python
"""Analyse how z_threshold values affect input validation defence effectiveness.

Compares z_threshold={3.0, 5.0, 10.0} against baseline (no defence) for CCFD
and Sparkov datasets, both neural and tree models.
"""

import argparse
import os
import re

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_registry(path: str) -> pd.DataFrame:
    from scripts.generate_figures import load_registry as _load
    return _load(path)


def extract_z_threshold(experiment_name: str):
    """Extract z_threshold from experiment name, e.g. 'ccfd_neural_input_val_z5' -> 5.0"""
    match = re.search(r"_z(\d+(?:\.\d+)?)$", experiment_name)
    if match:
        return float(match.group(1))
    # Default z_threshold for standard input_val configs
    if "input_val" in experiment_name:
        return 3.0
    return None


def main():
    parser = argparse.ArgumentParser(description="Z-threshold sweep analysis")
    parser.add_argument("--registry", default="results/registry.csv")
    parser.add_argument("--output", default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    df = load_registry(args.registry)
    print(f"Loaded {len(df)} rows from {args.registry}")

    # Filter to relevant experiments
    relevant = df[
        (df["dataset"].isin(["ccfd", "sparkov"]))
        & (df["defence_type"].isin(["none", "input_validation"]))
        & (df["seed"] == 42)
    ].copy()

    # Extract z_threshold
    relevant["z_threshold"] = relevant["experiment_name"].apply(extract_z_threshold)
    relevant.loc[relevant["defence_type"] == "none", "z_threshold"] = "none"

    print("\nZ-threshold comparison:")
    cols = ["dataset", "model_type", "defence_type", "z_threshold",
            "clean_pr_auc", "robust_pr_auc"]
    print(relevant[cols].sort_values(["dataset", "model_type", "z_threshold"]).to_string(index=False))

    # Save CSV
    csv_path = os.path.join(args.output, "z_threshold_analysis.csv")
    relevant[cols].to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
```

**Step 4: Run the analysis**

Run: `uv run python scripts/analyse_z_threshold.py --registry results/registry.csv`
Expected: CSV output showing clean/robust PR-AUC for each z_threshold value

**Step 5: Commit**

```bash
git add scripts/run_z_threshold_sweep.py scripts/analyse_z_threshold.py
git commit -m "feat: add z-threshold sweep runner and analysis script"
```

---

## Task 8: Adversarial Training Trade-off Analysis

**Files:**
- Create: `scripts/analyse_adv_training.py`

This task requires NO new experiments — all data is already in `registry_clean.csv`.

**Step 1: Write the analysis script**

```python
"""Analyse adversarial training trade-offs across datasets.

Computes clean accuracy cost, correlates effectiveness with dataset
characteristics, and documents the tree model limitation.
"""

import argparse
import os

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_registry(path: str) -> pd.DataFrame:
    from scripts.generate_figures import load_registry as _load
    return _load(path)


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    from scripts.generate_figures import aggregate_seeds as _agg
    return _agg(df)


# Dataset characteristics (from docs/Context.md and dataset cards)
DATASET_CHARS = {
    "ccfd": {"fraud_rate": 0.00173, "n_features": 30, "n_samples": 284807},
    "ieee_cis": {"fraud_rate": 0.03499, "n_features": 394, "n_samples": 590540},
    "lcld": {"fraud_rate": 0.11300, "n_features": 57, "n_samples": 100653},
    "sparkov": {"fraud_rate": 0.00579, "n_features": 22, "n_samples": 1296675},
}


def compute_tradeoffs(agg: pd.DataFrame) -> pd.DataFrame:
    """Compute clean accuracy cost and robustness gain from adv training."""
    merge_keys = ["dataset", "model_type", "attack_type", "attack_epsilon"]

    baseline = agg[agg["defence_type"] == "none"].copy()
    adv_train = agg[agg["defence_type"] == "adversarial_training"].copy()

    if baseline.empty or adv_train.empty:
        print("ERROR: need both 'none' and 'adversarial_training' rows.")
        return pd.DataFrame()

    merged = adv_train.merge(
        baseline[merge_keys + ["clean_pr_auc_mean", "robust_pr_auc_mean"]],
        on=merge_keys,
        suffixes=("_at", "_base"),
    )

    merged["clean_cost"] = merged["clean_pr_auc_mean_at"] - merged["clean_pr_auc_mean_base"]
    merged["robust_gain"] = merged["robust_pr_auc_mean_at"] - merged["robust_pr_auc_mean_base"]

    # Add dataset characteristics
    for dataset, chars in DATASET_CHARS.items():
        mask = merged["dataset"] == dataset
        for key, val in chars.items():
            merged.loc[mask, key] = val

    return merged


def main():
    parser = argparse.ArgumentParser(description="Adversarial training trade-off analysis")
    parser.add_argument("--registry", default="results/registry_clean.csv")
    parser.add_argument("--output", default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    df = load_registry(args.registry)
    print(f"Loaded {len(df)} rows")

    agg = aggregate_seeds(df)
    tradeoffs = compute_tradeoffs(agg)

    if tradeoffs.empty:
        print("No adversarial training data found.")
        return

    print("\n" + "=" * 80)
    print("ADVERSARIAL TRAINING TRADE-OFF ANALYSIS")
    print("=" * 80)

    for _, row in tradeoffs.iterrows():
        label = f"{row['dataset'].upper()} / {row['model_type']}"
        print(f"\n  {label}")
        print(f"    Clean PR-AUC cost:  {row['clean_cost']:+.4f}")
        print(f"    Robust PR-AUC gain: {row['robust_gain']:+.4f}")
        if "fraud_rate" in row:
            print(f"    Dataset fraud rate:  {row['fraud_rate']:.5f}")

    # Save CSV
    csv_path = os.path.join(args.output, "adv_training_tradeoffs.csv")
    tradeoffs.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    # Note about tree models
    print("\n" + "-" * 80)
    print("TREE MODEL LIMITATION:")
    print("  Adversarial Training requires gradient computation (backpropagation).")
    print("  XGBoost tree models are non-differentiable, making Adversarial Training")
    print("  architecturally inapplicable. This is a model-family constraint, not a bug.")
    print("  The runner raises ValueError if defence=adversarial_training with model=tree.")
    print("-" * 80)


if __name__ == "__main__":
    main()
```

**Step 2: Run the analysis**

Run: `uv run python scripts/analyse_adv_training.py`
Expected: Trade-off table printed, CSV saved to `results/figures/adv_training_tradeoffs.csv`

**Step 3: Commit**

```bash
git add scripts/analyse_adv_training.py
git commit -m "feat: add adversarial training trade-off analysis script"
```

---

## Task 9: Update Statistical Tests

**Files:**
- Modify: `scripts/statistical_tests.py`
- Modify: `tests/test_figures.py` (update tests for new pairs)

**Step 1: Update `pairwise_defence_tests` to include ensemble and Wilcoxon**

In `scripts/statistical_tests.py`, update the `pairs` list (line 39-43) and add Wilcoxon test:

Add `("none", "ensemble")`, `("adversarial_training", "ensemble")`, and `("input_validation", "ensemble")` to the pairs list:

```python
    pairs = [
        ("none", "adversarial_training"),
        ("none", "input_validation"),
        ("none", "ensemble"),
        ("adversarial_training", "input_validation"),
        ("adversarial_training", "ensemble"),
        ("input_validation", "ensemble"),
    ]
```

Add Wilcoxon test alongside the existing t-test. After the `t_stat, p_val = stats.ttest_rel(a, b)` line (line 102), add:

```python
            # Wilcoxon signed-rank (requires n >= 6 for meaningful results)
            if n_paired >= 6:
                w_stat, w_pval = stats.wilcoxon(a, b)
            else:
                w_stat, w_pval = np.nan, np.nan
```

And include `w_statistic` and `w_p_value` in the row dict.

**Step 2: Run tests**

Run: `uv run pytest tests/test_figures.py -v`
Expected: PASS (may need to update test expectations for new pairs)

**Step 3: Run statistical tests**

Run: `uv run python scripts/statistical_tests.py --registry results/registry.csv`
Expected: Results table including ensemble comparisons

**Step 4: Commit**

```bash
git add scripts/statistical_tests.py tests/test_figures.py
git commit -m "feat: add ensemble defence pairs and Wilcoxon test to statistical analysis"
```

---

## Task 10: Deduplicate Registry

After running ensemble and z-threshold experiments, the registry will contain new rows alongside existing data.

**Step 1: Create updated registry_clean.csv**

Run the deduplication script (or manually merge):

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('results/registry.csv')
# Deduplicate: keep last entry per (experiment_name, seed)
df = df.drop_duplicates(subset=['experiment_name', 'seed'], keep='last')
df = df.sort_values(['dataset', 'model_type', 'defence_type', 'seed'])
df.to_csv('results/registry_clean.csv', index=False)
print(f'registry_clean.csv: {len(df)} rows')
"
```

**Step 2: Commit**

```bash
git add results/registry_clean.csv
git commit -m "data: update registry_clean.csv with ensemble and z-threshold results"
```

---

## Task 11: Generate Final Figures

**Files:**
- No code changes needed — `generate_figures.py` auto-discovers defence types from registry

**Step 1: Run figure generation**

Run: `uv run python scripts/generate_figures.py --registry results/registry_clean.csv`
Expected: 6 figure types generated in `results/figures/`

**Step 2: Verify figures include ensemble data**

Check that `defence_heatmap.png` includes an "ensemble" column. Check that `robustness_bars.png` shows ensemble alongside other defences.

**Step 3: Run all analysis scripts**

```bash
uv run python scripts/analyse_input_validation.py --registry results/registry_clean.csv
uv run python scripts/analyse_adv_training.py --registry results/registry_clean.csv
uv run python scripts/statistical_tests.py --registry results/registry_clean.csv
uv run python scripts/analyse_z_threshold.py --registry results/registry.csv
```

**Step 4: Commit**

```bash
git add results/figures/
git commit -m "docs: generate final figures with ensemble defence data"
```

---

## Task 12: Write Reproducibility Documentation

**Files:**
- Modify: `README.md` (add dataset download instructions, defence docs, experiment reproduction)

**Step 1: Add dataset download instructions**

Add a "Datasets" section to README.md with:
- CCFD: Kaggle URL, expected file size, directory placement
- IEEE-CIS: Kaggle URL, expected files
- LCLD: Source URL, expected structure
- Sparkov: Source URL, expected structure
- Expected directory structure under `data/`

**Step 2: Add defence documentation**

Add a "Defences" section documenting:
- Adversarial Training: how it works, applicable to neural only, config params
- Input Validation: how it works, z_threshold parameter, known issues
- Ensemble: how it works, sub-model composition, applicable to all model families

**Step 3: Add experiment reproduction instructions**

Add a "Reproducing Results" section:
- How to run a single experiment: `uv run python -m runner.run --config configs/ccfd.yaml --seed 42`
- How to run all experiments: `uv run python scripts/run_all_seeds.py`
- How to run ensemble experiments: `uv run python scripts/run_ensemble_experiments.py`
- How to generate figures: `uv run python scripts/generate_figures.py`
- How to run statistical tests: `uv run python scripts/statistical_tests.py`

**Step 4: Document YAML config fields**

Add a "Configuration" section explaining every YAML field.

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: add dataset download, defence docs, and reproduction instructions"
```

---

## Experiment Summary

| Experiment Set | Attack | GPU/CPU | Configs | Seeds | Total Runs |
|---|---|---|---|---|---|
| Ensemble + CAPGD | capgd | GPU | 4 | 3 | 12 |
| Ensemble + Square | square | GPU (train) | 4 | 3 | 12 |
| Z-threshold neural | capgd | GPU | 4 | 1 | 4 |
| Z-threshold tree | capgd | CPU | 4 | 1 | 4 |
| **Total** | | | **16** | | **32** |

**Why all ensemble experiments need GPU:** The EnsembleModel trains a SimpleMLP on CUDA. Even Square Attack configs (black-box attack, CPU-only) still require GPU for the training phase. Schedule all ensemble runs on GPU-capable machines.

**Z-threshold tree experiments are CPU-only:** XGBoost trains on CPU, and CAPGD returns X unchanged for tree models (no gradients). These can run on CPU-only machines or in parallel with GPU jobs.

---

## Task Dependencies

```
Task 1 (tests) → Task 2 (implement) → Task 3 (runner) → Task 4 (configs) → Task 5 (run experiments)
                                                           ↓                    ↓ (GPU: CAPGD + Square)
Task 6 (z-threshold configs) → Task 7 (run z-threshold) ——————————————→ Task 10 (deduplicate)
                                ↓ GPU: neural                                    ↓
                                ↓ CPU: tree (parallel)                   Task 11 (figures)
                                                                                 ↓
Task 8 (adv training analysis) ————————————————————————————————————→ Task 12 (docs)
Task 9 (statistical tests) ————————————————————————————————————————→
```

**Parallelizable:**
- Tasks 6-7 (z-threshold) can run in parallel with Tasks 1-5 (ensemble) once Task 3 is done
- Task 7 GPU (neural) and CPU (tree) runs can execute in parallel on separate machines
- Task 5 GPU (CAPGD) and CPU (Square) can be run sequentially or interleaved
- Task 8 (adv training analysis) can run any time (uses existing data, no new experiments)
- Task 9 (statistical tests update) can be coded after Task 2 (run after experiments complete)

---

## Estimated Effort

| Task | Description | Estimate |
|------|-------------|----------|
| 1-4 | EnsembleModel implementation + 8 configs | ~3 hours |
| 5 | Run 24 ensemble experiments (12 CAPGD + 12 Square) | ~3 hours (compute) |
| 6-7 | Z-threshold sweep configs + 8 runs + analysis | ~2 hours |
| 8 | Adv training trade-off analysis | ~1 hour |
| 9 | Statistical tests update | ~1 hour |
| 10-11 | Registry dedup + figures | ~1 hour |
| 12 | Reproducibility docs | ~1 hour |
| **Total** | | **~12 hours** |
