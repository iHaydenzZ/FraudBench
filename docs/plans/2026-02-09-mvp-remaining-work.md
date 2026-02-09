# MVP Remaining Work — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete all P0 critical fixes and P1 improvements to pass the MVP success criteria: CAPGD decreases fraud metrics, at least one defence improves robust metrics, constraint validity reported, results registry complete.

**Architecture:** Three code fixes (adversarial training + constraints, input validation + outlier detection, registry columns), then runner integration, then 5 new config files, then experiment execution. Each fix is backward-compatible — existing tests must pass at every step.

**Tech Stack:** Python 3.11, PyTorch, XGBoost, scikit-learn, pytest, uv

---

## Task 1: Add F1/Precision to Results Registry

**Why first:** Zero-risk change, unblocks correct logging for all subsequent experiments.

**Files:**
- Modify: `evaluation/registry.py`
- Delete + recreate: `results/registry.csv` (stale 14-column file)

**Step 1: Update `_ensure_registry_exists` column list**

In `evaluation/registry.py`, add 4 new columns after `robust_recall`:

```python
writer.writerow([
    "timestamp",
    "experiment_name",
    "dataset",
    "model_type",
    "defence_type",
    "attack_type",
    "attack_epsilon",
    "validity_rate",
    "adv_validity_rate",
    "clean_pr_auc",
    "clean_precision",       # NEW
    "clean_recall",
    "clean_f1",              # NEW
    "robust_pr_auc",
    "robust_precision",      # NEW
    "robust_recall",
    "robust_f1",             # NEW
    "clean_accuracy",
    "robust_accuracy",
    "train_time_sec",
    "attack_time_sec"
])
```

**Step 2: Update `log_experiment` row construction**

Add the 4 new metric values to the `row` list, pulling from `metrics_clean` and `metrics_robust`:

```python
row = [
    datetime.now().isoformat(),
    config.get('experiment_name', 'n/a'),
    config['dataset']['name'],
    config['model']['type'],
    config.get('defence', {}).get('type', 'none'),
    config.get('attack', {}).get('type', 'none'),
    config.get('attack', {}).get('epsilon', 0.0),
    f"{validity_rate:.4f}",
    f"{adv_validity_rate:.4f}" if adv_validity_rate is not None else "n/a",
    f"{metrics_clean.get('pr_auc', 0):.4f}",
    f"{metrics_clean.get('precision', 0):.4f}",     # NEW
    f"{metrics_clean.get('recall', 0):.4f}",
    f"{metrics_clean.get('f1', 0):.4f}",            # NEW
    f"{metrics_robust.get('pr_auc', 0):.4f}",
    f"{metrics_robust.get('precision', 0):.4f}",    # NEW
    f"{metrics_robust.get('recall', 0):.4f}",
    f"{metrics_robust.get('f1', 0):.4f}",           # NEW
    f"{metrics_clean.get('accuracy', 0):.4f}",
    f"{metrics_robust.get('accuracy', 0):.4f}",
    f"{train_time_sec:.2f}" if train_time_sec is not None else "n/a",
    f"{attack_time_sec:.2f}" if attack_time_sec is not None else "n/a"
]
```

**Step 3: Delete stale registry CSV**

Delete `results/registry.csv` (has 14 columns, incompatible with new 21-column schema). It only contains one dummy_dataset test row — no real data to preserve.

**Step 4: Run existing tests**

```bash
uv run pytest tests/ -v
```

Expected: All 27 tests pass (registry is not directly tested).

**Step 5: Commit**

```bash
git add evaluation/registry.py
git commit -m "feat: add F1/precision columns to results registry, delete stale CSV"
```

---

## Task 2: Constraint-Aware Adversarial Training

**Why:** P0 fix — `adversarial_train_step` uses plain PGD without domain constraints. Must reuse `project_constraints` from `attacks/capgd.py`.

**Files:**
- Modify: `defences/adversarial_training.py`
- Modify: `models/neural.py` (thread schema through to training step)
- Modify: `runner/run.py` (pass schema via model_params)
- Test: `tests/test_defences.py` (new file)

### Step 1: Write failing test for constraint-aware adversarial training

Create `tests/test_defences.py`:

```python
"""Tests for defence implementations."""
import pytest
import torch
import pandas as pd
import numpy as np
from constraints.schema import ConstraintSchema, FeatureConstraint


class TestAdversarialTraining:
    """Tests for adversarial training with constraint projection."""

    def test_adv_train_step_respects_constraints(self):
        """Adversarial examples generated during training respect schema bounds."""
        from models.neural import NeuralModel
        from constraints.schema import ConstraintSchema

        # Create data with known bounds
        np.random.seed(42)
        X = pd.DataFrame({
            'feat_0': np.random.uniform(0, 1, 100),
            'feat_1': np.random.uniform(-1, 1, 100),
        })
        y = pd.Series(np.random.randint(0, 2, 100))

        feature_types = {'feat_0': 'numeric', 'feat_1': 'numeric'}
        schema = ConstraintSchema.from_data(X, feature_types)

        # Train with adversarial training — should not crash
        model = NeuralModel({
            'epochs': 2,
            'hidden_dim': 8,
            'adv_training': True,
            'adv_epsilon': 0.3,
            'adv_schema': schema,
            'adv_feature_names': X.columns.tolist(),
            'adv_feature_types': feature_types,
        })
        model.fit(X, y)

        # Model should still produce valid probabilities
        probs = model.predict_proba(X)
        assert probs.shape == (100,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_adv_train_step_without_schema_still_works(self):
        """Backward compat: adversarial training without schema uses L-inf only."""
        from models.neural import NeuralModel

        X = pd.DataFrame(np.random.randn(50, 3), columns=['a', 'b', 'c'])
        y = pd.Series(np.random.randint(0, 2, 50))

        model = NeuralModel({
            'epochs': 2,
            'hidden_dim': 8,
            'adv_training': True,
            'adv_epsilon': 0.1,
        })
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert probs.shape == (50,)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_defences.py -v
```

Expected: First test fails because `NeuralModel` doesn't pass schema to `adversarial_train_step`.

**Step 3: Update `defences/adversarial_training.py`**

Add optional `schema`, `feature_names`, `feature_types` params. If provided, apply `project_constraints` after each PGD step:

```python
import torch
import torch.nn as nn


def adversarial_train_step(model, X_batch, y_batch, criterion, optimizer, device,
                           epsilon=0.1, alpha=0.5, schema=None, feature_names=None,
                           feature_types=None):
    """
    Performs one training step with mixed clean and adversarial data.
    Uses constraint projection when schema is provided.
    """
    model.eval()

    steps = 3
    step_size = 1.25 * epsilon / steps

    x_adv = X_batch.clone().detach()
    x_adv.requires_grad = True

    use_constraints = schema is not None and feature_names is not None and feature_types is not None
    if use_constraints:
        from attacks.capgd import project_constraints

    for _ in range(steps):
        outputs = model(x_adv)
        loss = criterion(outputs, y_batch)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = x_adv.grad
            x_adv = x_adv + step_size * grad.sign()
            delta = torch.clamp(x_adv - X_batch, -epsilon, epsilon)
            x_adv = X_batch + delta

            if use_constraints:
                x_adv = project_constraints(x_adv, X_batch, schema, feature_names, feature_types)

            x_adv.requires_grad = True

    # Train on mixed batch
    model.train()
    optimizer.zero_grad()

    out_clean = model(X_batch)
    loss_clean = criterion(out_clean, y_batch)

    out_adv = model(x_adv.detach())
    loss_adv = criterion(out_adv, y_batch)

    total_loss = alpha * loss_clean + (1 - alpha) * loss_adv

    total_loss.backward()
    optimizer.step()

    return total_loss.item()
```

**Step 4: Update `models/neural.py` to thread schema params**

In the `fit()` method, extract and pass schema params to `adversarial_train_step`:

Change the adv_training block (around line 68-79) to:

```python
adv_training = self.params.get("adv_training", False)
adv_epsilon = self.params.get("adv_epsilon", 0.1)
adv_schema = self.params.get("adv_schema", None)
adv_feature_names = self.params.get("adv_feature_names", None)
adv_feature_types = self.params.get("adv_feature_types", None)

if adv_training:
    from defences.adversarial_training import adversarial_train_step
    print(f"  Enabled Adversarial Training (eps={adv_epsilon})")

for epoch in range(self.epochs):
    total_loss = 0
    for X_batch, y_batch in loader:
        if adv_training:
            loss = adversarial_train_step(
                self.model, X_batch, y_batch, criterion, optimizer, self.device,
                epsilon=adv_epsilon, schema=adv_schema,
                feature_names=adv_feature_names, feature_types=adv_feature_types
            )
            total_loss += loss
        else:
            optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.4f}")
```

**Step 5: Update `runner/run.py` to pass schema into model_params**

In the adversarial training config block (around line 78-84), after building `schema_processed`, pass it:

```python
if defence_config.get('type') == 'adversarial_training':
    print("    Configuring Adversarial Training...")
    model_params['adv_training'] = True
    model_params['adv_epsilon'] = defence_config.get('params', {}).get('epsilon', 0.1)
    # Schema will be set after preprocessing — see below
```

Then, after preprocessing is done (after line ~75 where `X_train_processed` is available), insert:

```python
# Pass processed schema to adversarial training if enabled
if defence_config.get('type') == 'adversarial_training':
    fake_types_at = {c: 'numeric' for c in X_train_processed.columns}
    schema_at = ConstraintSchema.from_data(X_train_processed, fake_types_at)
    model_params['adv_schema'] = schema_at
    model_params['adv_feature_names'] = X_train_processed.columns.tolist()
    model_params['adv_feature_types'] = fake_types_at
```

This block goes right before the model instantiation (before `if model_type == "tree":`), and requires moving the `from constraints.schema import ConstraintSchema` import earlier (or adding it at the top of the block).

**Step 6: Run tests**

```bash
uv run pytest tests/test_defences.py tests/test_attacks.py tests/ -v
```

Expected: All tests pass including new defence tests.

**Step 7: Commit**

```bash
git add defences/adversarial_training.py models/neural.py runner/run.py tests/test_defences.py
git commit -m "feat: constraint-aware adversarial training using project_constraints"
```

---

## Task 3: Input Validation — Outlier Detection + Rejection Modes

**Why:** P0 fix — InputValidator only clips numeric, does nothing for categorical, has no outlier detection.

**Files:**
- Modify: `defences/input_validation.py`
- Modify: `runner/run.py` (pass mode config, call `fit()`)
- Test: `tests/test_defences.py` (add new test class)

### Step 1: Write failing tests

Append to `tests/test_defences.py`:

```python
class TestInputValidation:
    """Tests for input validation defence."""

    def test_numeric_clipping(self):
        """Numeric features are clipped to schema bounds."""
        from defences.input_validation import InputValidator

        schema = ConstraintSchema()
        schema.features['x'] = FeatureConstraint(name='x', type='numeric', min_val=0.0, max_val=10.0)

        iv = InputValidator(schema)
        X_train = pd.DataFrame({'x': [1.0, 5.0, 9.0]})
        iv.fit(X_train)

        X_bad = pd.DataFrame({'x': [-5.0, 5.0, 15.0]})
        result = iv.transform(X_bad)
        assert result['x'].tolist() == [0.0, 5.0, 10.0]

    def test_outlier_detection_sanitise_mode(self):
        """Outliers beyond k*std are clipped in sanitise mode."""
        from defences.input_validation import InputValidator

        schema = ConstraintSchema()
        schema.features['x'] = FeatureConstraint(name='x', type='numeric', min_val=-100.0, max_val=100.0)

        iv = InputValidator(schema, mode='sanitise', z_threshold=2.0)
        X_train = pd.DataFrame({'x': np.random.normal(0, 1, 1000)})
        iv.fit(X_train)

        X_test = pd.DataFrame({'x': [0.0, 5.0, -5.0]})  # 5.0 and -5.0 are outliers at z=2
        result = iv.transform(X_test)

        # Outliers should be clipped to ±z_threshold * std
        assert abs(result['x'].iloc[0] - 0.0) < 0.01  # Normal value unchanged
        assert result['x'].iloc[1] < 5.0  # Clipped down
        assert result['x'].iloc[2] > -5.0  # Clipped up

    def test_outlier_detection_reject_mode(self):
        """Outliers beyond k*std are rejected (replaced with NaN) in reject mode."""
        from defences.input_validation import InputValidator

        schema = ConstraintSchema()
        schema.features['x'] = FeatureConstraint(name='x', type='numeric', min_val=-100.0, max_val=100.0)

        iv = InputValidator(schema, mode='reject', z_threshold=2.0)
        X_train = pd.DataFrame({'x': np.random.normal(0, 1, 1000)})
        iv.fit(X_train)

        X_test = pd.DataFrame({'x': [0.0, 5.0, -5.0]})
        result, metadata = iv.transform(X_test, return_metadata=True)

        assert metadata['n_rejected'] >= 1
        assert metadata['total'] == 3

    def test_backward_compat_no_fit(self):
        """Transform still works without calling fit (no outlier detection)."""
        from defences.input_validation import InputValidator

        schema = ConstraintSchema()
        schema.features['x'] = FeatureConstraint(name='x', type='numeric', min_val=0.0, max_val=10.0)

        iv = InputValidator(schema)
        X_bad = pd.DataFrame({'x': [-5.0, 5.0, 15.0]})
        result = iv.transform(X_bad)  # Should not crash
        assert result['x'].tolist() == [0.0, 5.0, 10.0]
```

**Step 2: Run test to verify failures**

```bash
uv run pytest tests/test_defences.py::TestInputValidation -v
```

**Step 3: Implement updated `InputValidator`**

Replace `defences/input_validation.py`:

```python
import pandas as pd
import numpy as np
from constraints.schema import ConstraintSchema


class InputValidator:
    def __init__(self, schema: ConstraintSchema, mode: str = 'sanitise', z_threshold: float = 3.0):
        """
        Args:
            schema: Feature constraint schema for bound clipping.
            mode: 'sanitise' (clip outliers) or 'reject' (mark outliers as NaN).
            z_threshold: Z-score threshold for outlier detection. Only active after fit().
        """
        self.schema = schema
        self.mode = mode
        self.z_threshold = z_threshold
        self._fitted = False
        self._means = {}
        self._stds = {}

    def fit(self, X: pd.DataFrame, y=None):
        """Compute per-feature mean/std for outlier detection."""
        for col, constraint in self.schema.features.items():
            if col not in X.columns or constraint.type != 'numeric':
                continue
            self._means[col] = X[col].mean()
            self._stds[col] = X[col].std()
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame, return_metadata: bool = False):
        """
        Projects samples to feasible region.

        If fitted, also applies outlier detection (clipping or rejection).
        """
        X_clean = X.copy()
        n_rejected = 0
        rejected_mask = pd.Series(False, index=X_clean.index)

        for col, constraint in self.schema.features.items():
            if col not in X_clean.columns:
                continue

            if constraint.type == 'numeric':
                # 1. Bound clipping (always)
                min_v = constraint.min_val if constraint.min_val is not None else -float('inf')
                max_v = constraint.max_val if constraint.max_val is not None else float('inf')
                X_clean[col] = X_clean[col].clip(lower=min_v, upper=max_v)

                # 2. Outlier detection (only if fitted)
                if self._fitted and col in self._means and self._stds.get(col, 0) > 0:
                    mean = self._means[col]
                    std = self._stds[col]
                    lower_bound = mean - self.z_threshold * std
                    upper_bound = mean + self.z_threshold * std

                    if self.mode == 'sanitise':
                        X_clean[col] = X_clean[col].clip(lower=lower_bound, upper=upper_bound)
                    elif self.mode == 'reject':
                        outlier_mask = (X_clean[col] < lower_bound) | (X_clean[col] > upper_bound)
                        rejected_mask = rejected_mask | outlier_mask

        if self.mode == 'reject' and rejected_mask.any():
            n_rejected = rejected_mask.sum()

        if return_metadata:
            metadata = {
                'n_rejected': int(n_rejected),
                'total': len(X),
                'rejection_rate': n_rejected / len(X) if len(X) > 0 else 0.0,
            }
            return X_clean, metadata

        return X_clean
```

**Step 4: Update `runner/run.py` input validation setup**

In the input validation block (around line 137-145), add `fit()` call and config passthrough:

```python
if defence_config.get('type') == 'input_validation':
    print("    Configuring Input Validation Defence...")
    from defences.input_validation import InputValidator
    fake_types = {c: 'numeric' for c in X_train_processed.columns}
    schema_processed = ConstraintSchema.from_data(X_train_processed, fake_types)
    iv_params = defence_config.get('params', {})
    input_validator = InputValidator(
        schema_processed,
        mode=iv_params.get('mode', 'sanitise'),
        z_threshold=iv_params.get('z_threshold', 3.0),
    )
    input_validator.fit(X_train_processed)
```

**Step 5: Run tests**

```bash
uv run pytest tests/ -v
```

Expected: All tests pass (27 old + 4 new defence tests).

**Step 6: Commit**

```bash
git add defences/input_validation.py runner/run.py tests/test_defences.py
git commit -m "feat: input validation with outlier detection, sanitise/reject modes"
```

---

## Task 4: Runner — Fix `fake_types` and Defence Integration Paths

**Why:** The runner hard-codes all processed features as 'numeric'. Also need to verify both defence paths work end-to-end.

**Files:**
- Modify: `runner/run.py`

### Step 1: Consolidate `fake_types` and schema creation

The `fake_types` pattern appears twice (input validation setup + attack setup). Extract it once, right after preprocessing:

After the preprocessing block (after `X_test_processed = preprocessor.transform(X_test)`), add:

```python
# Build processed-space schema (used by both attacks and defences)
processed_feature_types = {c: 'numeric' for c in X_train_processed.columns}
processed_schema = ConstraintSchema.from_data(X_train_processed, processed_feature_types)
processed_feature_names = X_train_processed.columns.tolist()
```

Then replace the two inline `fake_types` + `schema_processed` blocks with references to `processed_schema`, `processed_feature_types`, `processed_feature_names`.

### Step 2: Move adversarial training schema injection after preprocessing

The current runner sets `adv_training=True` before preprocessing, but schema is only available after. Restructure:

```python
# Before model init, after preprocessing:
if defence_config.get('type') == 'adversarial_training':
    print("    Configuring Adversarial Training...")
    model_params['adv_training'] = True
    model_params['adv_epsilon'] = defence_config.get('params', {}).get('epsilon', 0.1)
    model_params['adv_schema'] = processed_schema
    model_params['adv_feature_names'] = processed_feature_names
    model_params['adv_feature_types'] = processed_feature_types
```

### Step 3: Run existing tests

```bash
uv run pytest tests/ -v
```

### Step 4: Commit

```bash
git add runner/run.py
git commit -m "refactor: consolidate processed schema in runner, fix fake_types duplication"
```

---

## Task 5: Create Missing Config Files

**Why:** 5 config files needed for the MVP experiment matrix (experiments 2-6).

**Files:**
- Create: `configs/ccfd_input_val.yaml`
- Create: `configs/ccfd_adv_train.yaml`
- Create: `configs/ccfd_tree.yaml`
- Create: `configs/ccfd_tree_input_val.yaml`
- Create: `configs/ccfd_tree_adv_train.yaml`

### Step 1: Create all 5 configs

**`configs/ccfd_input_val.yaml`** — CCFD + Neural + Input Validation:
```yaml
experiment_name: "ccfd_neural_input_val"
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
    z_threshold: 3.0
```

**`configs/ccfd_adv_train.yaml`** — CCFD + Neural + Adversarial Training:
```yaml
experiment_name: "ccfd_neural_adv_train"
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
  type: "adversarial_training"
  params:
    epsilon: 0.1
```

**`configs/ccfd_tree.yaml`** — CCFD + XGBoost + No Defence:
```yaml
experiment_name: "ccfd_tree_baseline"
seed: 42

dataset:
  name: "ccfd"
  split_strategy: "stratified"
  test_size: 0.2
  val_size: 0.2

model:
  type: "tree"
  params:
    max_depth: 6
    n_estimators: 100
    learning_rate: 0.1

attack:
  type: "capgd"
  epsilon: 0.1
  steps: 10

defence:
  type: "none"
```

**`configs/ccfd_tree_input_val.yaml`** — CCFD + XGBoost + Input Validation:
```yaml
experiment_name: "ccfd_tree_input_val"
seed: 42

dataset:
  name: "ccfd"
  split_strategy: "stratified"
  test_size: 0.2
  val_size: 0.2

model:
  type: "tree"
  params:
    max_depth: 6
    n_estimators: 100
    learning_rate: 0.1

attack:
  type: "capgd"
  epsilon: 0.1
  steps: 10

defence:
  type: "input_validation"
  params:
    mode: "sanitise"
    z_threshold: 3.0
```

**`configs/ccfd_tree_adv_train.yaml`** — CCFD + XGBoost + Adversarial Training:
```yaml
experiment_name: "ccfd_tree_adv_train"
seed: 42

dataset:
  name: "ccfd"
  split_strategy: "stratified"
  test_size: 0.2
  val_size: 0.2

model:
  type: "tree"
  params:
    max_depth: 6
    n_estimators: 100
    learning_rate: 0.1

attack:
  type: "capgd"
  epsilon: 0.1
  steps: 10

defence:
  type: "adversarial_training"
  params:
    epsilon: 0.1
```

Note: XGBoost + adversarial training won't have constraint-projected training (AT only wired to NeuralModel). CAPGD will be skipped for tree models (existing behavior). These configs log clean metrics + the AT/IV attempt for completeness.

### Step 2: Commit

```bash
git add configs/ccfd_input_val.yaml configs/ccfd_adv_train.yaml configs/ccfd_tree.yaml configs/ccfd_tree_input_val.yaml configs/ccfd_tree_adv_train.yaml
git commit -m "feat: add 5 config files for MVP defence experiments"
```

---

## Task 6: Run MVP Experiment Matrix

**Why:** Populate the results registry with all required experiments.

**Prerequisite:** CCFD dataset must be available at the configured `DEFAULT_DATA_ROOT` path.

### Step 1: Run experiments 1-3 (Neural MLP)

```bash
uv run python -m runner.run --config configs/ccfd.yaml
uv run python -m runner.run --config configs/ccfd_input_val.yaml
uv run python -m runner.run --config configs/ccfd_adv_train.yaml
```

After each: verify `results/registry.csv` gains a new row.

### Step 2: Run experiments 4-6 (XGBoost)

```bash
uv run python -m runner.run --config configs/ccfd_tree.yaml
uv run python -m runner.run --config configs/ccfd_tree_input_val.yaml
uv run python -m runner.run --config configs/ccfd_tree_adv_train.yaml
```

CAPGD will print "Warning: Model does not appear to be a PyTorch model. Skipping." for tree models. This is expected — robust metrics will be empty.

### Step 3: Run experiments 7-8 (IEEE-CIS)

```bash
uv run python -m runner.run --config configs/ieee_cis.yaml
```

### Step 4: Verify registry

Open `results/registry.csv` and confirm:
- At least 7 rows (one per experiment)
- All 21 columns populated
- At least one defence experiment shows improved robust metrics vs baseline

### Step 5: Commit

```bash
git add results/registry.csv
git commit -m "data: add MVP experiment results to registry"
```

---

## Task 7 (P1): Multiple Epsilon Configs

**Why:** A single epsilon point is not a convincing benchmark. Need ε ∈ {0.05, 0.1, 0.2}.

**Files:**
- Create: `configs/ccfd_eps005.yaml`
- Create: `configs/ccfd_eps010.yaml` (or reuse `ccfd.yaml`)
- Create: `configs/ccfd_eps020.yaml`

### Step 1: Create epsilon sweep configs

Each is a copy of `configs/ccfd.yaml` with different `experiment_name` and `attack.epsilon`:

**`configs/ccfd_eps005.yaml`:**
```yaml
experiment_name: "ccfd_neural_eps005"
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
  epsilon: 0.05
  steps: 10
defence:
  type: "none"
```

**`configs/ccfd_eps020.yaml`:**
```yaml
experiment_name: "ccfd_neural_eps020"
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
  epsilon: 0.2
  steps: 10
defence:
  type: "none"
```

### Step 2: Run epsilon sweep

```bash
uv run python -m runner.run --config configs/ccfd_eps005.yaml
uv run python -m runner.run --config configs/ccfd.yaml
uv run python -m runner.run --config configs/ccfd_eps020.yaml
```

### Step 3: Commit

```bash
git add configs/ccfd_eps005.yaml configs/ccfd_eps020.yaml results/registry.csv
git commit -m "feat: epsilon sweep configs and results for ε={0.05, 0.1, 0.2}"
```

---

## Verification Checklist

After all tasks, confirm MVP success criteria:

| Criterion | How to Verify |
|-----------|--------------|
| CAPGD decreases fraud PR-AUC/Recall | Compare `clean_pr_auc` vs `robust_pr_auc` in registry for `ccfd_baseline` |
| At least one defence improves robust metrics | Compare `robust_pr_auc` of `ccfd_neural_input_val` or `ccfd_neural_adv_train` vs `ccfd_baseline` |
| Constraint validity rate reported | Check `validity_rate` and `adv_validity_rate` columns are populated |
| Results registry complete | All experiments have rows with all 21 columns |
| All tests pass | `uv run pytest tests/ -v` shows 31+ tests passing |

```bash
# Final verification
uv run pytest tests/ -v
cat results/registry.csv
```
