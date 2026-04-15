# Mask Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `notebooks/mask_ablation.ipynb` that runs 6 new perturbation-mask variants (M2–M6) on LCLD, loads existing M0/M1 baselines, and produces summary tables, feasibility audit, perturbation stats, and E1 cost-weighted analysis for the next advisor meeting.

**Architecture:** One self-contained Colab notebook that copies setup/utility cells verbatim from `notebooks/tabularbench_comparison.ipynb`, reuses saved preprocessors, trains the neural model once per seed, and loops over mask variants inside that outer seed loop. Results land in `results/adv_examples/mask_ablation/` and are backed up to Google Drive.

**Tech Stack:** Python 3.11, PyTorch, pandas, scikit-learn, Jupyter/Colab. Existing FraudBench modules: `datasets.loader`, `preprocessing.processor`, `constraints.schema`, `models.neural`, `evaluation.metrics`, `attacks.capgd`.

**Execution environment:** Google Colab with A100 GPU. Repo at `/content/Capstone_FraudBench`, dataset symlinked from Google Drive. There is NO local `pytest` flow for this notebook — verification is by inspecting cell output against expected counts/ranges documented in each task.

**Reference spec:** `docs/plans/mask_ablation_experiment_plan.md` (read before starting — defines the scientific rationale for each variant).

**Reference notebook:** `notebooks/tabularbench_comparison.ipynb` (the source of truth for all copied cells; cell numbers below refer to this file).

---

## File Structure

**Created:**
- `notebooks/mask_ablation.ipynb` — the single deliverable. Self-contained; no shared helper module.

**Modified:**
- None.

**Read/consumed:**
- `notebooks/tabularbench_comparison.ipynb` — source for copied cells.
- `results/preprocessor_lcld_n134097_seed{42,123,456}.joblib` — reused, never refit.
- Google Drive `/content/drive/MyDrive/FraudBench/results/adv_examples/lcld_neural_{unmasked,masked}_seed{42,123,456}.parquet` — M0/M1 baselines.
- Google Drive `/content/drive/MyDrive/FraudBench/results/adv_examples/comparison_{unmasked,masked}.csv` — M0/M1 robust metrics.

**Written:**
- `results/adv_examples/mask_ablation/lcld_neural_{M2,M3,M4,M5,M6strict,M6relaxed}_seed{42,123,456}.parquet`
- `results/adv_examples/mask_ablation/mask_ablation_results.csv` — per-seed metrics
- `results/adv_examples/mask_ablation/mask_ablation_summary.csv` — mean±std table
- `results/adv_examples/mask_ablation/mask_ablation_feasibility.csv` — seed=42 feasibility detail
- `results/adv_examples/mask_ablation/mask_ablation_perturbation_stats.csv` — seed=42 delta stats
- `results/adv_examples/mask_ablation/e1_cost_summary.csv`
- `results/adv_examples/mask_ablation/e1_cost_distribution.png`
- `results/adv_examples/mask_ablation/e1_affordable_curve.png`
- Google Drive mirror of the entire `mask_ablation/` directory.

Each task below produces a self-contained notebook cell (or cell block). Commit after each task.

---

## Task 1: Create notebook skeleton with copied setup cells

**Files:**
- Create: `notebooks/mask_ablation.ipynb`
- Read: `notebooks/tabularbench_comparison.ipynb` (cells 1–5)

**Rationale:** Setup is identical to the reference notebook. Copy verbatim to guarantee environment parity.

- [ ] **Step 1: Create the empty notebook file**

In Colab: File → New notebook → rename to `mask_ablation.ipynb`. Save to `/content/Capstone_FraudBench/notebooks/`.

Alternatively, locally:

```bash
cp notebooks/tabularbench_comparison.ipynb notebooks/mask_ablation.ipynb
```

Then delete all cells EXCEPT cells 1–5 (GPU check, Drive mount, repo clone, pip install, dataset symlink). Use Jupyter's cell delete UI or edit the JSON to keep only the first 5 code cells and the title markdown.

- [ ] **Step 2: Update the title markdown cell**

Replace the first markdown cell with:

```markdown
# Mask Ablation Experiment — LCLD

Runs mask variants M2 (directional), M3 (derived-feature freeze), M4 (term OHE freeze), M5 (combined), M6-strict, M6-relaxed on LCLD with the neural model and CAPGD attack. Loads M0/M1 baselines from existing comparison results. Produces summary tables, feasibility audit, perturbation stats, and E1 cost-weighted analysis.

Reference: `notebooks/tabularbench_comparison.ipynb` (reused setup + utility functions).
Spec: `docs/plans/mask_ablation_experiment_plan.md`.
```

- [ ] **Step 3: Verify cells 1–5 run end-to-end on Colab**

Run cells 1 through 5 in order. Expected output:
- Cell 1 (GPU): `Tesla A100` or similar.
- Cell 2 (Drive mount): `Mounted at /content/drive`.
- Cell 3 (repo clone): `HEAD is now at <sha>` or `Already up to date.`.
- Cell 4 (pip install): finishes with `Successfully installed` (Colab will likely ask you to restart runtime — do so, then re-run cells 2 and 3).
- Cell 5 (symlink): `datasets/LCLD -> /content/drive/MyDrive/FraudBench/data/LCLD`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/mask_ablation.ipynb
git commit -m "feat(notebook): scaffold mask_ablation.ipynb with setup cells"
```

---

## Task 2: Cell 6 — Mask variant configuration

**Files:**
- Modify: `notebooks/mask_ablation.ipynb` (add Cell 6)

**Rationale:** All variant definitions live in one cell so reviewers can see the full experiment space in a single view. Each variant is just an immutable set + optional directional config.

- [ ] **Step 1: Add Cell 6 (code cell)**

```python
# Cell 6: Mask variant configuration
#
# Every variant is defined by an immutable set of RAW feature names.
# build_processed_mutable_mask() handles OHE expansion by prefix matching.

from typing import Set

# --- Baseline immutable set (copied verbatim from tabularbench_comparison.ipynb Cell 6) ---
LCLD_IMMUTABLE_RAW: Set[str] = {
    # LC internal pricing/grading
    "grade", "sub_grade", "int_rate", "installment",
    "funded_amnt", "funded_amnt_inv", "initial_list_status",
    # LC verification outcomes
    "verification_status", "verification_status_joint",
    # Credit bureau data
    "delinq_2yrs", "inq_last_6mths", "inq_last_12m", "inq_fi",
    "open_acc", "open_acc_6m", "open_act_il",
    "open_il_12m", "open_il_24m", "open_rv_12m", "open_rv_24m",
    "pub_rec", "pub_rec_bankruptcies", "total_acc",
    "revol_bal", "revol_util", "il_util", "all_util",
    "tot_cur_bal", "tot_hi_cred_lim", "total_bal_il",
    "total_rev_hi_lim", "max_bal_bc",
    "pct_tl_nvr_dlq", "percent_bc_gt_75",
    "collections_12_mths_ex_med",
    "mths_since_last_delinq", "mths_since_last_il_delinq",
    "mths_since_last_major_delinq", "mths_since_last_record",
    "mths_since_rcnt_il",
    "payment_inc_ratio",
}

LCLD_MUTABLE_RAW: Set[str] = {
    "loan_amnt", "term", "purpose", "emp_length",
    "annual_inc", "annual_inc_joint", "home_ownership",
    "dti", "dti_joint", "application_type", "addr_state",
}

# --- Variant immutable sets ---
LCLD_IMMUTABLE_M3 = LCLD_IMMUTABLE_RAW | {"dti", "dti_joint"}
LCLD_IMMUTABLE_M4 = LCLD_IMMUTABLE_RAW | {"term"}
LCLD_IMMUTABLE_M5 = LCLD_IMMUTABLE_RAW | {"dti", "dti_joint", "term"}

# --- M6 attacker-capability profiles ---
MUTABLE_STRICT: Set[str] = {
    "loan_amnt", "purpose", "home_ownership", "application_type", "addr_state",
}
MUTABLE_RELAXED: Set[str] = {
    "loan_amnt", "purpose", "home_ownership", "application_type", "addr_state",
    "annual_inc", "annual_inc_joint", "emp_length",
}

# --- M2 directionality ---
# Only emp_length is increase-only among LCLD mutable features.
DIRECTION_CONSTRAINTS = {"emp_length": "increase"}

# --- E1 cost weights (raw-space, normalized units) ---
FEATURE_COSTS = {
    "loan_amnt":        1.0,
    "purpose":          0.5,
    "home_ownership":   3.0,
    "addr_state":       2.0,
    "application_type": 1.0,
    "annual_inc":       8.0,
    "annual_inc_joint": 8.0,
    "emp_length":       5.0,
    "dti":              7.0,
    "dti_joint":        7.0,
    "term":             1.0,
}

# --- Sanity checks ---
assert LCLD_IMMUTABLE_RAW.isdisjoint(LCLD_MUTABLE_RAW), "raw sets overlap"
assert set(FEATURE_COSTS.keys()) == LCLD_MUTABLE_RAW, "cost dict must cover all mutable features"
assert MUTABLE_STRICT <= LCLD_MUTABLE_RAW, "strict profile contains non-mutable feature"
assert MUTABLE_RELAXED <= LCLD_MUTABLE_RAW, "relaxed profile contains non-mutable feature"

print(f"Baseline mutable:   {len(LCLD_MUTABLE_RAW)} raw features")
print(f"M3 locks adds:      {sorted(LCLD_IMMUTABLE_M3 - LCLD_IMMUTABLE_RAW)}")
print(f"M4 locks adds:      {sorted(LCLD_IMMUTABLE_M4 - LCLD_IMMUTABLE_RAW)}")
print(f"M5 locks adds:      {sorted(LCLD_IMMUTABLE_M5 - LCLD_IMMUTABLE_RAW)}")
print(f"M6-strict mutable:  {sorted(MUTABLE_STRICT)} ({len(MUTABLE_STRICT)} features)")
print(f"M6-relaxed mutable: {sorted(MUTABLE_RELAXED)} ({len(MUTABLE_RELAXED)} features)")
```

- [ ] **Step 2: Run Cell 6 and verify output**

Expected: all 5 print lines appear, all 4 assertions pass, no errors.

- [ ] **Step 3: Commit**

```bash
git add notebooks/mask_ablation.ipynb
git commit -m "feat(notebook): add mask variant configuration cell"
```

---

## Task 3: Cell 7 — Copy utility functions from reference notebook

**Files:**
- Modify: `notebooks/mask_ablation.ipynb` (add Cell 7)
- Read: `notebooks/tabularbench_comparison.ipynb` (Cell 7: `build_processed_mutable_mask` and `capgd_attack_masked`)

**Rationale:** Reuse the exact same mask builder and attack function as the baseline so M0/M1 comparisons remain apples-to-apples. We add ONE new function `capgd_attack_masked_directional` for M2.

- [ ] **Step 1: Copy `build_processed_mutable_mask` and `capgd_attack_masked` verbatim**

Open `notebooks/tabularbench_comparison.ipynb` Cell 7. Copy the entire cell body (both function definitions, imports, and the trailing print) into a new Cell 7 in `mask_ablation.ipynb`.

DO NOT modify the copied code.

- [ ] **Step 2: Append `capgd_attack_masked_directional` to the same cell**

After the existing `capgd_attack_masked` definition, append:

```python
def capgd_attack_masked_directional(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    schema,
    feature_types: Dict[str, str],
    mutable_mask: np.ndarray,
    direction_cols: Dict[int, str],  # {processed_col_idx: "increase"|"decrease"}
    params: Dict[str, Any] = None,
) -> pd.DataFrame:
    """
    CAPGD with mask + per-column directional constraints.
    direction_cols maps processed column index to 'increase' (delta >= 0)
    or 'decrease' (delta <= 0). Applied after the epsilon-ball projection
    at every step.
    """
    from attacks.capgd import project_constraints

    if params is None:
        params = {}

    epsilon = params.get("epsilon", 0.1)
    steps = params.get("steps", 10)
    step_size = params.get("step_size", epsilon / 4)

    if not hasattr(model, "model") or not isinstance(model.model, nn.Module):
        print("Warning: Model is not PyTorch. Returning clean data.")
        return X

    torch_model = model.model
    device = model.device
    torch_model.eval()

    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(device)
    feature_names = X.columns.tolist()

    mutable_t = torch.tensor(mutable_mask, dtype=torch.bool).to(device)

    # Build directional clip tensors once
    inc_idx = torch.tensor(
        [i for i, d in direction_cols.items() if d == "increase"],
        dtype=torch.long, device=device,
    )
    dec_idx = torch.tensor(
        [i for i, d in direction_cols.items() if d == "decrease"],
        dtype=torch.long, device=device,
    )

    noise = torch.zeros_like(X_tensor).uniform_(-epsilon, epsilon)
    noise[:, ~mutable_t] = 0
    x_adv = X_tensor + noise
    # Apply directional clip to initial noise
    if len(inc_idx) > 0:
        x_adv[:, inc_idx] = torch.maximum(x_adv[:, inc_idx], X_tensor[:, inc_idx])
    if len(dec_idx) > 0:
        x_adv[:, dec_idx] = torch.minimum(x_adv[:, dec_idx], X_tensor[:, dec_idx])
    x_adv = project_constraints(x_adv, X_tensor, schema, feature_names, feature_types)
    x_adv = x_adv.detach()
    x_adv.requires_grad = True

    use_logits = hasattr(model, "_use_logits") and model._use_logits
    criterion = nn.BCEWithLogitsLoss() if use_logits else nn.BCELoss()

    for step in range(steps):
        outputs = torch_model(x_adv)
        loss = criterion(outputs, y_tensor)

        torch_model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = x_adv.grad
            grad[:, ~mutable_t] = 0

            x_adv = x_adv + step_size * grad.sign()

            if epsilon > 0:
                delta = x_adv - X_tensor
                delta = torch.clamp(delta, -epsilon, epsilon)
                delta[:, ~mutable_t] = 0
                x_adv = X_tensor + delta

            # Directional clip (processed space; StandardScaler preserves direction)
            if len(inc_idx) > 0:
                x_adv[:, inc_idx] = torch.maximum(x_adv[:, inc_idx], X_tensor[:, inc_idx])
            if len(dec_idx) > 0:
                x_adv[:, dec_idx] = torch.minimum(x_adv[:, dec_idx], X_tensor[:, dec_idx])

            x_adv = project_constraints(
                x_adv, X_tensor, schema, feature_names, feature_types
            )
            x_adv.requires_grad = True

    return pd.DataFrame(
        x_adv.detach().cpu().numpy(), columns=feature_names, index=X.index
    )


print("Directional CAPGD function defined.")
```

- [ ] **Step 2a: Add a helper to resolve raw→processed column indices for directional constraints**

Append to the same cell:

```python
def resolve_direction_indices(
    processed_feature_names: list,
    direction_raw: Dict[str, str],
) -> Dict[int, str]:
    """Map raw feature directional config to processed column indices.

    For numeric raw features the processed name equals the raw name.
    For OHE-expanded categoricals, matches by prefix. M2 currently only
    uses numeric features (emp_length), so prefix matching is defensive.
    """
    out: Dict[int, str] = {}
    for i, col in enumerate(processed_feature_names):
        if col in direction_raw:
            out[i] = direction_raw[col]
            continue
        parts = col.split("_")
        for k in range(1, len(parts)):
            prefix = "_".join(parts[:k])
            if prefix in direction_raw:
                out[i] = direction_raw[prefix]
                break
    return out


print("Direction index resolver defined.")
```

- [ ] **Step 3: Run Cell 7 and verify**

Expected output:
```
Masked CAPGD function defined.
Directional CAPGD function defined.
Direction index resolver defined.
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/mask_ablation.ipynb
git commit -m "feat(notebook): add mask utilities and directional CAPGD variant"
```

---

## Task 4: Cell 8 — Main experiment loop (trains once per seed, runs all variants)

**Files:**
- Modify: `notebooks/mask_ablation.ipynb` (add Cell 8)
- Read: `notebooks/tabularbench_comparison.ipynb` (Cell 8 — this is what we adapt)

**Rationale:** Train the neural model once per seed, then evaluate all 6 new variants against it. Saves ~5× training time vs re-training per variant. Preprocessor is LOADED (never refit) to guarantee scaling parity with M0/M1.

- [ ] **Step 1: Add Cell 8**

```python
# Cell 8: Main experiment loop — train once per seed, run all 6 new variants
import time
import os
import numpy as np
import pandas as pd

from datasets.loader import load_dataset
from datasets.splitter import split_dataset
from preprocessing.processor import DataPreprocessor, get_preprocessor_path
from constraints.schema import ConstraintSchema
from models.neural import NeuralModel
from evaluation.metrics import compute_metrics

SEEDS = [42, 123, 456]
EPSILON = 0.1
SAMPLE_FRAC = 0.1
ATTACK_PARAMS = {"epsilon": EPSILON, "steps": 10, "step_size": EPSILON / 4}
MODEL_PARAMS = {"epochs": 20, "hidden_dim": 128, "batch_size": 256, "lr": 0.001}

ADV_SAVE_DIR = "results/adv_examples/mask_ablation"
os.makedirs(ADV_SAVE_DIR, exist_ok=True)

# {variant_name: (attack_fn_kind, immutable_set_or_None, extra)}
# attack_fn_kind in {"masked", "directional"}.
# For M6 the immutable set is computed per-seed from dataset.X.columns.
VARIANTS = {
    "M2":         ("directional", LCLD_IMMUTABLE_RAW, {"direction": DIRECTION_CONSTRAINTS}),
    "M3":         ("masked",      LCLD_IMMUTABLE_M3,  {}),
    "M4":         ("masked",      LCLD_IMMUTABLE_M4,  {}),
    "M5":         ("masked",      LCLD_IMMUTABLE_M5,  {}),
    "M6strict":   ("masked",      "from_profile",      {"profile": MUTABLE_STRICT}),
    "M6relaxed":  ("masked",      "from_profile",      {"profile": MUTABLE_RELAXED}),
}

rows = []  # per-(variant, seed) row for mask_ablation_results.csv

for seed in SEEDS:
    print(f"\n{'='*60}\n  SEED = {seed}\n{'='*60}")

    # --- Load, split, preprocess (reuse saved preprocessor) ---
    dataset = load_dataset("lcld", config={"sample_frac": SAMPLE_FRAC})
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        dataset, test_size=0.2, val_size=0.2, random_state=seed,
    )
    preprocessor_path = get_preprocessor_path("lcld", seed, len(dataset.X))
    assert os.path.exists(preprocessor_path), (
        f"Expected preprocessor at {preprocessor_path} (should have been saved by "
        f"tabularbench_comparison.ipynb). Run that notebook first or sync from Drive."
    )
    preprocessor = DataPreprocessor.load(preprocessor_path)
    X_train_p = preprocessor.transform(X_train)
    X_test_p = preprocessor.transform(X_test)
    processed_feature_types = {c: "numeric" for c in X_train_p.columns}
    processed_schema = ConstraintSchema.from_data(X_train_p, processed_feature_types)
    print(f"  Data: train={len(X_train)}, test={len(X_test)}, processed={X_test_p.shape[1]}")

    # --- Train once ---
    model = NeuralModel(MODEL_PARAMS)
    t0 = time.time()
    model.fit(X_train_p, y_train)
    print(f"  Trained in {time.time()-t0:.1f}s")

    clean_probs = model.predict_proba(X_test_p)
    clean_metrics = compute_metrics(y_test, clean_probs)
    print(f"  Clean  -- PR-AUC: {clean_metrics['pr_auc']:.4f}, Acc: {clean_metrics['accuracy']:.4f}")

    # Raw feature universe for M6 immutable-set construction
    raw_all = set(dataset.X.columns)

    # --- Loop over variants ---
    for vname, (kind, imm, extra) in VARIANTS.items():
        if imm == "from_profile":
            immutable_raw = raw_all - extra["profile"]
        else:
            immutable_raw = imm

        mutable_mask = build_processed_mutable_mask(
            X_test_p.columns.tolist(), immutable_raw
        )
        n_mut = int(mutable_mask.sum())
        n_imm = int((~mutable_mask).sum())

        t0 = time.time()
        if kind == "directional":
            dir_idx = resolve_direction_indices(
                X_test_p.columns.tolist(), extra["direction"]
            )
            X_adv = capgd_attack_masked_directional(
                model, X_test_p, y_test, processed_schema, processed_feature_types,
                mutable_mask, dir_idx, params=ATTACK_PARAMS,
            )
        else:
            X_adv = capgd_attack_masked(
                model, X_test_p, y_test, processed_schema, processed_feature_types,
                mutable_mask, params=ATTACK_PARAMS,
            )
        dt = time.time() - t0

        robust_probs = model.predict_proba(X_adv)
        robust = compute_metrics(y_test, robust_probs)
        print(
            f"  {vname:10s} mut={n_mut:3d} imm={n_imm:3d} "
            f"-- Robust PR-AUC: {robust['pr_auc']:.4f}, Acc: {robust['accuracy']:.4f} ({dt:.1f}s)"
        )

        parquet_path = os.path.join(ADV_SAVE_DIR, f"lcld_neural_{vname}_seed{seed}.parquet")
        X_adv.to_parquet(parquet_path)

        rows.append({
            "variant": vname, "seed": seed,
            "n_mutable": n_mut, "n_immutable": n_imm,
            "clean_pr_auc":  clean_metrics["pr_auc"],
            "clean_accuracy": clean_metrics["accuracy"],
            "clean_recall":   clean_metrics.get("recall", np.nan),
            "clean_f1":       clean_metrics.get("f1", np.nan),
            "robust_pr_auc":  robust["pr_auc"],
            "robust_accuracy": robust["accuracy"],
            "robust_recall":   robust.get("recall", np.nan),
            "robust_f1":       robust.get("f1", np.nan),
            "attack_time_s": dt,
        })

results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(ADV_SAVE_DIR, "mask_ablation_results.csv"), index=False)
print(f"\nSaved per-seed results to {ADV_SAVE_DIR}/mask_ablation_results.csv")
print(results_df)
```

- [ ] **Step 2: Run Cell 8 and verify**

Expected output structure:
- Three `SEED = X` blocks.
- Each block: 6 variant rows (`M2`, `M3`, `M4`, `M5`, `M6strict`, `M6relaxed`).
- `M6strict` should show the smallest `mut` count (expect ~30–40 in processed space).
- `M6relaxed` should be slightly higher than `M6strict`.
- `M3/M4` should be within 1–3 of the M1 baseline mutable count (123 for seeds 42/456, 122 for seed 123).
- Total runtime: ~3–6 minutes for all 3 seeds × 6 variants.
- Final `results_df` has 18 rows (3 seeds × 6 variants).

If the assertion about the preprocessor path fails, run `tabularbench_comparison.ipynb` first (or pull preprocessors from Drive into `results/`).

- [ ] **Step 3: Commit**

```bash
git add notebooks/mask_ablation.ipynb
git commit -m "feat(notebook): add main experiment loop for mask ablation"
```

---

## Task 5: Cell 9 — Load M0/M1 baselines from existing artifacts

**Files:**
- Modify: `notebooks/mask_ablation.ipynb` (add Cell 9)

**Rationale:** Avoid recomputing baselines; re-use M0/M1 parquets and comparison CSVs already produced by `tabularbench_comparison.ipynb`. Appending them to `results_df` gives a single unified table.

- [ ] **Step 1: Add Cell 9**

```python
# Cell 9: Load M0 (no-mask) and M1 (binary-mask) baselines into the same results frame
import shutil

BASELINE_SRC = "/content/drive/MyDrive/FraudBench/results/adv_examples"
BASELINE_LOCAL = "results/adv_examples"  # parquets already produced by reference notebook

# Copy baseline parquets locally if missing (they may only exist on Drive)
os.makedirs(BASELINE_LOCAL, exist_ok=True)
for seed in SEEDS:
    for variant_label, baseline_file in [
        ("M0", f"lcld_neural_unmasked_seed{seed}.parquet"),
        ("M1", f"lcld_neural_masked_seed{seed}.parquet"),
    ]:
        src = os.path.join(BASELINE_SRC, baseline_file)
        dst = os.path.join(BASELINE_LOCAL, baseline_file)
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy(src, dst)

# Load comparison CSVs to get the robust metrics the baseline notebook recorded.
m0_csv = os.path.join(BASELINE_SRC, "comparison_unmasked.csv")
m1_csv = os.path.join(BASELINE_SRC, "comparison_masked.csv")
assert os.path.exists(m0_csv) and os.path.exists(m1_csv), (
    "Baseline comparison CSVs not found on Drive. Run tabularbench_comparison.ipynb first."
)
m0 = pd.read_csv(m0_csv)
m1 = pd.read_csv(m1_csv)

def _canon(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Normalize baseline rows to the same schema as results_df."""
    out = pd.DataFrame({
        "variant": variant,
        "seed": df["seed"].astype(int),
        "n_mutable":   df.get("n_mutable",   np.nan),
        "n_immutable": df.get("n_immutable", np.nan),
        "clean_pr_auc":   df.get("clean_pr_auc",   np.nan),
        "clean_accuracy": df.get("clean_accuracy", np.nan),
        "clean_recall":   df.get("clean_recall",   np.nan),
        "clean_f1":       df.get("clean_f1",       np.nan),
        "robust_pr_auc":   df["robust_pr_auc"],
        "robust_accuracy": df["robust_accuracy"],
        "robust_recall":   df.get("robust_recall", np.nan),
        "robust_f1":       df.get("robust_f1",     np.nan),
        "attack_time_s":   df.get("attack_time_s", np.nan),
    })
    return out

baseline_rows = pd.concat([_canon(m0, "M0"), _canon(m1, "M1")], ignore_index=True)
all_results = pd.concat([baseline_rows, results_df], ignore_index=True)
all_results.to_csv(os.path.join(ADV_SAVE_DIR, "mask_ablation_results.csv"), index=False)
print(f"Combined results ({len(all_results)} rows = 8 variants × 3 seeds):")
print(all_results)
```

- [ ] **Step 2: Run Cell 9 and verify**

Expected:
- `all_results` has 24 rows (8 variants × 3 seeds).
- Variants present: M0, M1, M2, M3, M4, M5, M6strict, M6relaxed.
- M0 rows show `robust_accuracy ≈ 0.05` (from prior runs).
- M1 rows show `robust_accuracy ≈ 0.17`.
- If the baseline columns `clean_*` are NaN, that's acceptable — they are re-derived from the current-run clean metrics at summary time if needed.

- [ ] **Step 3: Commit**

```bash
git add notebooks/mask_ablation.ipynb
git commit -m "feat(notebook): load M0/M1 baselines into unified results table"
```

---

## Task 6: Cell 10 — Feasibility audit utility functions

**Files:**
- Modify: `notebooks/mask_ablation.ipynb` (add Cell 10)
- Read: `notebooks/tabularbench_comparison.ipynb` (Cells 11–13)

**Rationale:** Copy the audit functions verbatim so feasibility is measured with the exact same definitions as the baseline report.

- [ ] **Step 1: Add Cell 10**

Copy into one new cell, from `tabularbench_comparison.ipynb`:
- The full `inverse_transform_numeric` function (Cell 11 body).
- All constraint checks: `check_g1_installment`, `check_g2_open_total`, `check_g3_bankruptcies`, `check_g4_term`, `check_g4_processed` (Cell 12 body).
- `reconstruct_term_from_ohe` (Cell 12 body).
- `compute_aggregate_feasibility` (Cell 13 body).

DO NOT modify the functions. At the end of the cell add:

```python
print("Feasibility audit functions loaded.")
```

- [ ] **Step 2: Run Cell 10 and verify**

Expected output:
```
Feasibility audit functions loaded.
```
No NameError — all referenced modules (`numpy`, `pandas`) are already imported from Cell 8.

- [ ] **Step 3: Commit**

```bash
git add notebooks/mask_ablation.ipynb
git commit -m "feat(notebook): add feasibility audit functions (copied from reference)"
```

---

## Task 7: Cell 11 — Run feasibility audit on all variants (seed=42 only)

**Files:**
- Modify: `notebooks/mask_ablation.ipynb` (add Cell 11)

**Rationale:** Full audit on one seed keeps runtime manageable. Robust metrics are multi-seed (Task 4/5); feasibility is seed=42 only per spec §4.

- [ ] **Step 1: Add Cell 11**

```python
# Cell 11: Feasibility audit on seed=42 adversarial examples (all variants)
AUDIT_SEED = 42

# Load seed-42 preprocessor and splits once
dataset = load_dataset("lcld", config={"sample_frac": SAMPLE_FRAC})
_, _, X_test, _, _, y_test = split_dataset(
    dataset, test_size=0.2, val_size=0.2, random_state=AUDIT_SEED,
)
preprocessor = DataPreprocessor.load(
    get_preprocessor_path("lcld", AUDIT_SEED, len(dataset.X))
)
X_test_p = preprocessor.transform(X_test)

# Extract scaler the same way the reference notebook does (Cell 11).
# The preprocessor wraps a sklearn ColumnTransformer; the numeric branch is a
# Pipeline with a "scaler" step.
num_feature_names = []
num_transformer = None
for name, transformer, columns in preprocessor.pipeline.transformers_:
    if name == "num":
        num_transformer = transformer
        num_feature_names = list(columns)
        break
assert num_transformer is not None, "Could not locate numeric transformer in preprocessor"
scaler = num_transformer.named_steps["scaler"]

VARIANT_FILES = {
    "M0":        f"{BASELINE_LOCAL}/lcld_neural_unmasked_seed{AUDIT_SEED}.parquet",
    "M1":        f"{BASELINE_LOCAL}/lcld_neural_masked_seed{AUDIT_SEED}.parquet",
    "M2":        f"{ADV_SAVE_DIR}/lcld_neural_M2_seed{AUDIT_SEED}.parquet",
    "M3":        f"{ADV_SAVE_DIR}/lcld_neural_M3_seed{AUDIT_SEED}.parquet",
    "M4":        f"{ADV_SAVE_DIR}/lcld_neural_M4_seed{AUDIT_SEED}.parquet",
    "M5":        f"{ADV_SAVE_DIR}/lcld_neural_M5_seed{AUDIT_SEED}.parquet",
    "M6strict":  f"{ADV_SAVE_DIR}/lcld_neural_M6strict_seed{AUDIT_SEED}.parquet",
    "M6relaxed": f"{ADV_SAVE_DIR}/lcld_neural_M6relaxed_seed{AUDIT_SEED}.parquet",
}

feas_rows = []
for vname, path in VARIANT_FILES.items():
    assert os.path.exists(path), f"Missing parquet: {path}"
    X_adv_p = pd.read_parquet(path)

    # Inverse-transform numeric features to raw space; OHE columns stay in processed space.
    X_adv_raw = inverse_transform_numeric(X_adv_p, num_feature_names, scaler)
    # Reconstruct term from OHE so g1 (installment formula) has a raw term value.
    X_adv_raw["term"] = reconstruct_term_from_ohe(X_adv_p)

    agg, per_constraint = compute_aggregate_feasibility(X_adv_raw, X_proc=X_adv_p)
    feas_rows.append({
        "variant": vname,
        "g1_installment": per_constraint["g1"],
        "g2_open_total":  per_constraint["g2"],
        "g3_bankruptcy":  per_constraint["g3"],
        "g4_ohe":         per_constraint.get("g4_processed", per_constraint.get("g4", np.nan)),
        "aggregate":      agg,
    })

feas_df = pd.DataFrame(feas_rows)
feas_df.to_csv(os.path.join(ADV_SAVE_DIR, "mask_ablation_feasibility.csv"), index=False)
print(feas_df.to_string(index=False))
```

- [ ] **Step 2: Run Cell 11 and verify**

Expected:
- 8 rows (one per variant).
- M0 row reproduces the baseline numbers (g1 ≈ 0.02, g4 ≈ 0.19, aggregate ≈ 0.001) — within ±0.005.
- M4/M5 rows should show `g4_ohe ≈ 1.0` (term OHE frozen → always valid).
- M6strict may show higher g1 pass if fewer features move, but aggregate still depends on g1's installment formula coupling.

If numbers for M0/M1 deviate from the baseline by more than 0.01, STOP — the audit code or scaler is mismatched; investigate before proceeding.

- [ ] **Step 3: Commit**

```bash
git add notebooks/mask_ablation.ipynb results/adv_examples/mask_ablation/mask_ablation_feasibility.csv
git commit -m "feat(notebook): run feasibility audit across all mask variants"
```

---

## Task 8: Cell 12 — Perturbation statistics (seed=42)

**Files:**
- Modify: `notebooks/mask_ablation.ipynb` (add Cell 12)

**Rationale:** Quantify how much each variant actually moves key features. Reveals whether a "masked" feature is truly frozen and whether M2's direction clip worked (no negative emp_length deltas).

- [ ] **Step 1: Add Cell 12**

```python
# Cell 12: Perturbation statistics per variant (seed=42)
KEY_FEATURES = [
    "loan_amnt", "annual_inc", "dti", "emp_length",
    "int_rate", "installment",
]

# Original (clean) raw features for seed=42 are X_test; processed equivalent is X_test_p
X_test_raw_scaled = X_test_p  # processed (scaled + OHE)
X_test_raw = inverse_transform_numeric(X_test_p, num_feature_names, scaler)

pert_rows = []
for vname, path in VARIANT_FILES.items():
    X_adv_p = pd.read_parquet(path)
    X_adv_raw = inverse_transform_numeric(X_adv_p, num_feature_names, scaler)

    row = {"variant": vname}
    for feat in KEY_FEATURES:
        if feat in X_test_raw.columns and feat in X_adv_raw.columns:
            delta = (X_adv_raw[feat] - X_test_raw[feat]).abs()
            row[f"{feat}_mean_abs_delta"] = delta.mean()
            row[f"{feat}_pct_changed"] = (delta > 1e-6).mean()
        else:
            row[f"{feat}_mean_abs_delta"] = np.nan
            row[f"{feat}_pct_changed"] = np.nan

    # Term OHE: report max absolute delta across term_* columns in processed space
    term_cols = [c for c in X_adv_p.columns if c.startswith("term_")]
    if term_cols:
        row["term_ohe_max_abs_delta"] = (
            (X_adv_p[term_cols] - X_test_p[term_cols]).abs().max(axis=1).mean()
        )
    else:
        row["term_ohe_max_abs_delta"] = np.nan

    # M2-specific sanity: emp_length deltas should be >= 0 (increase-only)
    if vname == "M2" and "emp_length" in X_adv_raw.columns:
        raw_delta = X_adv_raw["emp_length"] - X_test_raw["emp_length"]
        row["emp_length_min_raw_delta"] = raw_delta.min()
        row["emp_length_pct_negative"] = (raw_delta < -1e-6).mean()

    pert_rows.append(row)

pert_df = pd.DataFrame(pert_rows)
pert_df.to_csv(os.path.join(ADV_SAVE_DIR, "mask_ablation_perturbation_stats.csv"), index=False)
with pd.option_context("display.max_columns", None, "display.width", 200):
    print(pert_df)
```

- [ ] **Step 2: Run Cell 12 and verify**

Expected:
- M3 row: `dti_mean_abs_delta ≈ 0` and `dti_pct_changed ≈ 0` (dti frozen).
- M4 row: `term_ohe_max_abs_delta ≈ 0`.
- M5 row: both of the above.
- M2 row: `emp_length_min_raw_delta >= 0` (with small tolerance for float noise) and `emp_length_pct_negative` near 0.
- If the M2 `pct_negative` is not near zero, the directional clip is broken — go back to Task 3 and fix before continuing.

- [ ] **Step 3: Commit**

```bash
git add notebooks/mask_ablation.ipynb results/adv_examples/mask_ablation/mask_ablation_perturbation_stats.csv
git commit -m "feat(notebook): add per-variant perturbation statistics"
```

---

## Task 9: Cell 13 — E1 cost-weighted evaluation

**Files:**
- Modify: `notebooks/mask_ablation.ipynb` (add Cell 13)

**Rationale:** Measures how "expensive" each successful attack is for the attacker. Re-uses existing M0/M1 parquets — no new attack runs. Includes ×2/×0.5 sensitivity check.

- [ ] **Step 1: Add Cell 13**

```python
# Cell 13: E1 cost-weighted evaluation (M0 vs M1, seed=42)
import matplotlib.pyplot as plt

# Use training-set ranges for normalization, with winsorization to resist outliers.
# p1/p99 (not min/max) — LCLD annual_inc has extreme tails.
X_train = split_dataset(dataset, test_size=0.2, val_size=0.2, random_state=AUDIT_SEED)[0]
feature_ranges = {}
for feat in FEATURE_COSTS:
    if feat in X_train.columns and np.issubdtype(X_train[feat].dtype, np.number):
        p1, p99 = np.nanpercentile(X_train[feat], [1, 99])
        feature_ranges[feat] = max(p99 - p1, 1e-6)
    else:
        # Categorical: use 1.0 so any change contributes full cost weight
        feature_ranges[feat] = 1.0

def total_cost(X_orig_raw: pd.DataFrame, X_adv_raw: pd.DataFrame,
               costs: dict, ranges: dict) -> pd.Series:
    total = pd.Series(0.0, index=X_adv_raw.index)
    for feat, c in costs.items():
        if feat not in X_adv_raw.columns or feat not in X_orig_raw.columns:
            continue
        if np.issubdtype(X_adv_raw[feat].dtype, np.number):
            delta_norm = (X_adv_raw[feat] - X_orig_raw[feat]).abs() / ranges[feat]
        else:
            delta_norm = (X_adv_raw[feat] != X_orig_raw[feat]).astype(float)
        total = total + c * delta_norm
    return total

e1_targets = ["M0", "M1"]
e1_costs = {}
for vname in e1_targets:
    X_adv_p = pd.read_parquet(VARIANT_FILES[vname])
    X_adv_raw = inverse_transform_numeric(X_adv_p, num_feature_names, scaler)
    X_adv_raw["term"] = reconstruct_term_from_ohe(X_adv_p)
    e1_costs[vname] = total_cost(X_test_raw, X_adv_raw, FEATURE_COSTS, feature_ranges)

# Histogram
fig, ax = plt.subplots(figsize=(8, 4))
for vname, costs_s in e1_costs.items():
    ax.hist(costs_s, bins=50, alpha=0.5, label=vname, density=True)
ax.set_xlabel("Total attack cost (normalized)")
ax.set_ylabel("Density")
ax.legend()
ax.set_title("E1: Attack cost distribution (seed=42)")
plt.tight_layout()
plt.savefig(os.path.join(ADV_SAVE_DIR, "e1_cost_distribution.png"), dpi=150)
plt.show()

# Affordable curve
fig, ax = plt.subplots(figsize=(8, 4))
budgets = np.linspace(0, max(c.max() for c in e1_costs.values()), 200)
for vname, costs_s in e1_costs.items():
    frac = [(costs_s <= B).mean() for B in budgets]
    ax.plot(budgets, frac, label=vname)
ax.set_xlabel("Attacker cost budget")
ax.set_ylabel("Fraction of attacks affordable")
ax.legend()
ax.set_title("E1: Affordable attack fraction vs budget")
plt.tight_layout()
plt.savefig(os.path.join(ADV_SAVE_DIR, "e1_affordable_curve.png"), dpi=150)
plt.show()

# Summary table (base, ×2, ×0.5 sensitivity)
sensitivity_rows = []
for scale in (1.0, 2.0, 0.5):
    scaled_costs = {k: v * scale for k, v in FEATURE_COSTS.items()}
    for vname in e1_targets:
        X_adv_p = pd.read_parquet(VARIANT_FILES[vname])
        X_adv_raw = inverse_transform_numeric(X_adv_p, num_feature_names, scaler)
        X_adv_raw["term"] = reconstruct_term_from_ohe(X_adv_p)
        s = total_cost(X_test_raw, X_adv_raw, scaled_costs, feature_ranges)
        sensitivity_rows.append({
            "variant": vname, "cost_scale": scale,
            "mean": s.mean(), "median": s.median(), "p95": s.quantile(0.95),
        })

e1_summary = pd.DataFrame(sensitivity_rows)
e1_summary.to_csv(os.path.join(ADV_SAVE_DIR, "e1_cost_summary.csv"), index=False)
print(e1_summary)
```

- [ ] **Step 2: Run Cell 13 and verify**

Expected:
- Two PNGs saved in `results/adv_examples/mask_ablation/`.
- `e1_summary` has 6 rows (2 variants × 3 scales).
- Under scale=1.0, medians and p95 should be strictly positive and M0's cost >= M1's (M0 has more mutable features, so more delta accumulates) — flag in the writeup if violated.
- Sensitivity check: mean/median/p95 at scale=2.0 should be exactly 2× the scale=1.0 values (within float tolerance). This is a correctness check on the scaling math.

- [ ] **Step 3: Commit**

```bash
git add notebooks/mask_ablation.ipynb results/adv_examples/mask_ablation/e1_*.csv results/adv_examples/mask_ablation/e1_*.png
git commit -m "feat(notebook): add E1 cost-weighted attack evaluation"
```

---

## Task 10: Cell 14 — Summary table and Drive backup

**Files:**
- Modify: `notebooks/mask_ablation.ipynb` (add Cell 14)

**Rationale:** Produce the mean±std table requested in spec §5.2 and mirror results to Drive so the advisor can access them.

- [ ] **Step 1: Add Cell 14**

```python
# Cell 14: Final summary table + Drive backup
VARIANT_ORDER = ["M0", "M1", "M2", "M3", "M4", "M5", "M6strict", "M6relaxed"]

agg = (
    all_results.groupby("variant")
    .agg({
        "n_mutable": "mean",
        "robust_pr_auc":   ["mean", "std"],
        "robust_accuracy": ["mean", "std"],
        "robust_recall":   ["mean", "std"],
        "robust_f1":       ["mean", "std"],
    })
)
agg.columns = ["_".join(c).strip("_") for c in agg.columns]
agg = agg.reindex(VARIANT_ORDER)

# Merge in seed=42 feasibility numbers (not averaged across seeds — per spec §4)
feas_idx = feas_df.set_index("variant")
agg["feasibility_seed42"] = feas_idx["aggregate"]
agg["g1_pass_seed42"] = feas_idx["g1_installment"]
agg["g4_pass_seed42"] = feas_idx["g4_ohe"]

agg.to_csv(os.path.join(ADV_SAVE_DIR, "mask_ablation_summary.csv"))
with pd.option_context("display.max_columns", None, "display.width", 220,
                       "display.float_format", "{:.4f}".format):
    print(agg)

# --- Drive backup ---
DRIVE_DEST = "/content/drive/MyDrive/FraudBench/results/mask_ablation"
os.makedirs(DRIVE_DEST, exist_ok=True)
for fname in os.listdir(ADV_SAVE_DIR):
    src = os.path.join(ADV_SAVE_DIR, fname)
    dst = os.path.join(DRIVE_DEST, fname)
    shutil.copy(src, dst)
print(f"\nBacked up {len(os.listdir(ADV_SAVE_DIR))} files to {DRIVE_DEST}")
```

- [ ] **Step 2: Run Cell 14 and verify**

Expected:
- Printed `agg` has 8 rows in order M0..M6relaxed.
- `feasibility_seed42`, `g1_pass_seed42`, `g4_pass_seed42` columns populated for every variant.
- Backup line reports a file count in the 15–20 range (parquets + CSVs + PNGs).
- M0 row's `robust_accuracy_mean ≈ 0.055` and `robust_pr_auc_mean ≈ 0.1051`, matching the baseline numbers in the spec §5.2 table.

- [ ] **Step 3: Commit**

```bash
git add notebooks/mask_ablation.ipynb results/adv_examples/mask_ablation/mask_ablation_summary.csv
git commit -m "feat(notebook): add summary table and Drive backup"
```

---

## Task 11: Update findings doc with cross-checks and caveats

**Files:**
- Modify: `docs/plans/mask_ablation_experiment_plan.md` (append a "Results" section) OR create `docs/findings_mask_ablation.md` following the convention of existing `findings_*.md` files in `docs/`.

**Rationale:** Lock in the advisor-meeting narrative. Numbers without a writeup get forgotten.

- [ ] **Step 1: Decide on the target file**

Check `docs/` for existing findings docs. If one per experiment is the convention, create `docs/findings_mask_ablation.md`. Otherwise append a `## Results` section to `docs/plans/mask_ablation_experiment_plan.md`.

- [ ] **Step 2: Populate the findings doc**

Structure:

```markdown
# Mask Ablation — Findings

**Date:** 2026-04-15
**Notebook:** `notebooks/mask_ablation.ipynb`
**Canonical results:** `results/adv_examples/mask_ablation/mask_ablation_summary.csv`

## Headline numbers

Paste the `mask_ablation_summary.csv` content as a markdown table.

## Sanity checks performed

- M0/M1 numbers in this run reproduce `comparison_{unmasked,masked}.csv` within ±0.01 on robust_pr_auc / robust_accuracy. [Verify from Cell 9 output.]
- M4/M5 achieve g4 ≈ 1.0 (term OHE frozen by construction — not a capability improvement).
- M2 emp_length raw delta min ≥ -1e-6 (directional clip holds).
- E1 cost at scale 2.0 is exactly 2× scale 1.0 (math sanity).

## Caveats

- Feasibility numbers are seed=42 only (per spec §4).
- M4 g4 = 1.0 is an artifact of freezing, not CAPGD learning to produce valid OHE. Framing: "freezing term recovers the aggregate feasibility that was blocked solely by g4", NOT "mask-level fixes solve OHE".
- E1 uses p1/p99 winsorized ranges rather than min/max to tame LCLD `annual_inc` tails.
- With 3 seeds the variance bands on robust metrics will overlap for most (variant, variant) pairs — flag the pairs where mean differs by >1 seed-level std as the only ones with signal.

## Answers to meeting questions (spec §9)

1. Biggest marginal effect on robust accuracy: [fill from numbers].
2. Does M4 lift aggregate feasibility? [fill].
3. Attacker-capability spectrum M6-strict vs M1: [fill].
4. E1 takeaway: [fill].
5. Next round proposal: [fill based on whether M5 feasibility is near-zero → constraint-aware attacks, or large → mask sufficiency].
```

Fill placeholders using the actual numbers from `mask_ablation_summary.csv`, `mask_ablation_feasibility.csv`, and `e1_cost_summary.csv`.

- [ ] **Step 3: Commit**

```bash
git add docs/findings_mask_ablation.md   # or docs/plans/mask_ablation_experiment_plan.md
git commit -m "docs: add mask ablation findings writeup for advisor meeting"
```

---

## Task 12: Final verification pass

**Files:** none (read-only verification)

- [ ] **Step 1: Re-run the notebook top-to-bottom on a fresh Colab runtime**

Runtime → Restart runtime → Run all. Confirm every cell completes without error. Total wall-clock budget: ~8 min (5 min experiment + 3 min overhead).

- [ ] **Step 2: Cross-check baseline reproduction**

Open `mask_ablation_results.csv`. Filter to `variant in {M0, M1}` and compare `robust_pr_auc` / `robust_accuracy` against the M0/M1 rows printed in the reference notebook (from the `comparison_{unmasked,masked}.csv` files). Max absolute difference should be 0 (we loaded them verbatim). If non-zero, something in Task 5 re-computed them — fix before shipping.

- [ ] **Step 3: Cross-check every deliverable listed in the plan exists**

```python
import os
deliverables = [
    "results/adv_examples/mask_ablation/mask_ablation_results.csv",
    "results/adv_examples/mask_ablation/mask_ablation_summary.csv",
    "results/adv_examples/mask_ablation/mask_ablation_feasibility.csv",
    "results/adv_examples/mask_ablation/mask_ablation_perturbation_stats.csv",
    "results/adv_examples/mask_ablation/e1_cost_summary.csv",
    "results/adv_examples/mask_ablation/e1_cost_distribution.png",
    "results/adv_examples/mask_ablation/e1_affordable_curve.png",
]
for seed in (42, 123, 456):
    for v in ("M2", "M3", "M4", "M5", "M6strict", "M6relaxed"):
        deliverables.append(f"results/adv_examples/mask_ablation/lcld_neural_{v}_seed{seed}.parquet")
missing = [p for p in deliverables if not os.path.exists(p)]
assert not missing, f"Missing deliverables: {missing}"
print(f"All {len(deliverables)} deliverables present.")
```

- [ ] **Step 4: Confirm Drive backup**

```python
drive_files = os.listdir("/content/drive/MyDrive/FraudBench/results/mask_ablation")
print(f"Drive has {len(drive_files)} files.")
assert len(drive_files) >= 25, "Drive backup incomplete"
```

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: final mask ablation run — all deliverables verified" || echo "nothing to commit"
```

---

## Notes on known-risk items (from spec §8 analysis)

- **emp_length processed representation.** This plan treats `emp_length` as numeric (matching the raw feature_types setup in existing preprocessors). If it is somehow OHE-encoded in a future seed, `resolve_direction_indices` will still find it via prefix matching, so the directional clip still applies. The `term_ohe_max_abs_delta` check in Task 8 will expose any surprise.
- **dti_joint is NaN for most rows.** After preprocessor imputation it becomes a near-constant column. Freezing it (M3) is therefore a near-no-op by itself; the M3 effect in the summary table should be read alongside this caveat.
- **M6 immutable-set construction** uses `set(dataset.X.columns) - profile`. `dataset.X.columns` already excludes the target (verified from the reference notebook run log: `features=63`, same as the spec assumed).
- **Runtime will likely exceed the spec's 4-min estimate** because feasibility-audit constraint checks on 26k rows × 8 variants are not instant. Budget 8 min total, not 4.
