# NB2 — ICDM Square Model-Family Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `notebooks/icdm_square_model_family.ipynb` (NB2, spec §5): 36 Square-attack runs covering MLP/XGBoost/Ensemble × 4 datasets × 3 seeds at Protocol A, with Protocol B derived from saved adversarial parquet, N/A placeholder rows for the deferred tree/ensemble Protocol C, the §3.3 summary, and `fig_model_family`.

**Architecture:** 15-cell Colab notebook mirroring NB1 (`notebooks/icdm_capgd_protocol_grid.ipynb`) cell-for-cell. Bootstrap cells 1–5 are copied byte-identical from NB1 by the builder script. All outputs (results CSV + adversarial parquet + figures) write **directly to Google Drive** so the ~6 h loop is resumable across Colab sessions. Everything reuses existing repo modules — the only new logic is a three-family `model_hash` and the NB2 run loop.

**Tech Stack:** Python notebook (built as raw JSON, no nbformat dep), `attacks/square.py` (ART SquareAttack), `models/neural.py` / `models/tree.py` / `defences/ensemble.py`, `constraints/feasibility.py`, `evaluation/metrics.py`.

---

## Context and locked decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| XGBoost config | `{max_depth: 6, n_estimators: 100, learning_rate: 0.1}`, **no** `scale_pos_weight` (user-ratified) | Byte-identical to `configs/*_tree_square.yaml` which produced the prior Square registry rows → informal old-vs-new comparability. Spec §1.3's `scale_pos_weight` parenthetical refers to the Ensemble's internal XGB (`defences/ensemble.py:64-73`). |
| Protocol-C placeholders | Emit one placeholder row per (dataset, XGBoost/Ensemble, seed) = 24 rows, NaN metrics, Cartella note (user-ratified). **Encoded as `protocol="not_applicable"`**, not the literal string `"N/A"` | Spec §5.2 compliance; NB3's coverage table shows the gap explicitly. Codex review caught that pandas' default `na_values` converts a literal `"N/A"` cell to `NaN` on read-back, so `protocol != "N/A"` filters silently leak placeholder rows into metrics. `not_applicable` is pandas-safe; NB3 must filter `protocol != "not_applicable"`. |
| Output location | Drive-direct (`/content/drive/MyDrive/FraudBench/results/...`), local repo fallback when Drive absent | Spec §8 anticipates splitting across Colab sessions; per-run CSV append + parquet on Drive makes resume automatic. Replaces NB1's end-of-run backup cell with a session-status cell. |
| Per-constraint CSV | **None** for NB2 | Spec §3.4 lists only `square_family_results.csv / _summary.csv`. Feasibility scalars (`fsr`, `aggregate_feasibility`, `main_failed_constraint`) still come from `evaluate_feasibility`. |
| Protocol A semantics | Reuse `attacks/square.py` as-is | Its ART clip_values + post-attack numeric clip = processed-schema-bounds envelope, the same envelope stock CAPGD applies per step in NB1's Protocol A. No feasibility constraints added. |

Expected output: **96 rows** in `square_family_results.csv` (36 A + 36 derived B + 24 N/A), 72 parquet files, `square_family_summary.csv`, `fig_model_family.{pdf,png}`.

Historical Square timings (old registry, same `sample_frac=0.1`): CCFD ~1 min, IEEE-CIS ~13–15 min, LCLD ~7–14 min, Sparkov ~10–15 min per run → ~1.5–1.8 h per seed across all 12 (dataset, model) pairs, ~5–6 h total.

## File structure

- Create: `notebooks/icdm_square_model_family.ipynb` (the deliverable; built by a throwaway script)
- Throwaway (not committed): `/tmp/build_nb2.py` (notebook builder), `/tmp/nb2_smoke.py` (smoke test)
- No repo module changes. `model_hash` lives inline in the notebook, matching NB1's inline `weight_hash` precedent.

---

### Task 1: Build the notebook

**Files:**
- Create: `notebooks/icdm_square_model_family.ipynb` (via `/tmp/build_nb2.py`)

- [ ] **Step 1: Write the builder script to `/tmp/build_nb2.py`**

The script copies bootstrap cells 1–5 byte-identical from NB1 (outputs stripped) and defines cells 0 and 6–14 as literal strings. Run it from the repo root.

```python
"""Build notebooks/icdm_square_model_family.ipynb (NB2, spec §5).

Bootstrap cells 1-5 are copied byte-identical from NB1 (outputs stripped);
cells 0 and 6-14 are defined below. Pure-json: no nbformat dependency.
Run from the repo root: .venv/bin/python /tmp/build_nb2.py
"""
import json

MD0 = """# NB2 — ICDM Square Model-Family

**The model-family axis (spec §5).** Covers MLP / XGBoost / Ensemble on the
model-agnostic Square attack (score-based black-box, 100-query budget, ε=0.1)
at Protocol A and derived Protocol B — the cross-model-comparable table
TabularBench cannot produce. 36 Square runs (4 datasets × 3 models × 3 seeds),
defence = `none`, ~6 h total (the only real compute in the ICDM scope).

- **A_unconstrained** — stock Square (processed-schema clipping only).
- **B_posthoc_filter** — feasibility filter on A's saved examples (infeasible → revert to clean). *Derived, no new attack.*
- **Protocol C is NOT run here:** white-box in-attack constraints are MLP/CAPGD-only (NB1);
  Cartella-style black-box in-attack C for XGBoost/Ensemble is deferred to future work —
  recorded as `protocol = "N/A"` placeholder rows (spec §5.2).

**Multi-session safe:** all outputs (CSV + parquet + figures) write directly to
Drive; re-running the notebook resumes from the completed `run_id`s. Cell 13
prints what remains.

**Bootstrap cells 1–5** mirror NB1 (Drive mount, repo clone, deps install → restart, dataset symlinks).

Outputs: adversarial parquet under `FraudBench/results/adv_examples/icdm_square_family/`;
`square_family_results.csv` / `_summary.csv` + `fig_model_family` under `FraudBench/results/icdm_2026/`."""

C6 = '''# Cell 6: Configuration and imports
import os, time, random, hashlib
import numpy as np
import pandas as pd
import torch

from datasets.loader import load_dataset
from datasets.splitter import split_dataset
from preprocessing.processor import DataPreprocessor
from constraints.schema import ConstraintSchema
from constraints.validator import EVAL_TOL
from constraints.feasibility import evaluate_feasibility
from models.neural import NeuralModel
from models.tree import TreeModel
from defences.ensemble import EnsembleModel
from attacks.square import square_attack
from evaluation.metrics import compute_metrics

# --- experiment axes (spec §5.2) ---
SEEDS = [42, 123, 456]
EPS = 0.1
SQUARE_PARAMS = {"epsilon": EPS, "max_iter": 100, "norm": "inf"}
SAMPLE_FRAC = 0.1                          # matches NB1 + the prior Square runs

# Per-family training params — reuse the repo configs that produced the prior
# Square registry rows (configs/*_tree_square.yaml, *_ensemble_square.yaml) and
# NB1's MLP (clean PR-AUC anchors, spec §5.3). NB: scale_pos_weight lives only
# inside the Ensemble's XGB component (defences/ensemble.py), not TreeModel.
MLP_PARAMS = {"epochs": 20, "hidden_dim": 128, "batch_size": 256, "lr": 0.001}
XGB_PARAMS = {"max_depth": 6, "n_estimators": 100, "learning_rate": 0.1}
ENS_PARAMS = {"epochs": 15, "hidden_dim": 128, "batch_size": 256, "lr": 0.001}

# (registry_name, loader_name)
DATASETS = [
    ("CCFD", "ccfd"),
    ("IEEE-CIS", "ieee_cis"),
    ("LCLD", "lcld"),
    ("Sparkov", "sparkov"),
]
MODELS = ["MLP", "XGBoost", "Ensemble"]

NB, DEFENCE, ATTACK = "nb2_square_family", "none", "Square"
# Spec §5.2's "protocol = N/A" placeholder marker. NOT the literal string "N/A":
# pandas' default na_values would convert that cell to NaN on read-back, and
# `protocol != "N/A"` filters would silently keep the placeholder rows.
PROTO_NA = "not_applicable"

# Outputs go straight to Drive when mounted (multi-session resume, spec §8);
# fall back to the local repo tree otherwise.
DRIVE_ROOT = "/content/drive/MyDrive/FraudBench"
ROOT = DRIVE_ROOT if os.path.isdir(DRIVE_ROOT) else "."
ADV_DIR = os.path.join(ROOT, "results/adv_examples/icdm_square_family")
OUT_DIR = os.path.join(ROOT, "results/icdm_2026")
FIG_DIR = os.path.join(OUT_DIR, "figures")
MODELS_DIR = os.path.join(ROOT, "results/models/icdm_square_family")  # spec §1.9.1: persist weights
for d in (ADV_DIR, OUT_DIR, FIG_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)
RESULTS_CSV = os.path.join(OUT_DIR, "square_family_results.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "square_family_summary.csv")

# Canonical long-format schema (spec §3.1).
RESULTS_COLUMNS = [
    "run_id", "notebook", "dataset", "model", "defence", "attack", "protocol",
    "seed", "epsilon", "same_model_group_id", "model_weight_hash", "n_test",
    "clean_pr_auc", "robust_pr_auc", "clean_roc_auc", "robust_roc_auc",
    "clean_accuracy", "robust_accuracy", "flipped_count", "feasible_count",
    "feasible_flipped_count", "fsr", "aggregate_feasibility",
    "main_failed_constraint", "attack_runtime_sec", "notes",
]

print(f"EVAL_TOL = {EVAL_TOL}; outputs -> {os.path.abspath(ROOT)}")
print(f"Plan: {len(DATASETS)} datasets x {len(MODELS)} models x {len(SEEDS)} seeds = "
      f"{len(DATASETS) * len(MODELS) * len(SEEDS)} Square runs (A) + derived B")'''

C7 = '''# Cell 7: Reproducibility helpers + same-model training per group
def set_all_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_hash(model) -> str:
    """Stable 16-hex hash over the fitted parameters of any model family
    (spec §1.9): torch state_dict bytes for the MLP, the XGBoost booster's
    UBJSON dump for the tree, and the concatenation of all three sub-models
    (LR coefficients + booster + MLP weights) for the ensemble."""
    if isinstance(model, EnsembleModel):
        buf = (
            model.lr_model.coef_.tobytes() + model.lr_model.intercept_.tobytes()
            + bytes(model.xgb_model.get_booster().save_raw(raw_format="ubj"))
            + b"".join(p.detach().cpu().numpy().tobytes()
                       for p in model.mlp.state_dict().values())
        )
    elif isinstance(model, TreeModel):
        buf = bytes(model.model.get_booster().save_raw(raw_format="ubj"))
    else:  # NeuralModel
        buf = b"".join(p.detach().cpu().numpy().tobytes()
                       for p in model.model.state_dict().values())
    return hashlib.sha256(buf).hexdigest()[:16]


def build_model(model_name):
    if model_name == "MLP":
        return NeuralModel(dict(MLP_PARAMS))
    if model_name == "XGBoost":
        return TreeModel(dict(XGB_PARAMS))
    if model_name == "Ensemble":
        return EnsembleModel(dict(ENS_PARAMS))
    raise ValueError(f"Unknown model {model_name!r}")


def train_group(registry_name, loader_name, model_name, seed):
    """Train ONE model for a (dataset, model, seed) group (defence = none);
    reused by Protocol A's attack and B's derivation. Weights persist to
    MODELS_DIR (spec §1.9.1) so a multi-session resume reloads the *same*
    fitted model instead of retraining — the same-model hash is exact across
    sessions. Returns the fitted model, preprocessor, processed test split,
    the processed-space schema (Square's clip values), cached clean
    predictions, and the hash."""
    set_all_seeds(seed)
    dataset = load_dataset(loader_name, config={"sample_frac": SAMPLE_FRAC})
    X_train, _X_val, X_test, y_train, _y_val, y_test = split_dataset(
        dataset, test_size=0.2, val_size=0.2, random_state=seed
    )
    pre = DataPreprocessor(dataset.feature_types)
    X_train_p = pre.fit_transform(X_train)
    X_test_p = pre.transform(X_test)
    proc_ft = {c: "numeric" for c in X_train_p.columns}
    schema = ConstraintSchema.from_data(X_train_p, proc_ft)

    model = build_model(model_name)
    ext = ".pt" if model_name == "MLP" else ".joblib"
    ckpt = os.path.join(MODELS_DIR, f"{registry_name}__{model_name}__{DEFENCE}__s{seed}{ext}")
    t0 = time.time()
    if os.path.exists(ckpt):
        model = type(model).load(ckpt)
        verb = "loaded"
    else:
        model.fit(X_train_p, y_train)
        model.save(ckpt)
        verb = "trained"
    train_time = time.time() - t0

    grp = dict(
        model=model, pre=pre, schema=schema, proc_ft=proc_ft,
        X_test_p=X_test_p, y_test=y_test,
        clean_probs=model.predict_proba(X_test_p), hash=model_hash(model),
    )
    print(f"  {verb} {registry_name}/{model_name}/s{seed} in {train_time:.1f}s  "
          f"hash={grp['hash']}  n_test={len(y_test)}  proc_dim={X_test_p.shape[1]}")
    return grp'''

C8 = '''# Cell 8: Row builder (spec §3.1), Protocol-B derivation, CSV append
def make_run_id(reg, model_name, seed, eps, protocol):
    return f"{NB}__{reg}__{model_name}__{DEFENCE}__s{seed}__e{eps}__{protocol}"


def build_row(grp, reg, model_name, seed, protocol, X_adv, runtime):
    """One §3.1 results row. flipped_count = positives the clean model caught
    (pred=1) that the attack flipped to negative; feasible_flipped additionally
    requires the adv row to pass the full feasibility conjunction (identical
    definitions to NB1's build_rows)."""
    adv_probs = grp["model"].predict_proba(X_adv)
    clean_m = compute_metrics(grp["y_test"], grp["clean_probs"])
    rob_m = compute_metrics(grp["y_test"], adv_probs)
    feas = evaluate_feasibility(reg, X_adv, preprocessor=grp["pre"])

    yv = grp["y_test"].values
    clean_pred = (grp["clean_probs"] >= 0.5).astype(int)
    adv_pred = (adv_probs >= 0.5).astype(int)
    pos = yv == 1
    fmask = feas.feasible_row_mask.values
    flipped = int(((clean_pred == 1) & (adv_pred == 0) & pos).sum())
    feas_flipped = int(((clean_pred == 1) & (adv_pred == 0) & pos & fmask).sum())
    fsr = (feas_flipped / flipped) if flipped > 0 else float("nan")

    row = {
        "run_id": make_run_id(reg, model_name, seed, EPS, protocol),
        "notebook": NB, "dataset": reg, "model": model_name,
        "defence": DEFENCE, "attack": ATTACK, "protocol": protocol, "seed": seed,
        "epsilon": EPS, "same_model_group_id": f"{reg}__{model_name}__{DEFENCE}__s{seed}",
        "model_weight_hash": model_hash(grp["model"]), "n_test": int(len(yv)),
        "clean_pr_auc": clean_m["pr_auc"], "robust_pr_auc": rob_m["pr_auc"],
        "clean_roc_auc": clean_m["roc_auc"], "robust_roc_auc": rob_m["roc_auc"],
        "clean_accuracy": clean_m["accuracy"], "robust_accuracy": rob_m["accuracy"],
        "flipped_count": flipped, "feasible_count": int(fmask.sum()),
        "feasible_flipped_count": feas_flipped, "fsr": fsr,
        "aggregate_feasibility": feas.aggregate_feasibility,
        "main_failed_constraint": feas.main_failed_constraint,
        "attack_runtime_sec": round(runtime, 3), "notes": "",
    }
    return row, feas


def derive_protocol_B(A_adv, A_feas_mask, X_test_p):
    """Protocol B: keep feasible A rows, revert infeasible rows to clean."""
    B = A_adv.copy()
    B.loc[~A_feas_mask] = X_test_p.loc[~A_feas_mask]
    return B


def append_csv(path, rows, columns=None):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if columns is not None:
        df = df[columns]
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)'''

C9 = '''# Cell 9: Main loop — one Square run (A) + derived B per (dataset, model, seed); resumable
done = set(pd.read_csv(RESULTS_CSV)["run_id"]) if os.path.exists(RESULTS_CSV) else set()
print(f"Resuming: {len(done)} run_ids already complete.")

t_session = time.time()
for reg, loader_name in DATASETS:
    for model_name in MODELS:
        for seed in SEEDS:
            a_rid = make_run_id(reg, model_name, seed, EPS, "A_unconstrained")
            b_rid = make_run_id(reg, model_name, seed, EPS, "B_posthoc_filter")
            if a_rid in done and b_rid in done:
                continue

            print(f"\\n=== {reg} | {model_name} | seed {seed} ===")
            grp = train_group(reg, loader_name, model_name, seed)
            parq = os.path.join(ADV_DIR, a_rid + ".parquet")

            if a_rid in done:
                # Session died between A and B last time: reuse A's parquet
                # against the checkpoint-reloaded model (hashes must agree).
                assert os.path.exists(parq), (
                    f"{a_rid} is in the CSV but its parquet is missing ({parq}). "
                    "Recovery: delete the A row from the CSV and re-run this group.")
                A_adv = pd.read_parquet(parq)
                stored = pd.read_csv(RESULTS_CSV).set_index("run_id").loc[a_rid, "model_weight_hash"]
                assert stored == grp["hash"], (
                    f"checkpoint hash {grp['hash']} != stored A hash {stored} for {a_rid}. "
                    "The Drive checkpoint was likely deleted and the model retrained. "
                    "Recovery: delete this group's A/B rows, parquet, and checkpoint, then re-run.")
                A_feas = evaluate_feasibility(reg, A_adv, preprocessor=grp["pre"]).feasible_row_mask
            else:
                t0 = time.time()
                A_adv = square_attack(grp["model"], grp["X_test_p"], grp["y_test"],
                                      grp["schema"], grp["proc_ft"], params=dict(SQUARE_PARAMS))
                rt = time.time() - t0
                A_adv.to_parquet(parq)
                A_adv = pd.read_parquet(parq)   # spec §1.9.3: B + feasibility derive from the
                                                # SAVED file; also validates the Drive write
                row, feas = build_row(grp, reg, model_name, seed, "A_unconstrained", A_adv, rt)
                assert row["model_weight_hash"] == grp["hash"], \\
                    f"model hash drift in {a_rid}: {row['model_weight_hash']} != {grp['hash']}"
                append_csv(RESULTS_CSV, [row], RESULTS_COLUMNS)
                done.add(a_rid)
                A_feas = feas.feasible_row_mask
                print(f"  [{'A_unconstrained':>16}] robPR={row['robust_pr_auc']:.3f}  "
                      f"flip={row['flipped_count']}  feasflip={row['feasible_flipped_count']}  "
                      f"FSR={row['fsr']:.3f}  agg={row['aggregate_feasibility']:.4f}  "
                      f"bind={row['main_failed_constraint']}  ({rt / 60:.1f} min)")

            if b_rid not in done:
                B_adv = derive_protocol_B(A_adv, A_feas, grp["X_test_p"])
                B_adv.to_parquet(os.path.join(ADV_DIR, b_rid + ".parquet"))
                brow, _ = build_row(grp, reg, model_name, seed, "B_posthoc_filter", B_adv, 0.0)
                brow["notes"] = "derived from A_unconstrained (post-hoc feasibility filter)"
                append_csv(RESULTS_CSV, [brow], RESULTS_COLUMNS)
                done.add(b_rid)
                print(f"  [{'B_posthoc_filter':>16}] robPR={brow['robust_pr_auc']:.3f}  "
                      f"flip={brow['flipped_count']}  feasflip={brow['feasible_flipped_count']}  "
                      f"FSR={brow['fsr']:.3f}")

print(f"\\nSession done in {(time.time() - t_session) / 3600:.2f} h. Total run_ids: {len(done)}")'''

C10 = '''# Cell 10: Protocol-C placeholder rows for XGBoost / Ensemble (spec §5.2)
# Black-box in-attack constraint enforcement (Cartella et al. 2021) is deferred:
# one explicit placeholder row per (dataset, tree-family model, seed) so the
# master registry and NB3's coverage table show the gap instead of silently
# omitting it. protocol = PROTO_NA ("not_applicable") encodes spec §5.2's
# "N/A" in a pandas-safe form; NB3 must filter protocol != PROTO_NA before
# computing metrics.
CARTELLA_NOTE = "Cartella-style black-box in-attack Protocol C deferred to future work"

res = pd.read_csv(RESULTS_CSV)
existing = set(res["run_id"])
a_rows = res[res["protocol"] == "A_unconstrained"].set_index("same_model_group_id")

NAN_COLS = ["clean_pr_auc", "robust_pr_auc", "clean_roc_auc", "robust_roc_auc",
            "clean_accuracy", "robust_accuracy", "flipped_count", "feasible_count",
            "feasible_flipped_count", "fsr", "aggregate_feasibility"]
na_rows = []
for reg, _loader in DATASETS:
    for model_name in ["XGBoost", "Ensemble"]:
        for seed in SEEDS:
            rid = make_run_id(reg, model_name, seed, EPS, PROTO_NA)
            gid = f"{reg}__{model_name}__{DEFENCE}__s{seed}"
            if rid in existing or gid not in a_rows.index:
                continue   # already emitted, or the A row doesn't exist yet
            a = a_rows.loc[gid]
            na_rows.append({
                "run_id": rid, "notebook": NB, "dataset": reg, "model": model_name,
                "defence": DEFENCE, "attack": ATTACK, "protocol": PROTO_NA, "seed": seed,
                "epsilon": EPS, "same_model_group_id": gid,
                "model_weight_hash": a["model_weight_hash"], "n_test": int(a["n_test"]),
                **{c: float("nan") for c in NAN_COLS},
                "main_failed_constraint": "", "attack_runtime_sec": float("nan"),
                "notes": CARTELLA_NOTE,
            })
append_csv(RESULTS_CSV, na_rows, RESULTS_COLUMNS)
print(f"Appended {len(na_rows)} Protocol-C placeholder rows (protocol={PROTO_NA}; "
      f"total expected: {len(DATASETS) * 2 * len(SEEDS)}).")'''

C11 = '''# Cell 11: Per-(dataset,model,protocol) summary over seeds (spec §3.3) + §5.3 sanity checks
res = pd.read_csv(RESULTS_CSV)
measured = res[res["protocol"] != PROTO_NA]
metrics_m = ["robust_pr_auc", "robust_roc_auc", "flipped_count",
             "feasible_flipped_count", "fsr", "aggregate_feasibility", "robust_accuracy"]
grp = measured.groupby(["dataset", "model", "defence", "protocol", "epsilon"])
summ = grp[metrics_m].agg(["mean", "std"])
summ.columns = [f"{stat}_{m}" for m, stat in summ.columns]
summ = summ.reset_index()
summ["n_seeds"] = grp.size().values
summ.to_csv(SUMMARY_CSV, index=False)
print(f"Saved summary ({len(summ)} groups) -> {SUMMARY_CSV}\\n")

cols = ["dataset", "model", "protocol", "mean_robust_pr_auc", "std_robust_pr_auc",
        "mean_fsr", "mean_feasible_flipped_count", "mean_aggregate_feasibility"]
print(summ[cols].to_string(index=False))

# Check 1 — MLP clean PR-AUC anchors (spec §5.3).
ANCHORS = {"CCFD": 0.633, "IEEE-CIS": 0.428, "LCLD": 0.302, "Sparkov": 0.606}
mlp_a = measured[(measured["model"] == "MLP") & (measured["protocol"] == "A_unconstrained")]
print("\\nMLP clean PR-AUC vs anchors (spec §5.3):")
for ds, anchor in ANCHORS.items():
    got = mlp_a[mlp_a["dataset"] == ds]["clean_pr_auc"].mean()
    flag = "OK" if abs(got - anchor) < 0.05 else "DEVIATES — record in experiment_status.md"
    print(f"  {ds:9s} {got:.3f} vs {anchor:.3f}  [{flag}]")

# Check 2 — CCFD negative control: A ≈ B for every model (gap ≈ 1x).
ccfd = summ[summ["dataset"] == "CCFD"].pivot(index="model", columns="protocol",
                                             values="mean_robust_pr_auc")
ccfd["gap_B_over_A"] = ccfd["B_posthoc_filter"] / ccfd["A_unconstrained"]
print("\\nCCFD A vs B (negative control, expect gap ≈ 1x):")
print(ccfd.to_string())'''

C12 = '''# Cell 12: fig_model_family — robust PR-AUC (Square) by model family, A vs B (spec §5.4)
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({"font.family": "serif", "font.size": 9})
PROTO_COLOR = {"A_unconstrained": "#4477aa", "B_posthoc_filter": "#66ccee"}
PROTO_LABEL = {"A_unconstrained": "A", "B_posthoc_filter": "B"}
MODEL_ORDER = ["XGBoost", "MLP", "Ensemble"]

fig, axes = plt.subplots(1, 4, figsize=(11, 3), sharey=True)
for j, (ds, _loader) in enumerate(DATASETS):
    ax = axes[j]
    sub = summ[summ["dataset"] == ds]
    x = np.arange(len(MODEL_ORDER))
    width = 0.36
    for k, proto in enumerate(["A_unconstrained", "B_posthoc_filter"]):
        s = sub[sub["protocol"] == proto].set_index("model").reindex(MODEL_ORDER)
        ax.bar(x + (k - 0.5) * width, s["mean_robust_pr_auc"], width,
               yerr=s["std_robust_pr_auc"].fillna(0), capsize=3,
               color=PROTO_COLOR[proto], label=PROTO_LABEL[proto])
    ax.set_title(ds)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=20)
    if j == 0:
        ax.set_ylabel("robust PR-AUC (Square)")
        ax.legend(fontsize=8, title="protocol")
fig.suptitle("Model-family robustness under Square (ε=0.1, no defence) — CCFD shows A≈B",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG_DIR, f"fig_model_family.{ext}"), dpi=200, bbox_inches="tight")
plt.show()
print("Saved fig_model_family.{pdf,png}")'''

C13 = '''# Cell 13: Session status — what's done, what remains (multi-session runbook)
expected = [make_run_id(reg, m, s, EPS, p)
            for reg, _loader in DATASETS for m in MODELS for s in SEEDS
            for p in ["A_unconstrained", "B_posthoc_filter"]]
done_now = set(pd.read_csv(RESULTS_CSV)["run_id"]) if os.path.exists(RESULTS_CSV) else set()
remaining = [r for r in expected if r not in done_now]
print(f"Measured rows: {len(expected) - len(remaining)}/{len(expected)}")
if remaining:
    print(f"Remaining ({len(remaining)}):")
    for r in remaining:
        print("  ", r)
    print("\\nNext session: run cells 1-5 (restart runtime after 4), then 6-9.")
else:
    print("All 72 measured rows complete. Run cells 10-12 for placeholders, summary, figure.")'''

MD14 = """## Notes

- **Same-model proof:** B is derived from A's saved parquet and evaluated
  against the same fitted model; the A row asserts
  `model_weight_hash == model_hash(trained model)`. Weights persist to
  `MODELS_DIR` on Drive (spec §1.9.1), so a resume after a dead session
  *reloads* the identical model — Cell 9 asserts the stored A hash matches
  the reloaded checkpoint's hash before deriving B.
- **XGBoost config** reuses `configs/*_tree_square.yaml` exactly (max_depth=6,
  n_estimators=100, lr=0.1, no scale_pos_weight) so the fresh layer is
  comparable to the prior Square registry rows; `scale_pos_weight` lives only
  inside the Ensemble's XGB component (`defences/ensemble.py`), which is what
  spec §1.3's parenthetical refers to.
- **Protocol A semantics match NB1:** `attacks/square.py` clips to processed
  schema bounds (ART clip_values + post-attack numeric clip) — the same
  envelope stock CAPGD applies per step — and adds no feasibility constraints.
- **Protocol C placeholder rows** (Cell 10) implement spec §5.2's
  `protocol = N/A`, encoded as `"not_applicable"` — a literal `"N/A"` cell is
  in pandas' default `na_values` and round-trips to `NaN`, silently leaking
  placeholders through `!= "N/A"` filters. NB3 must filter
  `protocol != "not_applicable"` before computing metrics.
- **Expected shape (§5.3):** CCFD is the negative control (A ≈ B, gap ≈ 1×);
  Square on the MLP should degrade PR-AUC *less* than CAPGD did in NB1
  (black-box, weaker attack) — a different attack, not a contradiction."""


def lines(s):
    return s.splitlines(keepends=True)


def code(src, i):
    return {"cell_type": "code", "metadata": {}, "source": lines(src),
            "outputs": [], "execution_count": None, "id": f"nb2-{i:02d}"}


def md(src, i):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(src),
            "id": f"nb2-{i:02d}"}


nb1 = json.load(open("notebooks/icdm_capgd_protocol_grid.ipynb"))
bootstrap = []
for i in range(1, 6):   # cells 1-5 byte-identical to NB1, outputs stripped
    c = nb1["cells"][i]
    bootstrap.append(code("".join(c["source"]), i))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": nb1["metadata"],
    "cells": [md(MD0, 0), *bootstrap, code(C6, 6), code(C7, 7), code(C8, 8),
              code(C9, 9), code(C10, 10), code(C11, 11), code(C12, 12),
              code(C13, 13), md(MD14, 14)],
}
out = "notebooks/icdm_square_model_family.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print(f"Wrote {out} with {len(nb['cells'])} cells")
```

- [ ] **Step 2: Run the builder**

Run from the repo root: `.venv/bin/python /tmp/build_nb2.py`
Expected: `Wrote notebooks/icdm_square_model_family.ipynb with 15 cells`

- [ ] **Step 3: Validate the notebook JSON structure**

```bash
.venv/bin/python - <<'EOF'
import json
nb = json.load(open("notebooks/icdm_square_model_family.ipynb"))
assert len(nb["cells"]) == 15, len(nb["cells"])
nb1 = json.load(open("notebooks/icdm_capgd_protocol_grid.ipynb"))
for i in range(1, 6):   # bootstrap cells byte-identical to NB1
    assert "".join(nb["cells"][i]["source"]) == "".join(nb1["cells"][i]["source"]), i
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    compile(src, f"cell{i}", "exec") if c["cell_type"] == "code" and i not in (2, 3, 4, 5) else None
print("STRUCTURE OK")
EOF
```

Expected: `STRUCTURE OK` (cells 2–5 contain Colab `!`/`google.colab` syntax and are excluded from `compile`; cell 1 is plain Python and must compile).

Note: cell 3 contains `!git pull` lines — only cells 1, 6–13 are compiled. If cell 1 fails to compile, fix the builder string and re-run Step 2.

---

### Task 2: Smoke-test the new logic against the built artifact

**Files:**
- Throwaway: `/tmp/nb2_smoke.py` (not committed)

The smoke test `exec()`s cells 6–8 **out of the built notebook itself** (zero copy-paste drift), then exercises the genuinely new code paths on synthetic data: `model_hash` for all three families (stable, 16-hex, distinct across families), a tiny real `square_attack` run per family, `build_row` schema compliance, and the CCFD `derive_protocol_B` identity.

- [ ] **Step 1: Write `/tmp/nb2_smoke.py`**

```python
"""Smoke-test NB2 cells 6-8 on synthetic data. Run from the repo root:
.venv/bin/python /tmp/nb2_smoke.py
"""
import io
import json
import os
import tempfile
import numpy as np
import pandas as pd

nb = json.load(open("notebooks/icdm_square_model_family.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"]]
g = {}
for i in (6, 7, 8):     # config, helpers, row builder
    exec(cells[i], g)

# The placeholder protocol sentinel must survive a CSV round-trip (pandas
# treats a literal "N/A" cell as NaN — the reason PROTO_NA exists).
rt = pd.read_csv(io.StringIO(f"protocol\n{g['PROTO_NA']}\n"))
assert rt.loc[0, "protocol"] == g["PROTO_NA"], "PROTO_NA is pandas-NA-coerced!"

# build_model returns the right family with the configured params
assert isinstance(g["build_model"]("XGBoost"), g["TreeModel"])
assert isinstance(g["build_model"]("MLP"), g["NeuralModel"])
assert isinstance(g["build_model"]("Ensemble"), g["EnsembleModel"])

rng = np.random.default_rng(0)
n, d = 60, 12
X = pd.DataFrame(rng.normal(size=(n, d)).astype(np.float64),
                 columns=[f"V{i}" for i in range(d)])
y = pd.Series((rng.random(n) < 0.3).astype(int))

small = {
    "MLP": g["NeuralModel"]({"epochs": 2, "hidden_dim": 8, "batch_size": 32, "lr": 0.01}),
    "XGBoost": g["TreeModel"]({"n_estimators": 5, "max_depth": 2}),
    "Ensemble": g["EnsembleModel"]({"epochs": 2, "hidden_dim": 8, "batch_size": 32, "lr": 0.01}),
}
proc_ft = {c: "numeric" for c in X.columns}
schema = g["ConstraintSchema"].from_data(X, proc_ft)

tmpdir = tempfile.mkdtemp()
hashes = {}
for name, m in small.items():
    g["set_all_seeds"](0)
    m.fit(X, y)
    h1, h2 = g["model_hash"](m), g["model_hash"](m)
    assert h1 == h2 and len(h1) == 16, f"{name}: hash not a stable 16-hex"
    hashes[name] = h1

    # Checkpoint round-trip: the resume path reloads from Drive and asserts
    # the hash is unchanged — verify save -> load -> model_hash is identity.
    ckpt = os.path.join(tmpdir, f"{name}{'.pt' if name == 'MLP' else '.joblib'}")
    m.save(ckpt)
    m2 = type(m).load(ckpt)
    assert g["model_hash"](m2) == h1, f"{name}: save/load round-trip changed the hash"

    adv = g["square_attack"](m, X, y, schema, proc_ft,
                             params={"epsilon": 0.1, "max_iter": 5, "norm": "inf"})
    assert adv.shape == X.shape and list(adv.columns) == list(X.columns)
    assert adv.index.equals(X.index)

    # Parquet round-trip (Cell 9 re-reads the saved file before deriving B).
    pq = os.path.join(tmpdir, f"{name}.parquet")
    adv.to_parquet(pq)
    adv = pd.read_parquet(pq)
    pd.testing.assert_frame_equal(adv, pd.read_parquet(pq))

    grp = dict(model=m, pre=None, X_test_p=X, y_test=y,
               clean_probs=m.predict_proba(X), hash=h1)
    row, feas = g["build_row"](grp, "CCFD", name, 0, "A_unconstrained", adv, 1.0)
    assert set(row) == set(g["RESULTS_COLUMNS"]), f"{name}: row keys != §3.1 schema"
    assert row["model_weight_hash"] == h1
    assert row["aggregate_feasibility"] == 1.0      # CCFD: no constraints
    assert row["main_failed_constraint"] == "none"
    B = g["derive_protocol_B"](adv, feas.feasible_row_mask, X)
    assert B.equals(adv)                            # CCFD: B == A
    print(f"  {name}: hash={h1}  robPR={row['robust_pr_auc']:.3f}  OK")

assert len(set(hashes.values())) == 3, "hashes must differ across families"

# derive_protocol_B with a partially-False mask: infeasible rows revert to
# clean, feasible rows keep the adversarial values (not exercisable via CCFD).
fake_adv = X + 1.0
mask = pd.Series([i % 2 == 0 for i in range(len(X))], index=X.index)
Bm = g["derive_protocol_B"](fake_adv, mask, X)
assert Bm[mask].equals(fake_adv[mask]), "feasible rows must keep adversarial values"
assert Bm[~mask].equals(X[~mask]), "infeasible rows must revert to clean"
print("SMOKE OK")
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python /tmp/nb2_smoke.py`
Expected output ends with `SMOKE OK` after three `... OK` lines (one per family). The Ensemble prints its internal `Training Ensemble (LR + XGBoost + MLP)...` progress — that's normal.

- [ ] **Step 3: Fix any failures by editing the builder strings, not the .ipynb**

If an assert fires (e.g., `save_raw(raw_format="ubj")` signature differs in the installed xgboost, or `compute_metrics` lacks a key), fix the corresponding cell string in `/tmp/build_nb2.py`, re-run Task 1 Steps 2–3, then re-run this smoke test. The notebook JSON is generated, never hand-edited. Specific contingency: if the XGBoost save/load hash round-trip fails (booster re-serialisation not byte-identical in the installed version), switch `model_hash` for trees to hash the booster's `save_raw` captured *once at fit time* and stored on the instance — or fall back to comparing `predict_proba` outputs in the resume assert — and document the change in the plan.

---

### Task 3: Commit the notebook

- [ ] **Step 1: Confirm only the notebook changed**

Run: `git status --porcelain`
Expected: `?? notebooks/icdm_square_model_family.ipynb` (untracked; possibly empty untracked `results/` dirs created by the smoke test's Cell 6 exec — do not commit those).

- [ ] **Step 2: Commit**

```bash
git add notebooks/icdm_square_model_family.ipynb
git commit -m "feat(notebooks): NB2 ICDM Square model-family (A/B + C-N/A placeholders, Drive-direct resume)

36 Square runs (4 datasets x MLP/XGBoost/Ensemble x 3 seeds, eps=0.1, defence
none), Protocol B derived from A's saved parquet, 24 protocol=N/A placeholder
rows for the deferred Cartella-style tree/ensemble Protocol C (spec §5.2).
Outputs write directly to Drive for multi-session resume; model_hash covers
all three families (state_dict / booster UBJ / LR+booster+MLP concat).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Colab execution runbook (manual, ~5–6 h GPU across sessions)

This task is run by Xitong on Colab (A100/T4), not locally. The notebook is multi-session safe.

- [ ] **Step 1: First session** — push master, open the notebook in Colab, run cells 1–4, restart the runtime (Runtime → Restart session), run cells 5–9. Cell 9 prints one block per (dataset, model, seed) with per-run timing. CCFD groups finish in ~1–2 min; IEEE-CIS/LCLD/Sparkov take ~7–15 min per Square run.

- [ ] **Step 2: Subsequent sessions (if the 6 h doesn't fit one session)** — rerun cells 1–5 (restart after 4), then 6 and 9 (cells 7–8 are definitions, also required: run 6–9 in order). Cell 9 resumes from the Drive CSV; previously completed run_ids are skipped, and half-done groups reload their Drive checkpoint instead of retraining. Run cell 13 anytime to see what remains. **Never run two NB2 sessions concurrently** — the `done` set is loaded once per Cell 9 run and the Drive CSV append is not concurrency-safe; overlapping sessions would duplicate run_ids.

- [ ] **Step 3: After all 72 measured rows complete** — run cells 10 (N/A placeholders), 11 (summary + §5.3 checks), 12 (figure), 13 (status shows complete).

- [ ] **Step 4: Verification checklist (on Colab, before leaving the session)**

```python
import pandas as pd
res = pd.read_csv(RESULTS_CSV)

# Exact expected run_id sets (not just counts).
measured_ids = {make_run_id(reg, m, s, EPS, p)
                for reg, _l in DATASETS for m in MODELS for s in SEEDS
                for p in ["A_unconstrained", "B_posthoc_filter"]}
na_ids = {make_run_id(reg, m, s, EPS, PROTO_NA)
          for reg, _l in DATASETS for m in ["XGBoost", "Ensemble"] for s in SEEDS}
assert res["run_id"].is_unique, "duplicate run_ids — were two sessions run concurrently?"
assert set(res["run_id"]) == measured_ids | na_ids, {
    "missing": (measured_ids | na_ids) - set(res["run_id"]),
    "unexpected": set(res["run_id"]) - (measured_ids | na_ids)}
assert (res["protocol"] == PROTO_NA).sum() == 24
assert (res.groupby("same_model_group_id")["model_weight_hash"].nunique() == 1).all()

# Exact expected parquet set (A + B per measured run).
expected_parq = {rid + ".parquet" for rid in measured_ids}
actual_parq = {f for f in os.listdir(ADV_DIR) if f.endswith(".parquet")}
assert actual_parq == expected_parq, {
    "missing": expected_parq - actual_parq, "unexpected": actual_parq - expected_parq}
print("VERIFY OK")
```

Expected: `VERIFY OK`. Also confirm Cell 11 printed `[OK]` for all four MLP clean PR-AUC anchors and CCFD `gap_B_over_A ≈ 1`. The hash-uniqueness assert must hold strictly — checkpoints persist to Drive, so resumes reload rather than retrain; a failure means a checkpoint was deleted mid-grid and that group should be re-run from scratch.

- [ ] **Step 5: Commit the executed notebook** (mirrors NB1's `c82f02a` convention — outputs in the notebook, CSVs live on Drive)

```bash
git add notebooks/icdm_square_model_family.ipynb
git commit -m "run: NB2 Square model-family Colab outputs (<summarise: anchors OK/deviations>)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

Record in the commit body: MLP clean PR-AUC anchor deltas, the CCFD A≈B gap, and any resume-retrain hash warnings.

---

## Self-review notes

- **Spec coverage:** §5.1 goal → Tasks 1+4; §5.2 matrix (36 runs, A+B, no C, N/A rows) → Cells 9+10; §5.3 expected-shape checks → Cell 11; §5.4 figure → Cell 12; §1.9 reproducibility (hash, parquet, resumable CSV append) → Cells 7–9; §3.1/§3.3 schemas → Cells 6/8/11; §3.4 layout (`square_family_results.csv`/`_summary.csv` only, no per-constraint file) → Cell 6 paths.
- **Known deviations from NB1's layout, intentional:** (a) outputs Drive-direct instead of local+backup (multi-session resume, spec §8); Cell 13 is a status cell instead of a backup cell. (b) Model weights persist to `MODELS_DIR` (spec §1.9.1 — NB1 skipped this; for a 6 h multi-session grid, checkpoint reload makes the same-model hash exact across sessions instead of relying on retrain determinism).
- **NB3 dependency created here:** NB3 must filter `protocol != "not_applicable"` and read NB2's CSVs from the Drive `icdm_2026` dir alongside NB1's backups.
- **External review (Codex gpt-5.5, 2026-06-04):** 7 findings, all addressed — (1) blocker: literal `"N/A"` protocol cells are pandas-NA-coerced on read-back → replaced with `PROTO_NA = "not_applicable"` + smoke-test guard; (2) resume branch now asserts the A parquet exists with recovery instructions; (3) runbook forbids concurrent sessions; (4) fresh path re-reads A from the saved parquet per spec §1.9.3; (5) smoke test gained save/load hash round-trips, a false-mask `derive_protocol_B` case, and parquet round-trip; (6) verification compares exact run_id/parquet sets, not counts; (7) §3.1-enum deviation documented for NB3.
