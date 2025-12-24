# MVP Plan / To-Do List (FRBS – Minimum Viable Benchmark Suite)

## MVP Goal (1 sentence)

Deliver an **end-to-end reproducible benchmark** that runs on **at least 1–2 datasets**, supports **CAPGD with MVP constraints**, evaluates **2 model families + 2 defences**, reports **fraud-first metrics + cost**, and outputs a **structured results registry** that can be extended later.

---

## 0. MVP Scope Lock (non-negotiable)

### Must include (MVP)

* **Datasets:** 1–2 (recommended: CCFD + IEEE-CIS)
* **Models:** 2 (one tree-based + one neural tabular)
* **Attacks:** CAPGD (constrained white-box) **required**
* **Defences:** at least 2 (recommended: adversarial training + input validation)
* **Metrics:** PR-AUC, fraud precision/recall/F1 + accuracy (secondary) + cost logging
* **Reproducibility:** config-driven runs + fixed splits/seeds + saved outputs

### Not required in MVP (post-MVP)

* HopSkipJump / Square (black-box)
* CTGAN augmentation (can be Phase 2)
* Full 4-dataset coverage
* Transferability matrix (optional after MVP)

---

## 1. Repository & Experiment Runner (Foundation)

### 1.1 Project skeleton

* [ ] Create repo structure:

  * `datasets/`, `models/`, `attacks/`, `defences/`, `constraints/`, `runner/`, `reports/`, `configs/`, `results/`
* [ ] Add environment spec (requirements/lockfile) and a single entrypoint command.

**Definition of Done (DoD):**

* A new user can install deps and run `python -m runner.run --config configs/mvp.yaml`.

### 1.2 Config-driven execution

* [ ] Implement config schema (YAML/JSON) containing:

  * dataset id, split strategy, preprocessing params
  * model choice + hyperparams
  * defence choice + params
  * attack choice + budgets
  * seeds + output paths

**DoD:**

* Changing only the config reproduces the run without code edits.

---

## 2. Dataset Layer (1–2 datasets end-to-end)

### 2.1 Dataset loaders

* [ ] Implement `load_dataset(dataset_name)` for MVP datasets.
* [ ] Output a unified object:

  * `X`, `y`, `feature_types`, `feature_names`, `meta`

**DoD:**

* `X` is numeric matrix ready for preprocessing; categorical fields are flagged in `feature_types`.

### 2.2 Splits (leakage-safe)

* [ ] Implement split manager:

  * stratified train/val/test (MVP)
  * (optional later) time-based split if dataset supports it

**DoD:**

* Split indices are saved and reused across runs.

### 2.3 Dataset cards (minimal)

* [ ] Create a short dataset card per MVP dataset:

  * label meaning, imbalance rate, feature types, known leakage risks, chosen split strategy

**DoD:**

* Cards exist as markdown files under `datasets/cards/`.

---

## 3. Preprocessing Layer (consistent + reproducible)

### 3.1 Leakage-safe preprocessing

* [ ] Fit preprocessing on train only; apply to val/test.
* [ ] Support:

  * scaling numeric features
  * encoding categorical (for MVP: one-hot or ordinal, but **must be consistent**)
* [ ] Save preprocessing artifacts.

**DoD:**

* Re-running uses the same fitted preprocessing when split + seed match.

---

## 4. Model Layer (2 baselines)

### 4.1 Tree-based model wrapper

* [ ] Implement baseline tree model wrapper (e.g., XGBoost/LightGBM or sklearn equivalent).
* [ ] Standard methods: `fit()`, `predict_proba()`, `evaluate()`.

### 4.2 Neural tabular model wrapper

* [ ] Implement MLP (simple) wrapper:

  * early stopping on validation PR-AUC (recommended)
* [ ] Ensure deterministic seeds (as much as feasible).

**DoD (Models):**

* Both models train and achieve reasonable clean PR-AUC on MVP dataset(s), and expose a common prediction interface.

---

## 5. Constraint Layer (MVP Constraints + Validator) — core for CAPGD

### 5.1 Constraint schema

* [ ] Define per-dataset constraint schema file (JSON/YAML):

  * feature type map (continuous/categorical/binary)
  * per-feature min/max bounds (from train statistics or known bounds)
  * non-negativity flags
  * one-hot groups (if using one-hot)
  * immutable feature list (optional, if clear)

**DoD:**

* A schema exists for each MVP dataset and is loaded by the attack.

### 5.2 Constraint validator

* [ ] Implement `validate(x_adv, schema)`:

  * range check
  * non-negativity check
  * one-hot validity check (sum==1 and values in {0,1} with tolerance)
* [ ] Report validity rate.

**DoD:**

* Every CAPGD run outputs “validity rate” and rejects/flags invalid adversarial samples.

---

## 6. Attack Harness (CAPGD MVP)

### 6.1 CAPGD integration

* [ ] Implement or integrate CAPGD with:

  * constraint-aware projection step
  * budget controls (eps/steps/step-size)
* [ ] Standard interface:

  * `generate(model, X, y, constraints, attack_params) -> X_adv`

**DoD:**

* For at least one model + dataset, CAPGD produces adversarial samples that:

  * reduce fraud PR-AUC / recall
  * satisfy constraints at high validity rate

### 6.2 Attack logging

* [ ] Log per-run:

  * attack parameters
  * runtime per batch
  * success rate (as you define it consistently)
  * validity rate

**DoD:**

* Attack logs saved into `results/registry.csv` or JSON.

---

## 7. Defence Harness (2 defences for MVP)

### 7.1 Adversarial training (CAPGD-based)

* [ ] Implement adversarial training wrapper:

  * generate adversarial examples on training (or mini-batch)
  * train model with clean+adv mix
* [ ] Keep budgets small for MVP.

**DoD:**

* Adversarially trained model shows improved robust fraud metrics vs undefended baseline.

### 7.2 Input validation / sanitisation

* [ ] Implement a simple, explicit defence:

  * clipping to bounds
  * fixing one-hot validity (argmax)
  * optional outlier rejection rule (simple threshold)

**DoD:**

* Defence runs as a pre-processing step before inference and produces measurable changes under attack.

---

## 8. Evaluation & Reporting (Fraud-first + cost)

### 8.1 Metrics module (fraud-first)

* [ ] Compute:

  * PR-AUC
  * fraud precision/recall/F1
  * accuracy (secondary)
* [ ] Compute for:

  * clean test set
  * adversarial test set (CAPGD)

**DoD:**

* One results table per run: clean vs robust metrics clearly separated.

### 8.2 Cost logging

* [ ] Record:

  * training time
  * attack generation time
  * (optional) GPU usage if easy, otherwise omit

**DoD:**

* Results registry includes cost columns.

### 8.3 Results registry

* [ ] Save all outcomes as structured rows:

  * dataset, split id, model, defence, attack, params hash, metrics, cost, validity rate

**DoD:**

* You can filter/aggregate results without rerunning experiments.

---

## 9. MVP Validation Runs (minimum experiment matrix)

Run these 6–8 experiments to prove MVP works:

1. Dataset A, Model 1, **No defence**, Clean + CAPGD
2. Dataset A, Model 1, **Input validation**, Clean + CAPGD
3. Dataset A, Model 1, **Adversarial training**, Clean + CAPGD
4. Dataset A, Model 2, **No defence**, Clean + CAPGD
5. Dataset A, Model 2, **Input validation**, Clean + CAPGD
6. Dataset A, Model 2, **Adversarial training**, Clean + CAPGD
   (+ repeat on Dataset B if time permits)

**MVP Success Criteria:**

* CAPGD decreases fraud PR-AUC/recall vs clean baseline
* At least one defence improves robust fraud metrics (relative gain) with logged cost
* Constraint validity rate is reported and acceptable (you define a threshold and justify it)
* Results registry is complete and reproducible from config

---

## 10. Post-MVP Backlog (only after MVP passes)

* Add remaining datasets (LCLD, Sparkov)
* Add CTGAN augmentation defence
* Add black-box attacks (HSJ, Square) + query budgeting
* Add transferability matrix + rule-triggered defence combinations
* Add time-based splits where appropriate
* Add benchmarking report auto-generation (tables/plots)

