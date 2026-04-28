# FraudBench: Supplementary MVP To-Do List

> **Created:** 2026-02-20
> **Last updated:** 2026-04-28
> **Status:** All P1 items below are complete. Project focus has since shifted to the constraint-aware evaluation arc for ICAIF 2026 — see `docs/Context.md` §9 and `docs/constraint_evaluation_guidance.md`. This doc is retained as a record of the supervisor-meeting MVP scope.
> **Purpose:** Close remaining gaps before supervisor meeting to present a complete MVP
> **Estimated total effort:** ~16 hours (retrospective; all items completed Feb–Mar 2026)

---

## Context

The core experimental matrix (CAPGD × 4 datasets × 2 models × 3 seeds + epsilon sweeps) is complete. Square Attack is fully covered. HopSkipJump is partially covered. The benchmark has 150 clean experiment rows in `registry_clean.csv`.

**However, the thesis title promises a "Comparative Study of Defence Mechanisms", and the current MVP only has 2 defence methods — one of which is proven harmful.** This is the primary gap that must be closed before the MVP can support a productive supervisor discussion.

The items below are ordered by impact on MVP completeness, not by ease of implementation.

---

## 1. Implement Ensemble Defence [P1 — NEW] — ✅ DONE

**Status (2026-04-28):** Implemented in `defences/ensemble.py` + `models/ensemble.py`. 24 experiment runs in `results/registry_clean.csv`. Configs: `*_ensemble.yaml` and `*_ensemble_square.yaml`. Defence comparison table is now 3-way as planned.

---


**Why:** The thesis title centres on comparing defence mechanisms. Currently only Adversarial Training is effective; Input Validation is actively harmful. Adding Ensemble creates a meaningful 3-way comparison and addresses the "tree model defence vacuum" (XGBoost cannot use Adversarial Training).

**What:**
- [ ] Implement `defences/ensemble.py` — heterogeneous ensemble (e.g. Logistic Regression + XGBoost + Neural MLP with majority voting or stacking)
- [ ] Create config files: `configs/{dataset}_ensemble.yaml` for all 4 datasets
- [ ] Run experiments: 4 datasets × 3 seeds = **12 experiments**
- [ ] Evaluate under CAPGD attack (Neural-based ensemble) and Square Attack (tree-compatible)

**Outcome:** Defence comparison table expands from 2 → 3 methods. Ensemble is the only defence applicable to both model families, which is itself a finding.

**Estimated effort:** 6 hours (implementation + experiments)

---

## 2. Deepen Input Validation Analysis [P1 — UPGRADED from P0 #2] — ✅ DONE (analysis); writeup pending

**Status (2026-04-28):** `scripts/analyse_input_validation.py` produces `results/figures/input_validation_analysis.csv` and `.png`. The mechanistic finding (CAPGD respects bounds → bound clip is no-op → z-score clip destroys signal) is documented numerically. Thesis Discussion writeup remains pending.

---


**Why:** Input Validation is not just "ineffective" — it is actively harmful. On Sparkov, it drops tree clean PR-AUC from 0.747 to 0.232 even without any attack. This is a strong, publishable negative finding, but it needs evidence beyond raw numbers.

**What:**
- [ ] Run supplementary experiments with `z_threshold` = {3.0 (current), 5.0, 10.0, None} on CCFD and Sparkov (Neural + Tree), seeds 42 only → **~8 quick runs**
- [ ] Quantify: fraction of features clipped per sample, feature distribution shift pre/post clipping
- [ ] Draft 1–2 paragraphs explaining the mechanism: constrained attacks already respect feature bounds → bound clipping is a no-op → z-score clipping destroys discriminative signal

**Outcome:** Transforms a raw negative result into an explained, defensible finding for Discussion section.

**Estimated effort:** 3 hours (experiments + analysis + writing)

---

## 3. Adversarial Training Trade-off Analysis [P1 — NEW] — ✅ DONE

**Status (2026-04-28):** Output in `results/figures/adv_training_tradeoffs.csv`. Cross-dataset effectiveness variance (CCFD -7.2% vs Sparkov -64.6%) tabulated; tree-model architectural N/A documented as a model-family-dependent finding.

---


**Why:** Adversarial Training effectiveness varies dramatically across datasets (CCFD: -7.2% drop vs Sparkov: -64.6% drop). Understanding why is essential for the "comparative study" narrative.

**What:**
- [ ] Tabulate clean accuracy cost: how much does Adversarial Training reduce clean PR-AUC on each dataset?
- [ ] Correlate effectiveness with dataset characteristics (fraud rate, feature count, sample size, class separability)
- [ ] Document the tree model limitation: Adversarial Training is architecturally inapplicable to XGBoost — this is a model-family-dependent constraint, not a bug

**Outcome:** Adds analytical depth to the core defence comparison. Demonstrates that "best defence" is dataset- and model-dependent.

**Estimated effort:** 2 hours (analysis + writing, no new experiments needed — data already in registry)

---

## 4. Statistical Significance Tests [P1 — UPGRADED from P2 #9] — ✅ DONE

**Status (2026-04-28):** `scripts/statistical_tests.py` runs paired t-tests + Wilcoxon signed-rank + Cohen's d. Cross-model ensemble comparisons added Feb 21. Results in `results/figures/statistical_tests.csv`.

---


**Why:** With 3 seeds per configuration and 3 defence methods (after Ensemble), statistical tests give the comparison rigour expected of a benchmark paper.

**What:**
- [ ] Run `scripts/statistical_tests.py` (or implement if not yet functional)
- [ ] Wilcoxon signed-rank test: Adv Training vs No Defence, Input Validation vs No Defence, Ensemble vs No Defence
- [ ] Report p-values in results table
- [ ] Consider effect size (Cohen's d or similar) for practical significance

**Outcome:** Validates that observed defence differences are statistically significant, not seed noise.

**Estimated effort:** 2 hours

---

## 5. Generate Figures + Reproducibility Docs [P1 — EXISTING #5 + #7] — ✅ DONE (figures); reproducibility docs partial

**Status (2026-04-28):** `scripts/generate_figures.py` produces 6 figure types (robustness bars/curves, attack comparison, defence heatmap, training time, summary table) — all present in `results/figures/`. Reproducibility docs (dataset download URLs, README dataset sections, YAML field documentation) remain partial — see legacy `ToDo.md` §5.

---


**Why:** A benchmark is not just experiments — it requires reproducible documentation and clear visualisation. These are MVP deliverables, not enhancements.

**What:**
- [ ] Run `scripts/generate_figures.py` after all experiments complete
- [ ] Verify 5 figure types: robustness bars, attack comparison, summary table, defence heatmap, training time
- [ ] Write dataset download instructions (exact URLs, directory structure, expected file sizes)
- [ ] Update README with LCLD, Sparkov, HopSkipJump, Square Attack sections
- [ ] Document all YAML config fields

**Outcome:** Supervisor can see polished visualisations; any researcher can clone and reproduce.

**Estimated effort:** 3 hours

---

## Deprioritised / Deferred to Post-MVP

These items remain on the backlog but should **not** be pursued before the supervisor meeting. They are discussion points, not action items.

| Item | Original Priority | Decision | Rationale |
|------|------------------|----------|-----------|
| HopSkipJump remaining 6 runs | P1 | **Defer** | ~36-60h CPU time; Square Attack already provides full black-box coverage across all 4 datasets × 3 seeds. CCFD has complete HSJ 3-seed data. Diminishing returns. |
| SignOPT attack | New proposal | **Defer** | Adds a 4th attack method but does not address the defence gap. Attack dimension is already well-covered. |
| CTGAN augmentation | P2 | **Defer** | 4th defence method; discuss with supervisor whether to pursue after Ensemble results are available. |
| Transferability experiments | P2 | **Defer** | Interesting but not core to the defence comparison narrative. |
| Model zoo | P2 | **Defer** | Post-benchmark enhancement for community use. |

---

## Suggested Execution Order

| Day | Task | Hours | Cumulative Output |
|-----|------|-------|-------------------|
| 1 | Ensemble defence: implement + run 12 experiments | ~6h | 3-way defence comparison data |
| 2 | Input Validation analysis (z_threshold experiments + writeup) | ~3h | Explained negative finding |
| 2 | Adv Training trade-off analysis (no new experiments) | ~2h | Cross-dataset defence analysis |
| 3 | Statistical tests | ~2h | p-values for all defence comparisons |
| 3 | Figures + reproducibility docs | ~3h | Polished MVP deliverables |

---

## MVP Completeness After This List

| Thesis Title Component | Before | After |
|------------------------|--------|-------|
| Benchmarking | ✅ Multi-seed, config-driven, CI/CD | ✅ + figures, docs, reproducibility |
| Adversarial Robustness | ✅ 3 attack types | ✅ (unchanged) |
| Fraud Detection Models | ✅ Neural + XGBoost | ✅ (unchanged) |
| **Comparative Study of Defence Mechanisms** | ⚠️ 2 methods (1 effective, 1 harmful) | ✅ **3 methods + statistical tests + depth analysis** |
| Constrained Adaptive Attacks | ✅ ConstraintSchema + projection | ✅ (unchanged) |
| Heterogeneous Financial Datasets | ✅ 4 datasets | ✅ (unchanged) |

---

## Questions for Supervisor Discussion

After completing the MVP supplement, bring these to the supervisor meeting:

1. **Scope of defence comparison:** Is 3 methods (Adv Training + Input Validation + Ensemble) sufficient, or should CTGAN be pursued as a 4th?
2. **HopSkipJump completion:** Is partial HSJ coverage acceptable given Square Attack's full coverage, or is completing all 12 HSJ runs worth the ~40h compute investment?
3. **Input Validation finding:** Should the "actively harmful" result be framed as a standalone contribution, or as part of the broader defence comparison?
4. **Benchmark packaging:** What level of documentation/tooling is expected — research code with README, or installable Python package with CLI?
5. **Conference target:** Does the scope support a workshop paper submission, or should additional experiments be pursued for a full conference paper?
