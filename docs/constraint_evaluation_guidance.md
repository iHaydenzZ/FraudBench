# FraudBench Constraint-Aware Evaluation: Strategic Guidance

**Date:** 2026-04-15  
**Status:** Internal guidance document  
**Target venue:** ICAIF 2026 (~July deadline, 8-page ACM format)  
**Prerequisites:** TabularBench comparison findings, Mask ablation findings (both verified)

---

## 1. Executive Summary

FraudBench's next experimental phase centres on **constraint-aware evaluation** — the idea that adversarial robustness on financial tabular data must account for domain constraints, attacker capabilities, and feature semantics. Our completed experiments establish three foundational results:

1. **Constraint gap:** ~99.8% of CAPGD adversarial examples on LCLD are domain-infeasible (aggregate feasibility 0.14–0.22%), driven almost entirely by the g1 installment formula (~98% failure rate). This produces the largest ADV/ADV+CTR gap (+55 pp) on the LCLD leaderboard.

2. **Capability spectrum:** Mask ablation (M0→M6-strict) yields a monotone robust accuracy gradient from 0.042 to 0.340 (+29.8 pp), confirming that realistic attacker-capability modelling materially changes robustness assessment. However, robust PR-AUC remains locked at 0.1051 ± 0.0001 across all eight variants — capability constraints improve accuracy but not the precision-recall trade-off under attack.

3. **Metric sensitivity:** Accuracy-based ranking on imbalanced fraud data produces substantially different model orderings than F1/MCC (Kendall's τ < 0.74), with 10/70 degenerate TabNet models unpenalised by accuracy.

The central challenge for the next phase: **only LCLD has strong mathematical constraints (g1)**. CCFD features are PCA-anonymised; IEEE-CIS V-features are opaque; Sparkov is synthetic with limited documentation. This is not a weakness — it is the empirical basis for a **constraint richness gradient** that itself constitutes a benchmark contribution.

---

## 2. Evaluation Framework: Three Tiers

We propose a tiered evaluation framework where each dataset is assessed at whichever tiers its constraint structure supports. This mirrors TabularBench's own practice: its five datasets range from 9 nonlinear constraints (LCLD) to simple linear boundary constraints (URL, WiDS).

### Tier A — Capability-Aware (all 4 datasets)

**What it evaluates:** "Which features can a realistic attacker modify?"

Requires only a mutable/immutable feature partition and basic range validity. No domain formulas needed. The M6-strict experiment on LCLD already demonstrates this tier: freezing features to a realistic 5-feature attacker profile lifts robust accuracy by +18 pp vs the full-mask baseline.

**Applicable constraints:** Mutable/immutable partition, min/max feature ranges, OHE validity for categoricals, cost-weighted perturbation budget.

### Tier B — Structure-Aware (LCLD, IEEE-CIS, Sparkov)

**What it evaluates:** "Do adversarial perturbations preserve basic domain structure?"

Requires interpretable feature semantics — at minimum, knowing that some features are counts (non-negative, integer), some are categorical (must stay valid), and some have monotonic or inequality relationships. Does not require closed-form formulas.

**Applicable constraints:** Non-negativity, integer-valued counts, simple inequalities between fields, categorical validity beyond OHE encoding, temporal monotonicity.

### Tier C — Formula-Aware (LCLD only, for now)

**What it evaluates:** "Do adversarial perturbations satisfy explicit financial equations?"

Requires domain knowledge sufficient to write mathematical constraint functions. Currently only LCLD qualifies, with its installment amortisation formula (g1) and auxiliary constraints (g2–g6).

**Applicable constraints:** Closed-form coupling equations, derived-ratio consistency, nonlinear inter-feature relationships.

---

## 3. Per-Dataset Constraint Inventory

### 3.1 LCLD (Lending Club Loan Default)

| Dimension | Details |
|-----------|---------|
| **Total features** | 63 raw → 188 processed (after OHE) |
| **Interpretable features** | All 63 — loan terms, borrower profile, credit history, application metadata |
| **Mutable features** | 11 raw: loan_amnt, term, purpose, emp_length, annual_inc, annual_inc_joint, home_ownership, dti, dti_joint, application_type, addr_state |
| **Immutable features** | 41 raw: grade, sub_grade, int_rate, installment, funded_amnt, credit bureau fields (delinq_2yrs, open_acc, pub_rec, total_acc, revol_bal, etc.), verification_status, payment_inc_ratio |

**Hard constraints (verified, implemented):**

| ID | Constraint | Type | Clean pass | Adv pass | Binding? |
|----|-----------|------|-----------|---------|----------|
| g1 | installment = f(loan_amnt, int_rate, term) | Nonlinear formula | 99.8% | 1.0–2.0% | **Dominant bottleneck** |
| g2 | open_acc ≤ total_acc | Inequality | 100% | 98.4% | Weak (masked fixes to 100%) |
| g3 | pub_rec_bankruptcies ≤ pub_rec | Inequality | 99.9% | 77.9% | Moderate (masked fixes to 100%) |
| g4 | term ∈ {36, 60} (OHE valid) | Discrete validity | 100% | 19.3% | Strong (g4 + g1 together → ~0.1% feasibility) |
| g5 | ratio_loan/inc consistency | Derived ratio | 99.98% | trivial | Vacuous (recomputed from base) |
| g6 | ratio_open/total consistency | Derived ratio | 100% | trivial | Vacuous (recomputed from base) |

**Soft filters (implementable, not yet implemented):**

| Filter | Description | Evidence | Priority |
|--------|------------|----------|----------|
| dti consistency | dti should approximate total_debt / annual_inc | Domain knowledge from LC documentation | Medium — already frozen in M3 variant |
| grade ↔ int_rate monotonicity | Higher grade (worse credit) → higher rate | LC pricing model | Low — both immutable |
| emp_length increase-only | Employment length cannot decrease over time | Common sense | Low — M2 showed no effect (OHE argmax never flips at ε=0.1) |

**Evidence strength:** Very strong. All hard constraints verified against clean data (99.8–100% pass rates). g1 is mathematically derived from standard loan amortisation. Constraints are documented in TabularBench (Simonetto et al., NeurIPS 2024).

**Implementation priority:** Highest — already implemented and verified. Next step is constraint-aware attack integration (penalty-based projection for g1).

---

### 3.2 CCFD (Credit Card Fraud Detection)

| Dimension | Details |
|-----------|---------|
| **Total features** | 30 (Time, V1–V28, Amount) |
| **Interpretable features** | 2 only: Time, Amount |
| **Mutable features** | Amount (attacker chooses transaction amount); Time is system-generated → immutable |
| **Immutable features** | V1–V28 (PCA-transformed, semantics unknown), Time |

**Hard constraints:**

| ID | Constraint | Type | Evidence | Binding? |
|----|-----------|------|----------|----------|
| c1 | Amount ≥ 0 | Range | Trivial | Weak |
| c2 | Time ≥ 0, monotonically increasing | Range + monotonicity | System-generated | Weak (immutable anyway) |

**Soft filters (candidate, not yet implemented):**

| Filter | Description | Evidence | Priority |
|--------|------------|----------|----------|
| PCA manifold plausibility | Adversarial V-feature vectors should fall within the learned PCA subspace distribution | Statistical — can verify via density estimation or reconstruction error on training data | **High — this is the only meaningful filter for CCFD** |
| Amount range per transaction context | Unusually large amounts may be implausible | Empirical distribution from training data | Medium |
| V-feature correlation structure | PCA components have specific inter-correlation patterns that perturbation may break | Computable from training covariance matrix | Medium |

**Evidence strength:** Weak for hard constraints (trivial bounds only). The PCA manifold filter is statistically motivated but not domain-verified — we cannot confirm what the original features were, so we cannot write business rules. This is an inherent limitation of the dataset's anonymisation.

**Implementation priority:** Low for hard constraints (almost nothing to check). Medium for statistical plausibility filters — these would be a novel contribution but require careful framing as "statistical" rather than "domain" constraints.

**Key caveat for paper:** CCFD is the **no-constraint baseline** in the gradient. If unconstrained CAPGD adversarial examples on CCFD cannot be filtered by any domain check, this demonstrates that Tier A (capability-aware) evaluation is the only meaningful layer here — and that constraint-aware methods are not universally necessary, but dataset-dependent. This is a feature, not a bug.

---

### 3.3 IEEE-CIS (Vesta E-Commerce Fraud Detection)

| Dimension | Details |
|-----------|---------|
| **Total features** | 392 transaction-only (434 with identity) |
| **Interpretable features** | ~53: TransactionAmt, ProductCD, card1–6, addr1–2, dist1–2, P/R_emaildomain, C1–C14, D1–D15, M1–M9 |
| **Opaque features** | 339: V1–V339 (Vesta-engineered, semantics undocumented) |
| **Mutable features (candidate)** | TransactionAmt (attacker controls), ProductCD (attacker chooses product), possibly addr/dist fields |
| **Immutable features (candidate)** | card1–6 (issuer-determined), V1–V339 (internal aggregates), D1–D15 (timedelta from internal events), C1–C14 (counting features — likely system-aggregated) |

**Hard constraints:**

| ID | Constraint | Type | Evidence | Binding? |
|----|-----------|------|----------|----------|
| i1 | ProductCD ∈ {W, H, C, S, R} | Categorical validity | Dataset documentation | Weak (OHE handles this) |
| i2 | card4 ∈ {visa, mastercard, discover, amex} | Categorical validity | Dataset documentation | Weak |
| i3 | card6 ∈ {debit, credit, charge, debit or credit} | Categorical validity | Dataset documentation | Weak |
| i4 | M1–M9 ∈ {T, F} (match flags) | Categorical validity | Dataset documentation | Weak |
| i5 | C1–C14 ≥ 0 (counting features) | Non-negativity | Semantic — "counting" implies integer ≥ 0 | Moderate |
| i6 | TransactionAmt > 0 | Range | Business logic | Weak |

**Soft filters (candidate, not yet implemented):**

| Filter | Description | Evidence | Priority |
|--------|------------|----------|----------|
| C-feature integrality | C1–C14 should be integer-valued | Naming convention ("counting") | Medium |
| D-feature non-negativity | D1–D15 are timedeltas, should be ≥ 0 | Naming convention | Medium |
| Email domain validity | P_emaildomain and R_emaildomain should be real domain strings | Lookup table from training data | Low (already categorical in pipeline) |
| V-feature manifold | Same statistical plausibility idea as CCFD | Feasible but low interpretability gain | Low |
| Card-TransactionAmt range | Different card types may have different typical transaction ranges | Empirical, but no documentation to support | Low |

**Evidence strength:** Moderate for categorical validity (well-documented field names). Weak for C/D-feature constraints (based on naming convention inference, not official documentation). Very weak for V-features (completely opaque).

**Implementation priority:** Medium. The categorical + non-negativity constraints are easy to implement and defensible. The main contribution from IEEE-CIS is demonstrating Tier A (capability-aware) evaluation on a high-dimensional, partially opaque feature space — a common real-world scenario.

---

### 3.4 Sparkov (Synthetic Credit Card Transactions)

| Dimension | Details |
|-----------|---------|
| **Total features** | 11 (after PII/high-cardinality removal) |
| **Interpretable features** | All 11: category, amt, gender, state, zip, lat, long, city_pop, unix_time, merch_lat, merch_long |
| **Mutable features (candidate)** | amt (attacker controls), category (attacker chooses merchant type), merch_lat/merch_long (attacker picks merchant) |
| **Immutable features (candidate)** | gender (cardholder attribute), state/zip/lat/long (cardholder location), city_pop (census data), unix_time (system timestamp) |

**Hard constraints:**

| ID | Constraint | Type | Evidence | Binding? |
|----|-----------|------|----------|----------|
| s1 | category ∈ {grocery_pos, gas_transport, ...} (14 values) | Categorical validity | Dataset documentation | Weak (OHE) |
| s2 | gender ∈ {M, F} | Categorical validity | Dataset documentation | Weak |
| s3 | state ∈ 50 US states | Categorical validity | Dataset documentation | Weak |
| s4 | amt > 0 | Range | Business logic | Weak |
| s5 | city_pop > 0, integer | Range + integrality | Census data | Weak |
| s6 | unix_time monotonically increasing | Temporal | System-generated | Weak (immutable) |

**Soft filters (candidate, not yet implemented):**

| Filter | Description | Evidence | Priority |
|--------|------------|----------|----------|
| **Geo-consistency: cardholder ↔ state/zip** | lat/long should correspond to stated state/zip | Deterministic lookup (ZIP→lat/long mapping) | **High — strongest non-trivial constraint on Sparkov** |
| Geo-consistency: merchant location | merch_lat/merch_long should be plausible US coordinates | Bounding box check | Medium |
| Category ↔ amount range | Different merchant categories have different typical transaction ranges (grocery ~$10–200, gas ~$20–80) | Empirical from training distribution | Medium — defensible but requires data-driven threshold setting |
| Cardholder ↔ merchant distance | Distance between (lat,long) and (merch_lat,merch_long) should be plausible for the transaction type | Heuristic — in-person categories should be relatively local | Low — no documentation on Sparkov's generation logic |

**Evidence strength:** Moderate for geo-consistency (ZIP↔lat/long is a deterministic public mapping). Weak-to-moderate for category↔amount ranges (empirically derivable but no ground-truth documentation from the Sparkov generator). Sparkov is synthetic data with no public documentation of its generation rules — any constraint we define is inferred, not verified.

**Implementation priority:** Medium. Geo-consistency is the strongest candidate because it's deterministic (lookup-table verifiable). Category↔amount ranges are empirically derivable but require careful framing. The main risk: since Sparkov is synthetic, a reviewer could argue that constraints inferred from synthetic data are circular.

---

## 4. Constraint Richness Gradient — Summary Table

| Dataset | Tier A (Capability) | Tier B (Structure) | Tier C (Formula) | Constraint Richness | Dominant Constraint |
|---------|--------------------|--------------------|------------------|--------------------|--------------------|
| **LCLD** | 11 mutable / 41 immutable features; M6-strict → +18pp robust acc | g2 (inequality), g3 (inequality), g4 (discrete validity) | **g1 (installment formula)** — 98% failure rate | **Very High** | g1 (nonlinear coupling) |
| **Sparkov** | amt/category/merch_loc mutable; cardholder attrs immutable | Geo-consistency (ZIP↔lat/long), category↔amount range | None | **Moderate** | Geo-consistency (lookup) |
| **IEEE-CIS** | TransactionAmt/ProductCD mutable; card/V-features immutable | C-feature non-negativity, D-feature non-negativity, categorical validity | None (V-features opaque) | **Low** | Categorical validity |
| **CCFD** | Amount only mutable; V1–V28 immutable | None (PCA anonymised) | None | **Very Low** | None (statistical plausibility only) |

**Paper narrative:** *"FraudBench deliberately includes datasets spanning the full constraint richness spectrum, from LCLD (rich mathematical constraints, ~0.1% adversarial feasibility) to CCFD (no interpretable constraints, unconstrained attacks provide the only feasible threat assessment). This gradient demonstrates that the appropriate evaluation methodology is dataset-dependent — a finding absent from both TabularBench (which evaluates all datasets with the same constraint-aware attack) and prior fraud detection benchmarks (which ignore constraints entirely)."*

---

## 5. Experimental Roadmap

### Phase 1: Zero-Cost Extensions (no new compute)

| Experiment | Dataset(s) | What it produces | Effort |
|-----------|-----------|-----------------|--------|
| **Cross-dataset feasibility audit** | CCFD, IEEE-CIS, Sparkov | Feasibility rate table across the gradient; confirms CCFD ≈ 100% feasibility (no constraints filter anything) | ~1 day (reuse existing adv examples + new constraint checkers) |
| **Cross-attack robustness transfer** | All 4 | Test AT-trained models against SquareAttack/HopSkipJump; separate from transfer learning | ~1 day (existing models + existing attack code) |
| **Degenerate model audit on FraudBench** | All 4 | Check if any FraudBench model has MCC ≈ 0 | ~0.5 day |

### Phase 2: Tier A/B Implementation (~1–2 weeks)

| Experiment | Dataset(s) | What it produces | Effort |
|-----------|-----------|-----------------|--------|
| **M6-strict cross-dataset** | CCFD, IEEE-CIS, Sparkov | Validates whether capability-based evaluation generalises across datasets; requires defining mutable/immutable partitions per dataset | ~3 days (define partitions + run attacks) |
| **Sparkov geo-consistency checker** | Sparkov | Implements ZIP↔lat/long lookup constraint; measures feasibility rate of existing adversarial examples | ~2 days |
| **IEEE-CIS categorical + count validity** | IEEE-CIS | Implements i1–i6 + C-feature integrality; measures feasibility rate | ~1 day |
| **Mutability mask standardisation** | All 4 | Formalise ConstraintSchema to include mutable/immutable annotations | ~2 days (code refactor) |

### Phase 3: Tier C — Constraint-Aware Attack on LCLD (~2 weeks)

| Experiment | What it produces | Effort |
|-----------|-----------------|--------|
| **Penalty-based projection for g1** | After each CAPGD step, recalculate installment = f(loan_amnt, int_rate, term). Measures: (a) g1 pass rate improvement, (b) attack success rate under constraint, (c) robust PR-AUC with feasible-only adversarial examples | ~5 days (implement projection step in attack loop) |
| **Constrained vs unconstrained systematic comparison** | Reframe existing 182 runs as "unconstrained"; compare with constrained results from above | ~2 days (analysis notebook) |

### Phase 4: Novel Defence — Fraud-Aware AT (if time permits)

| Experiment | What it produces | Effort |
|-----------|-----------------|--------|
| **Per-feature ε allocation** | Allocate perturbation budget proportional to feature mutability + cost during adversarial training | ~1 week |
| **Cost-sensitive weighting** | Higher AT weight for high-value transactions | ~3 days |

---

## 6. Key Caveats and Reviewer Defences

### Caveat 1: PR-AUC insensitivity to mask variants

All eight mask variants (M0–M6) show robust PR-AUC ≈ 0.1051. This means capability constraints improve robust accuracy but not the precision-recall trade-off. **Must report both metrics.** The paper framing should be: "Capability constraints change *which* samples are successfully attacked, but do not change the overall rank-ordering of fraudulent vs legitimate samples under attack."

### Caveat 2: Single-dataset Tier C

Only LCLD currently supports formula-aware evaluation. This is not a weakness if framed correctly: "LCLD serves as the high-constraint showcase; other datasets demonstrate that the same tiered framework gracefully degrades when domain knowledge is limited." TabularBench's own datasets have heterogeneous constraint richness.

### Caveat 3: Sparkov constraint circularity risk

Sparkov is synthetic. Any constraint we define is inferred from the synthetic data itself, not from real-world domain knowledge. Mitigation: (a) geo-consistency is based on external ZIP code databases, not the synthetic data; (b) frame Sparkov constraints as "structural validity" checks rather than "domain" constraints.

### Caveat 4: CCFD statistical plausibility is novel but unverified

Using density estimation or autoencoder reconstruction error to detect off-manifold adversarial examples on PCA-anonymised features is a reasonable approach but has no precedent specific to CCFD. It should be framed as an exploratory extension, not a core result.

### Caveat 5: E1 cost metric has weak discriminative power

The mask ablation's E1 analysis showed M0 and M1 differ by only +0.013 mean cost. If cost-aware evaluation is included in Tier A, the paper must acknowledge that cost sensitivity is dataset-dependent and may not always provide meaningful differentiation.

---

## 7. Paper Structure Mapping

The three-tier framework maps cleanly to paper sections:

| Paper Section | Content | Primary Evidence |
|--------------|---------|-----------------|
| §1 Introduction | Motivation: unconstrained attacks overstate vulnerability on financial data | +55pp ADV/ADV+CTR gap |
| §2 Related Work | TabularBench (NeurIPS 2024), Amazon FDB, ART | Complementary positioning |
| §3 FraudBench Framework | Tier A/B/C definition; constraint richness gradient table; ConstraintSchema design | This document |
| §4 Datasets & Experimental Setup | Per-dataset constraint inventory (Section 3 of this doc); models; attacks; defences | Dataset cards + constraint tables |
| §5 Results — Metric Analysis | Accuracy vs PR-AUC/MCC; degenerate model identification; ranking sensitivity | τ < 0.74, 10/70 degenerate models |
| §5 Results — Constraint Analysis | Cross-dataset feasibility gradient; LCLD deep-dive (g1 dominance); mask ablation attacker spectrum | 0.1% feasibility; M0→M6 +29.8pp |
| §5 Results — Defence Analysis | AT vs input validation vs ensemble; AT as most reliable; input validation negative finding | Existing 182-run registry |
| §6 Discussion | Constraint-richness-determines-methodology argument; complementarity with TabularBench | Gradient table |
| §7 Conclusion | FraudBench fills the intersection of adversarial robustness + fraud-specific evaluation | Synthesis |

---

## 8. Decision Log

| Decision | Rationale | Date |
|----------|-----------|------|
| Tier A/B/C naming (from ChatGPT analysis) over Level 1–4 | Tier names describe evaluation purpose (what we're measuring), not constraint mechanism (how). More reader-friendly for ICAIF audience. | 2026-04-15 |
| CCFD as "no-constraint baseline" rather than attempting PCA manifold filter | PCA manifold filtering is novel but unverified; risks reviewer pushback. Better to use CCFD as the anchor point that shows constraints are dataset-dependent. | 2026-04-15 |
| Penalty-based projection (Path A) over MOEVA integration (Path B) for Tier C | Path A is implementable within ICAIF timeline; Path B requires significant engineering. Path A suffices to demonstrate the constrained-vs-unconstrained gap. | 2026-04-15 |
| Report robust PR-AUC insensitivity as finding, not suppress it | Transparent reporting strengthens credibility. PR-AUC insensitivity is itself an interesting result: it distinguishes capability constraints (which help accuracy) from domain constraints (which affect the PR trade-off). | 2026-04-15 |
| Sparkov geo-consistency via external ZIP database | Avoids circularity critique (constraint derived from external source, not from synthetic data itself). | 2026-04-15 |
