# FraudBench Constraint-Aware Evaluation: Strategic Guidance

**Date:** 2026-04-15 (last updated 2026-04-22)
**Status:** Internal guidance document
**Target venue:** ICAIF 2026 (~July deadline, 8-page ACM format)
**Prerequisites:** TabularBench comparison findings, Mask ablation findings, **Cross-dataset feasibility findings (Phase 1, 2026-04-22)**, **g1-projection findings (Phase 3 + M1+g1, 2026-04-22)**

---

## 1. Executive Summary

FraudBench's constraint-aware evaluation effort tests the hypothesis that adversarial robustness on financial tabular data must account for domain constraints, attacker capabilities, and feature semantics. As of 2026-04-22 the following results are established:

1. **Constraint-aware attacks vs post-hoc filtering are not equivalent (the headline finding).** Stock CAPGD on LCLD produces 99.95% domain-infeasible adversarial examples (filtered success = 0.05%). Replacing the attack with a g1-formula-projected variant at the same ε keeps the flipped-prediction count unchanged (≤2-flip delta on the same model) while raising filtered success to **50.2%**. Adding the M1 mutability mask raises it further to **95.3%** — 2774 feasible-and-flipped attacks per seed vs 1 under stock CAPGD, a **~2000× underestimate** of realistic attacker success. The +55pp ADV/ADV+CTR gap (`tabularbench_comparison_findings.md`) measures attack-generation inefficiency, not defender safety. (`g1_projection_findings.md`)

2. **Empirical "constraint richness" is binary, not a gradient.** Three of four datasets (IEEE-CIS, LCLD, Sparkov) cluster below 1% adversarial feasibility under stock CAPGD; CCFD stands alone at 100%. The most restrictive is **IEEE-CIS at 0.014% adv feas** — *not* LCLD as the a-priori "richness" framing predicted — because high-dimensional OHE expansions implicitly create many cheap per-row checks that, ANDed together, filter harder than a single nonlinear formula in isolation. The paper claim should be a binary dichotomy ("any structure" vs "statistical only"), not a 4-tier ordering. (`cross_dataset_feasibility_findings.md`)

3. **Capability spectrum.** Mask ablation (M0→M6-strict) yields a monotone robust accuracy gradient from 0.042 to 0.340 (+29.8 pp), confirming that realistic attacker-capability modelling materially changes robustness assessment. Robust PR-AUC remains locked at **0.1051 ± 0.0001 across six independent experimental axes** (8 mask variants, 3 cross-dataset seeds, 3 g1-projection regimes) — capability and constraint regimes improve robust accuracy but not the precision-recall rank ordering at LCLD ε=0.1.

4. **Metric sensitivity.** Accuracy-based ranking on imbalanced fraud data produces substantially different model orderings than F1/MCC (Kendall's τ < 0.74), with 10/70 degenerate TabNet models unpenalised by accuracy.

The original framing positioned LCLD as the "Very High" constraint anchor with other datasets degrading gracefully along a richness gradient. The cross-dataset audit refuted that gradient, but the central thesis — *constraints matter, methodology is dataset-dependent, and constraint-aware attacks reveal an attack surface that post-hoc filtering hides* — is **strengthened** by the binary result and the M1+g1 progression.

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

## 4. Constraint Richness — A-Priori vs Empirical

The original a-priori "richness gradient" (LCLD ≫ Sparkov > IEEE-CIS ≫ CCFD) did not hold under the cross-dataset feasibility audit (`cross_dataset_feasibility_findings.md`, 2026-04-22). The empirical ordering, by adversarial feasibility under stock CAPGD at ε=0.1, is shown below; the table is reordered ascending by adv feasibility (most restrictive first).

| Dataset | Tier A (Capability) | Tier B (Structure) | Tier C (Formula) | **Empirical adv feas** | A-priori "richness" | **Empirical binding constraint** (lowest adv pass) |
|---------|---------------------|--------------------|------------------|-----------------------:|---------------------|----------------------------------------------------|
| **IEEE-CIS** | TransactionAmt/ProductCD mutable; card/V immutable | ProductCD/card4/card6 OHE validity (binding); D1–D15 non-negativity (modest); C1–C14 non-negativity (preserved) | None (V opaque) | **0.014%** | "Low" | `i_product_ohe` (0.018 adv pass) — 3 OHE checks dominate; C-nonneg passes 1.000 |
| **LCLD**     | 11 mutable / 41 immutable; M6-strict → +18pp robust acc | g2/g3 inequalities, g4 OHE | **g1 (installment formula)** — 99% adv-failure | **0.093%** | "Very High" | `g1_installment` (0.012); `g4_term_ohe` second (0.185) |
| **Sparkov**  | amt/category/merch mutable; cardholder attrs immutable | state/category/gender OHE (binding); merch lat/long bbox (preserved) | None | **0.38%** | "Moderate" | `s_state_ohe` (0.0002) — geo-bbox passes 0.992 |
| **CCFD**     | Amount mutable; V1–V28 immutable | None (PCA anonymised) | None | **100%** | "Very Low" | None |

**Empirical finding.** IEEE-CIS — framed a-priori as "Low" — is empirically the *most* restrictive, an order of magnitude tighter than LCLD's 0.093%. The per-constraint decomposition (`cross_dataset_feasibility_findings.md`, 2026-04-22 update) reveals that the discriminating factor across constrained datasets is the **count of OHE-validity checks**, not non-negativity, geo-bbox, or formula constraints:

- **OHE-validity is the universal binding constraint at ε=0.1.** It's the cheapest constraint to detect (single argmax per OHE block) and the hardest for unconstrained CAPGD to satisfy: preprocessing distributes attack budget across all OHE columns equally, never producing a clean one-hot. IEEE-CIS has 3 (ProductCD/card4/card6 → 0.018/0.048/0.206 adv pass), Sparkov has 3 (state/category/gender → 0.0002/0.017/0.265), LCLD has 1 (term → 0.185 adv pass — second-most-binding after g1).
- **Non-negativity, range, and bounding-box constraints are largely preserved.** C1–C14 non-negativity passes 1.000 because StandardScaler centering keeps perturbations comfortably above zero in raw space; merchant lat/long bbox passes 0.992 because ε in scaled space barely shifts continental-scale coordinates.
- **Formula constraints (LCLD g1) are dataset-specific** and dominate where they exist (0.012 adv pass), but the cross-dataset evidence shows they are *not* what makes the constrained datasets restrictive in general — OHE-validity does that work everywhere.

The three constrained datasets cluster <1%; CCFD stands alone at 100%.

**Refined paper claim.** Collapse the four-tier richness ordering to a binary dichotomy:

- **Constrained — any domain structure** (IEEE-CIS, LCLD, Sparkov): post-hoc feasibility filtering rejects ≥99% of unconstrained CAPGD output. Evaluation requires either (a) constraint-aware attacks, or (b) explicit reporting that the unconstrained robust metric overstates real attack success.
- **Unconstrained — statistical plausibility only** (CCFD): no domain constraint filters the attack output meaningfully. Tier A (capability-aware) is the only useful evaluation layer; constraint-aware methods provide no additional signal here.

This dichotomy is supported empirically (one column, four numbers) and is a stronger paper position than the original a-priori gradient.

**Paper narrative (revised):** *"FraudBench includes datasets spanning the full domain-constraint spectrum, from IEEE-CIS (many cheap structural constraints, 0.014% adv feasibility) and LCLD (one nonlinear formula constraint, 0.093%) to CCFD (no domain constraint applies, 100% feasibility). The cluster <1% vs 100% binary structure demonstrates that constraint-aware evaluation is necessary on three of four datasets and insufficient/inapplicable on the fourth — a methodology choice that is dataset-dependent. This is absent from both TabularBench (which uses the same constraint-aware attack on all datasets) and prior fraud detection benchmarks (which ignore constraints entirely)."*

**Note on the constraint-aware attack result (LCLD only).** The above table reports stock CAPGD adv feasibility. On LCLD, replacing stock CAPGD with g1-projected CAPGD raises adv feasibility from 0.093% to 69.3% at the same ε, and adding the M1 mutability mask raises it to 95.8% — see §1 result #1 and `g1_projection_findings.md`. IEEE-CIS / Sparkov have not been subjected to the equivalent structure-aware attack (Phase 2 work, partially obsoleted as motivation but still valuable as cross-dataset evidence — see §5).

---

## 5. Experimental Roadmap

### Phase 1: Zero-Cost Extensions — **DONE (2026-04-22)**

| Experiment | Status | Outcome |
|-----------|--------|---------|
| **Cross-dataset feasibility audit** | ✅ **Done** | `cross_dataset_feasibility_findings.md`. Refuted the a-priori richness gradient: empirical ordering is IEEE-CIS (0.014%) < LCLD (0.093%) < Sparkov (0.38%) ≪ CCFD (100%). Resulting refined thesis (binary presence/absence) folded into §1 and §4. |
| **Cross-attack robustness transfer** | Pending | Not yet run. Lower priority than Phase 2/3 follow-ups. |
| **Degenerate model audit on FraudBench** | Pending | Lower priority. |

### Phase 2: Tier A/B Implementation — **partially obsoleted as motivation, still valuable as cross-dataset evidence**

The Phase 1 audit refuted the tiered framework Phase 2 was designed to support, so the original *motivation* (filling out a 3-tier table) no longer applies. However, running constraint-aware attacks on IEEE-CIS / Sparkov would directly test whether the M1+g1-style "constraint-aware attack recovers the apparent strength loss" pattern (Phase 3 finding) generalizes beyond LCLD. That generalization is the strongest version of the paper's headline claim.

| Experiment | Status | Notes |
|-----------|--------|-------|
| **M-mask + structure-projection on IEEE-CIS** | Pending | Most valuable next step. IEEE-CIS has no formula constraint but rich OHE/non-negativity structure — project onto C/D non-negativity + ProductCD/card4/card6 OHE validity each step. Expected pattern: filtered success goes from 0.014% to >50% with attack power preserved. |
| **M-mask + structure-projection on Sparkov** | Pending | Geo-consistency projection (snap merch_lat/merch_long to nearest valid coordinate) is implementable but lower priority than IEEE-CIS. |
| **Per-constraint decomposition on IEEE-CIS / Sparkov** | Open flag | Already in the cross-dataset CSV; just needs an analysis cell to print per-constraint means. Cited in `cross_dataset_feasibility_findings.md` §Open flags #2. |
| **Mutability mask standardisation** | Deferred | Mask logic lives in `mask_ablation.ipynb` Cell 6; promote to library only if Phase 2 cross-dataset attacks materialize. |

### Phase 3: Tier C — Constraint-Aware Attack on LCLD — **DONE (2026-04-22)**

| Experiment | Status | Outcome |
|-----------|--------|---------|
| **Per-step g1 projection for CAPGD** | ✅ **Done** | `g1_projection_findings.md`. Same flipped-prediction count as stock CAPGD (≤2-flip delta same-model); filtered success 0.05% → 50.2%. |
| **M1 mask + g1 projection (follow-up)** | ✅ **Done** | Same doc. Closes g3 credit-bureau gap; filtered success 0.05% → 50.2% → **95.3%**. Includes float64 installment re-derivation fix (commit `adcd78d`). |
| **Constrained vs unconstrained systematic comparison** | ✅ **Done** | Same trained model per seed for the unconstrained vs g1-projected pair; M1+g1 retrains (between-model caveat documented). |

### Phase 4: Novel Defence — Fraud-Aware AT (if time permits)

| Experiment | What it produces | Effort |
|-----------|-----------------|--------|
| **Per-feature ε allocation** | Allocate perturbation budget proportional to feature mutability + cost during adversarial training | ~1 week |
| **Cost-sensitive weighting** | Higher AT weight for high-value transactions | ~3 days |

### Soft blockers before paper table is final

1. **Seed-42 sparse-categorical issue on LCLD** (`cross_dataset_feasibility_findings.md` §Open flags #1; `g1_projection_findings.md` Caveat #1). One sparse categorical (likely `addr_state` or `purpose`) has a test-only value in the seed-42 stratified split, capping seed-42 clean-feasibility at 0.888 vs ~0.991 on the other seeds and M1+g1 aggregate at 0.890 vs ~0.993. Fix: `OneHotEncoder(handle_unknown="ignore")` or stratify the split on the offending categorical. Effort: <1 day.
2. **CCFD robust PR-AUC variance** (cross-dataset Open flag #3). Robust PR-AUC swings 0.58 ± 0.23 across 3 seeds — likely model-init variance with 0.17% positive class and 20 epochs. Either run more seeds or note in the paper.

---

## 6. Key Caveats and Reviewer Defences

### Caveat 1: PR-AUC insensitivity across six experimental axes

Robust PR-AUC on LCLD is locked at 0.1051 ± 0.0001 across:
- 8 mask-ablation variants (M0–M6, `mask_ablation_findings.md`)
- 3 seeds in the cross-dataset audit (`cross_dataset_feasibility_findings.md`)
- 3 attack regimes in the g1-projection study — unconstrained, g1-projected, M1+g1 (`g1_projection_findings.md`, σ ≤ 9×10⁻⁶)

This is six independent axes where PR-AUC is unmoved at LCLD ε=0.1. **Must report both PR-AUC and robust accuracy** — robust accuracy moves 0.038 → 0.042 → 0.175 across the three g1-projection regimes, so it discriminates; PR-AUC does not. The paper framing should be: "Capability and constraint regimes change *which* samples are successfully attacked, but at LCLD ε=0.1 the rank ordering of fraudulent vs legitimate samples under attack is invariant. PR-AUC alone cannot distinguish constraint-aware from constraint-unaware attacks here." This deserves its own paper subsection rather than a footnote.

### Caveat 2: Single-dataset Tier C — formula-aware result confined to LCLD

Only LCLD has a verified nonlinear formula constraint (g1). The headline 0.05% → 50.2% → 95.3% filtered-success progression is therefore a single-dataset result. Mitigation: (a) the §1 result #2 binary dichotomy already establishes that *any* domain structure (not just formulas) suffices to make constraint-aware attacks recover hidden attack surface; (b) Phase 2 cross-dataset M+structure-projection runs on IEEE-CIS / Sparkov would directly test generalization (see §5 Phase 2). Until those land, the paper should frame the M1+g1 result as "demonstrated on LCLD; expected to hold on any dataset where post-hoc filtering rejects most adv examples — testable extension."

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
| §1 Introduction | Motivation: unconstrained attacks overstate vulnerability on financial data — *but post-hoc filtering also underestimates real attacker success.* | +55pp ADV/ADV+CTR gap reframed via 0.05/50.2/95.3% filtered-success progression (`g1_projection_findings.md`) |
| §2 Related Work | TabularBench (NeurIPS 2024), Amazon FDB, ART | Complementary positioning |
| §3 FraudBench Framework | Tier A/B/C definition (with empirical caveat — see §4 below); ConstraintSchema design; constraint-aware vs post-hoc dichotomy | This doc §1, §4 |
| §4 Datasets & Experimental Setup | Per-dataset constraint inventory (§3 of this doc); models; attacks; defences | Dataset cards + constraint tables |
| §5 Results — Metric Analysis | Accuracy vs PR-AUC/MCC; degenerate model identification; ranking sensitivity | τ < 0.74, 10/70 degenerate models |
| §5 Results — Constraint Analysis | **Headline:** g1+M1 progression on LCLD (0.05/50.2/95.3% filtered success); empirical constrained-vs-unconstrained dichotomy across 4 datasets; mask ablation attacker spectrum; PR-AUC invariance across 6 axes | `g1_projection_findings.md`, `cross_dataset_feasibility_findings.md`, `mask_ablation_findings.md` |
| §5 Results — Defence Analysis | AT vs input validation vs ensemble; AT as most reliable; input validation negative finding | Existing 182-run registry |
| §6 Discussion | Constraint-aware-attacks-recover-hidden-attack-surface argument; complementarity with TabularBench; methodology is dataset-dependent | Headline progression + binary dichotomy table |
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
| Collapse 4-tier richness gradient to binary "constrained vs unconstrained" | Phase 1 cross-dataset audit refuted the a-priori ordering (IEEE-CIS most restrictive, not LCLD). Empirical data clusters <1% vs 100% — a binary dichotomy is the empirically supported claim. | 2026-04-22 |
| Headline finding = constraint-aware attack vs post-hoc filtering equivalence (not the ADV/ADV+CTR gap) | Phase 3 + M1+g1 follow-up shows post-hoc filtering rejects 99.95% of attacks but constraint-aware generation produces ~2774× more useful attacks. The gap measured attack-generation inefficiency, not defender safety. | 2026-04-22 |
| M1+g1 retrains the model per seed (different init from Cell 10) — accept as caveat rather than refactor | Filtered-success rate is a within-row ratio so the headline result is unaffected. Refactoring to share models is engineering effort with no scientific gain. | 2026-04-22 |
| Replace "C/D non-negativity / geo-bbox dominate" with "OHE-validity is the universal binding constraint" | Per-constraint decomposition (cross-dataset Cell 14, 2026-04-22) showed C-nonneg passes 1.000 on IEEE-CIS and merch-bbox passes 0.992 on Sparkov; the actual binding constraints are OHE-validity checks on every constrained dataset. The "many cheap checks ANDed" framing still holds — the *type* of cheap check is OHE, not non-negativity. | 2026-04-22 |
