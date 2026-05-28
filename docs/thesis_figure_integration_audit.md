# Thesis Figure Integration Audit

**TeX root:** `/Users/xitong/Local_Document/githubClone/Capstone-Thesis`
**Included figures:** 7
**Captions parsed:** 27
**Figure labels parsed:** 8
**Passes:** 7
**Warnings:** 9
**Failures:** 0

## Included Figures

- `04_results.tex:169` `figures/adv_vs_advctr_lcld`
  Label: `fig:adv-advctr`
  Resolved thesis asset: `/Users/xitong/Local_Document/githubClone/Capstone-Thesis/figures/adv_vs_advctr_lcld.png`
  Caption: Constraint filtering across the 70-model TabularBench LCLD leaderboard. Each point is one leaderboard model; the axes are unconstrained adversarial recall (ADV) versus constraint-filtered recall (ADV+CTR), and the dashed line marks no filtering effect (ADV${=}$ADV+CTR). Standard-trained models (blue) scatter well below the diagonal---their reported adversarial recall depends heavily on whether constraint filtering is applied---while adversarially-trained models (orange) cluster nearer it. FraudBench's own MLP is not a leaderboard entry; its $+55.11$\,pp ADV+CTR${-}$ADV gap (Table~\ref{tab:tabbench-gap}) exceeds that of every model shown, underscoring how the evaluation protocol, not the model, drives the reported number.
- `04_results.tex:283` `figures/cross_dataset_gradient`
  Label: `fig:cross-gradient`
  Resolved thesis asset: `/Users/xitong/Local_Document/githubClone/Capstone-Thesis/figures/cross_dataset_gradient.png`
  Caption: Cross-dataset adversarial feasibility under unconstrained CAPGD at $\varepsilon=0.1$, companion to Table~\ref{tab:cross-headline} (mean values) and Table~\ref{tab:appx_crossds_perseed} (per-seed values). The three constrained datasets cluster near zero (IEEE-CIS~$0.000$, LCLD~$0.001$, Sparkov~$0.011$) while CCFD alone sits at~$1.000$: the \emph{a priori} four-tier ``constraint richness gradient'' is refuted in favour of a binary split between datasets with and without domain structure.
- `04_results.tex:392` `figures/g1_projection_bars`
  Label: `fig:g1-bars`
  Resolved thesis asset: `/Users/xitong/Local_Document/githubClone/Capstone-Thesis/figures/g1_projection_bars.png`
  Caption: g1-projection on LCLD across three attack regimes (unconstrained, g1-projected, M1${+}$g1) and three seeds; panel values correspond to Table~\ref{tab:g1-headline}. Aggregate feasibility rises $0.001 \to 0.791 \to 1.000$ and the g1 pass rate $0.012 \to 1.000$, while robust PR-AUC stays pinned at $0.105$ throughout. The figure makes the headline visible at a glance: a constraint-aware attacker reaches full feasibility at the same perturbation budget where post-hoc filtering of the unconstrained attack retains almost nothing---the $\approx 960\times$ feasible-flipped gap of \S~\ref{sec:results-g1}.
- `04_results.tex:475` `figures/ieee_ohe_projection_bars`
  Label: `fig:ohe-bars`
  Resolved thesis asset: `/Users/xitong/Local_Document/githubClone/Capstone-Thesis/figures/ieee_ohe_projection_bars.png`
  Caption: OHE-projection on IEEE-CIS across three attack regimes (unconstrained, OHE-projected, M${+}$OHE) and three seeds; values correspond to Table~\ref{tab:ohe-headline}. Filtered success rises $0\% \to 39.6\% \to 100\%$ and robust PR-AUC recovers from $0.063$ to $0.409$ under M${+}$OHE. The wide error bars on aggregate feasibility and D-non-negativity for the OHE-projected regime ($0.507 \pm 0.386$) visualise the seed instability that collapses once the D-fields are frozen under M${+}$OHE (\S~\ref{sec:results-ohe}).
- `04_results.tex:596` `figures/defence_heatmap`
  Label: `fig:defence-heatmap`
  Resolved thesis asset: `/Users/xitong/Local_Document/githubClone/Capstone-Thesis/figures/defence_heatmap.png`
  Caption: Defence effectiveness as $\Delta$ robust PR-AUC relative to the no-defence neural baseline, across four datasets and three defences; cell values correspond to Table~\ref{tab:defence-summary} (the ensemble column is compared against the neural baseline). Green marks improvement, red degradation: adversarial training is positive on every dataset; input validation is negative or non-beneficial across datasets---it reduces clean PR-AUC on every (dataset, model) cell and reduces robust PR-AUC in the non-floor cells, while the LCLD-neural and Sparkov-neural cells remain pinned at the CAPGD floor; the ensemble effect is dataset- dependent, including one negative case on IEEE-CIS.
- `04_results.tex:645` `figures/input_validation_analysis`
  Label: `fig:iv-clean-robust`
  Resolved thesis asset: `/Users/xitong/Local_Document/githubClone/Capstone-Thesis/figures/input_validation_analysis.png`
  Caption: Input-validation effect on clean and robust PR-AUC, four datasets~$\times$ two model families (neural, tree), mean over three seeds at the default $z{=}3$ clipping threshold. Baseline bars (blue~clean, green~robust) are paired with input-validation bars (lavender~clean, red~robust). \emph{Clean} PR-AUC drops on every (dataset, model) cell, indicating that z-score clipping is a general-discrimination loss rather than an adversarial-specific filter. \emph{Robust} PR-AUC drops on most cells---the largest are Sparkov tree ($0.747 \to 0.232$) and IEEE-CIS neural ($0.069 \to 0.018$). The LCLD-neural and Sparkov-neural cells are exceptions: their no-defence robust PR-AUC is already at the CAPGD floor for that (dataset, model) pair ($0.105$ and $0.005$ respectively), leaving no room for the clipping layer to reduce the metric further---a metric-floor artefact, not protection. Per-defence Cohen's $d$ values for the neural model are in Table~\ref{tab:defence-summary}.
- `07_appendix.tex:155` `figures/rank_sensitivity_lcld`
  Label: `fig:rank-sensitivity`
  Resolved thesis asset: `/Users/xitong/Local_Document/githubClone/Capstone-Thesis/figures/rank_sensitivity_lcld.pdf`
  Caption: Re-ranking of the 70-model TabularBench LCLD leaderboard under three alternative scoring metrics (companion to Table~\ref{tab:rank-correlations}). Each panel plots a model's alternative-metric rank against its original accuracy-based rank; points off the diagonal indicate rank changes. The ten degenerate TabNet variants (red) move the most, confirming that accuracy-based ranking rewards models that imbalance-appropriate metrics ($F_1$, MCC, and the harmonic mean with ADV+CTR) penalise.

## Parsed Figure Labels

- `03_method.tex:55` `fig:designmap`
- `04_results.tex:180` `fig:adv-advctr`
- `04_results.tex:292` `fig:cross-gradient`
- `04_results.tex:402` `fig:g1-bars`
- `04_results.tex:483` `fig:ohe-bars`
- `04_results.tex:607` `fig:defence-heatmap`
- `04_results.tex:662` `fig:iv-clean-robust`
- `07_appendix.tex:163` `fig:rank-sensitivity`

## Parsed Captions

- `03_method.tex:42` FraudBench evaluation-protocol design map. Shared datasets, models, seeds, attack budget, preprocessing, and training setup feed three matched protocol paths---Protocol~A (unconstrained), Protocol~B (post-hoc filtering, the TabularBench ADV+CTR convention), and Protocol~C (deployment-aware in-attack constraint integration)---through one fraud-appropriate metric suite. The benchmark reports the pairwise gaps $\Delta(\cdot,\cdot)$ between paths as protocol sensitivity. The headline $\Delta(\text{B},\text{C})\approx960\times$ is a ratio of feasible-flipped counts (M1+g1 versus post-hoc; LCLD, neural model, mean over three seeds); the count is reported rather than the filtered success rate because the rate saturates at $1.0$ under Protocol~C and loses discriminating power. Dashed elements denote methodological controls applied to every path.
- `03_method.tex:117` FraudBench datasets.
- `04_results.tex:35` MVB headline numbers: clean and CAPGD-robust PR-AUC on the neural model without defence, mean $\pm$ std across three seeds.
- `04_results.tex:86` Mask ablation on LCLD, neural model, 3 seeds. Feasibility and per- constraint pass rates reported on seed 42 only.
- `04_results.tex:149` ADV / ADV+CTR gap on LCLD, FraudBench MLP.
- `04_results.tex:170` Constraint filtering across the 70-model TabularBench LCLD leaderboard. Each point is one leaderboard model; the axes are unconstrained adversarial recall (ADV) versus constraint-filtered recall (ADV+CTR), and the dashed line marks no filtering effect (ADV${=}$ADV+CTR). Standard-trained models (blue) scatter well below the diagonal---their reported adversarial recall depends heavily on whether constraint filtering is applied---while adversarially-trained models (orange) cluster nearer it. FraudBench's own MLP is not a leaderboard entry; its $+55.11$\,pp ADV+CTR${-}$ADV gap (Table~\ref{tab:tabbench-gap}) exceeds that of every model shown, underscoring how the evaluation protocol, not the model, drives the reported number.
- `04_results.tex:202` Per-constraint failure rate on 2{,}897 flipped-positive adversarial examples, LCLD unmasked CAPGD.
- `04_results.tex:232` Rank correlation between TabularBench's accuracy-based ranking and the four alternatives, LCLD leaderboard.
- `04_results.tex:264` Cross-dataset feasibility audit, unconstrained CAPGD at $\varepsilon=0.1$, three seeds. Clean and adversarial feasibility, and robust PR-AUC.
- `04_results.tex:284` Cross-dataset adversarial feasibility under unconstrained CAPGD at $\varepsilon=0.1$, companion to Table~\ref{tab:cross-headline} (mean values) and Table~\ref{tab:appx_crossds_perseed} (per-seed values). The three constrained datasets cluster near zero (IEEE-CIS~$0.000$, LCLD~$0.001$, Sparkov~$0.011$) while CCFD alone sits at~$1.000$: the \emph{a priori} four-tier ``constraint richness gradient'' is refuted in favour of a binary split between datasets with and without domain structure.
- `04_results.tex:313` Per-constraint pass rate on IEEE-CIS, unconstrained CAPGD.
- `04_results.tex:332` Per-constraint pass rate on Sparkov, unconstrained CAPGD.
- `04_results.tex:373` g1-projection on LCLD, three seeds. ``Flipped'' is the number of flipped-positive predictions, ``Feas.-flipped'' the number that also pass all constraints, and ``Filtered success'' the ratio of the latter to the former.
- `04_results.tex:393` g1-projection on LCLD across three attack regimes (unconstrained, g1-projected, M1${+}$g1) and three seeds; panel values correspond to Table~\ref{tab:g1-headline}. Aggregate feasibility rises $0.001 \to 0.791 \to 1.000$ and the g1 pass rate $0.012 \to 1.000$, while robust PR-AUC stays pinned at $0.105$ throughout. The figure makes the headline visible at a glance: a constraint-aware attacker reaches full feasibility at the same perturbation budget where post-hoc filtering of the unconstrained attack retains almost nothing---the $\approx 960\times$ feasible-flipped gap of \S~\ref{sec:results-g1}.
- `04_results.tex:459` OHE-projection on IEEE-CIS, three seeds.
- `04_results.tex:476` OHE-projection on IEEE-CIS across three attack regimes (unconstrained, OHE-projected, M${+}$OHE) and three seeds; values correspond to Table~\ref{tab:ohe-headline}. Filtered success rises $0\% \to 39.6\% \to 100\%$ and robust PR-AUC recovers from $0.063$ to $0.409$ under M${+}$OHE. The wide error bars on aggregate feasibility and D-non-negativity for the OHE-projected regime ($0.507 \pm 0.386$) visualise the seed instability that collapses once the D-fields are frozen under M${+}$OHE (\S~\ref{sec:results-ohe}).
- `04_results.tex:521` Capability--feasibility asymmetry: feasible-flipped counts.
- `04_results.tex:551` Defence-level robust PR-AUC on the neural model across four datasets, three seeds. Cohen's $d$ relative to the no-defence baseline. $d > 0$ favours the defence.
- `04_results.tex:597` Defence effectiveness as $\Delta$ robust PR-AUC relative to the no-defence neural baseline, across four datasets and three defences; cell values correspond to Table~\ref{tab:defence-summary} (the ensemble column is compared against the neural baseline). Green marks improvement, red degradation: adversarial training is positive on every dataset; input validation is negative or non-beneficial across datasets---it reduces clean PR-AUC on every (dataset, model) cell and reduces robust PR-AUC in the non-floor cells, while the LCLD-neural and Sparkov-neural cells remain pinned at the CAPGD floor; the ensemble effect is dataset- dependent, including one negative case on IEEE-CIS.
- `04_results.tex:646` Input-validation effect on clean and robust PR-AUC, four datasets~$\times$ two model families (neural, tree), mean over three seeds at the default $z{=}3$ clipping threshold. Baseline bars (blue~clean, green~robust) are paired with input-validation bars (lavender~clean, red~robust). \emph{Clean} PR-AUC drops on every (dataset, model) cell, indicating that z-score clipping is a general-discrimination loss rather than an adversarial-specific filter. \emph{Robust} PR-AUC drops on most cells---the largest are Sparkov tree ($0.747 \to 0.232$) and IEEE-CIS neural ($0.069 \to 0.018$). The LCLD-neural and Sparkov-neural cells are exceptions: their no-defence robust PR-AUC is already at the CAPGD floor for that (dataset, model) pair ($0.105$ and $0.005$ respectively), leaving no room for the clipping layer to reduce the metric further---a metric-floor artefact, not protection. Per-defence Cohen's $d$ values for the neural model are in Table~\ref{tab:defence-summary}.
- `07_appendix.tex:32` Catalogue of domain constraints implemented in FraudBench (LCLD).
- `07_appendix.tex:51` Constraints for IEEE-CIS, Sparkov, and CCFD.
- `07_appendix.tex:88` Per-seed CAPGD results on LCLD under three attack regimes: unconstrained, $g1$-projected, and $M1 + g1$ (full constraint-aware). ``Feas.-flipped'' counts adversarial examples that simultaneously flip the prediction and satisfy all four constraints. Robust PR-AUC is invariant across all 9 rows.
- `07_appendix.tex:114` Full mask-ablation table on LCLD --- 8 mask variants $\times$ 3 seeds, averaged. \texttt{n\_mut} is the mean number of mutable processed features. Feasibility, $g1$ pass and $g4$ pass are seed 42 only (per the experimental protocol \cite{simonetto2024constrained}).
- `07_appendix.tex:137` Per-seed clean and adversarial aggregate feasibility under unconstrained CAPGD at $\varepsilon=0.1$, seeds 42 / 123 / 456. The dispersion column is the standard deviation across the three seeds.
- `07_appendix.tex:156` Re-ranking of the 70-model TabularBench LCLD leaderboard under three alternative scoring metrics (companion to Table~\ref{tab:rank-correlations}). Each panel plots a model's alternative-metric rank against its original accuracy-based rank; points off the diagonal indicate rank changes. The ten degenerate TabNet variants (red) move the most, confirming that accuracy-based ranking rewards models that imbalance-appropriate metrics ($F_1$, MCC, and the harmonic mean with ADV+CTR) penalise.
- `07_appendix.tex:227` Per-study artefact and commit map. Notebook file names are relative to \texttt{notebooks/}; per-study result file names are relative to \texttt{results/adv\_examples/<study>/}. Other paths (master plan, constraint specifications, MVP registry) are relative to the repository root.

## Warnings

- **results/adv_examples/ieee_ohe_projection/ieee_ohe_projection_dose_response.png**: result figure exists but is not used in thesis
  Evidence: provenance=notebooks/ieee_cis_ohe_projection_attack.ipynb
- **results/figures/attack_comparison.png**: result figure exists but is not used in thesis
  Evidence: provenance=scripts/generate_figures.py
- **results/figures/robustness_bars.png**: result figure exists but is not used in thesis
  Evidence: provenance=scripts/generate_figures.py
- **results/figures/robustness_curves.png**: result figure exists but is not used in thesis
  Evidence: provenance=scripts/generate_figures.py
- **results/figures/summary_table.png**: result figure exists but is not used in thesis
  Evidence: provenance=scripts/generate_figures.py
- **results/figures/training_time.png**: result figure exists but is not used in thesis
  Evidence: provenance=scripts/generate_figures.py
- **results/mask_ablation/e1_affordable_curve.png**: result figure exists but is not used in thesis
  Evidence: provenance=notebooks/mask_ablation.ipynb
- **results/mask_ablation/e1_cost_distribution.png**: result figure exists but is not used in thesis
  Evidence: provenance=notebooks/mask_ablation.ipynb
- **results/metric_analysis/rank_sensitivity_lcld.png**: result figure exists but is not used in thesis
  Evidence: provenance=notebooks/tabularbench_metric_analysis.ipynb

## Passes

- **04_results.tex:169 figures/adv_vs_advctr_lcld**: thesis figure has provenance entry
  Evidence: adv_vs_advctr_lcld.png <- notebooks/tabularbench_metric_analysis.ipynb
- **04_results.tex:283 figures/cross_dataset_gradient**: thesis figure has provenance entry
  Evidence: gradient.png <- notebooks/cross_dataset_feasibility.ipynb
- **04_results.tex:392 figures/g1_projection_bars**: thesis figure has provenance entry
  Evidence: g1_projection_bars.png <- notebooks/g1_projection_attack.ipynb
- **04_results.tex:475 figures/ieee_ohe_projection_bars**: thesis figure has provenance entry
  Evidence: ieee_ohe_projection_bars.png <- notebooks/ieee_cis_ohe_projection_attack.ipynb
- **04_results.tex:596 figures/defence_heatmap**: thesis figure has provenance entry
  Evidence: defence_heatmap.png <- scripts/generate_figures.py
- **04_results.tex:645 figures/input_validation_analysis**: thesis figure has provenance entry
  Evidence: input_validation_analysis.png <- scripts/analyse_input_validation.py
- **07_appendix.tex:155 figures/rank_sensitivity_lcld**: thesis figure has provenance entry
  Evidence: rank_sensitivity_lcld.pdf <- notebooks/tabularbench_metric_analysis.ipynb
