# Figure Data Audit Report

**Figures discovered:** 16
**Passes:** 56
**Warnings:** 12
**Failures:** 0

## Warnings

- **HSJ seed coverage**: HSJ must not be described as complete 3-seed coverage for every group
  Evidence: ('ieee_cis', 'tree', 'none') seeds=2; ('lcld', 'tree', 'none') seeds=1
- **results/adv_examples/ieee_ohe_projection/ieee_ohe_projection_summary.csv**: legacy n_mutable_dims blanks are backfilled from notebook output
  Evidence: source rows predate n_mutable_dims; summary/dose-response use notebook-recorded values
- **results/mask_ablation/e1_cost_distribution.png; results/mask_ablation/e1_affordable_curve.png**: original LCLD training split for p1/p99 cost ranges is not committed; figure verification is summary-level only
  Evidence: source=e1_cost_summary.csv; checks=scale proportionality
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

- **results/adv_examples/cross_dataset_feasibility/gradient.png**: valid nonblank image
  Evidence: size=1080x600; bytes=38258
- **results/adv_examples/g1_projection/g1_projection_bars.png**: valid nonblank image
  Evidence: size=2250x570; bytes=83313
- **results/adv_examples/ieee_ohe_projection/ieee_ohe_projection_bars.png**: valid nonblank image
  Evidence: size=2250x570; bytes=74413
- **results/adv_examples/ieee_ohe_projection/ieee_ohe_projection_dose_response.png**: valid nonblank image
  Evidence: size=1800x600; bytes=85925
- **results/figures/attack_comparison.png**: valid nonblank image
  Evidence: size=1500x900; bytes=47519
- **results/figures/defence_heatmap.png**: valid nonblank image
  Evidence: size=1200x600; bytes=69704
- **results/figures/input_validation_analysis.png**: valid nonblank image
  Evidence: size=2984x742; bytes=52622
- **results/figures/robustness_bars.png**: valid nonblank image
  Evidence: size=2100x1500; bytes=190186
- **results/figures/robustness_curves.png**: valid nonblank image
  Evidence: size=2700x1500; bytes=118023
- **results/figures/summary_table.png**: valid nonblank image
  Evidence: size=2685x2212; bytes=330772
- **results/figures/training_time.png**: valid nonblank image
  Evidence: size=1500x900; bytes=45997
- **results/mask_ablation/e1_affordable_curve.png**: valid nonblank image
  Evidence: size=1200x600; bytes=51009
- **results/mask_ablation/e1_cost_distribution.png**: valid nonblank image
  Evidence: size=1200x600; bytes=31728
- **results/metric_analysis/adv_vs_advctr_lcld.png**: valid nonblank image
  Evidence: size=1184x881; bytes=93797
- **results/metric_analysis/rank_sensitivity_lcld.pdf**: valid PDF
  Evidence: bytes=20357
- **results/metric_analysis/rank_sensitivity_lcld.png**: valid nonblank image
  Evidence: size=2234x731; bytes=173008
- **results/adv_examples/cross_dataset_feasibility/gradient.png**: direct filename provenance found
  Evidence: notebooks/cross_dataset_feasibility.ipynb
- **results/adv_examples/g1_projection/g1_projection_bars.png**: direct filename provenance found
  Evidence: notebooks/g1_projection_attack.ipynb
- **results/adv_examples/ieee_ohe_projection/ieee_ohe_projection_bars.png**: direct filename provenance found
  Evidence: notebooks/ieee_cis_ohe_projection_attack.ipynb
- **results/adv_examples/ieee_ohe_projection/ieee_ohe_projection_dose_response.png**: direct filename provenance found
  Evidence: notebooks/ieee_cis_ohe_projection_attack.ipynb
- **results/figures/attack_comparison.png**: direct filename provenance found
  Evidence: scripts/generate_figures.py
- **results/figures/defence_heatmap.png**: direct filename provenance found
  Evidence: scripts/generate_figures.py
- **results/figures/input_validation_analysis.png**: direct filename provenance found
  Evidence: scripts/analyse_input_validation.py
- **results/figures/robustness_bars.png**: direct filename provenance found
  Evidence: scripts/generate_figures.py
- **results/figures/robustness_curves.png**: direct filename provenance found
  Evidence: scripts/generate_figures.py
- **results/figures/summary_table.png**: direct filename provenance found
  Evidence: scripts/generate_figures.py
- **results/figures/training_time.png**: direct filename provenance found
  Evidence: scripts/generate_figures.py
- **results/mask_ablation/e1_affordable_curve.png**: direct filename provenance found
  Evidence: notebooks/mask_ablation.ipynb
- **results/mask_ablation/e1_cost_distribution.png**: direct filename provenance found
  Evidence: notebooks/mask_ablation.ipynb
- **results/metric_analysis/adv_vs_advctr_lcld.png**: direct filename provenance found
  Evidence: notebooks/tabularbench_metric_analysis.ipynb
- **results/metric_analysis/rank_sensitivity_lcld.pdf**: direct filename provenance found
  Evidence: notebooks/tabularbench_metric_analysis.ipynb
- **results/metric_analysis/rank_sensitivity_lcld.png**: direct filename provenance found
  Evidence: notebooks/tabularbench_metric_analysis.ipynb
- **results/figures/summary_table.csv**: reproduces from canonical registry
  Evidence: source=results/registry_clean.csv; filters=exclude z5/z10/eps_sweep; groupby=dataset,model_type,defence_type,attack_type,attack_epsilon
- **results/figures/input_validation_analysis.csv**: reproduces from canonical registry
  Evidence: source=results/registry_clean.csv; filters=epsilon=0.1, exclude z5/z10/eps_sweep; groupby=dataset,model_type,attack_type,attack_epsilon
- **results/figures/adv_training_tradeoffs.csv**: reproduces from canonical registry
  Evidence: source=results/registry_clean.csv; filters=exclude z5/z10/eps_sweep; groupby=dataset,model_type,attack_type,attack_epsilon
- **results/figures/statistical_tests.csv**: reproduces from canonical registry
  Evidence: source=results/registry_clean.csv; filters=epsilon=0.1, exclude z5/z10/eps_sweep; pairs=seed,attack_type,attack_epsilon
- **default figure filters**: exclude z5/z10 and epsilon-sweep rows
  Evidence: rows=102; canonical_epsilon=0.1
- **results/figures/robustness_bars.png**: canonical epsilon source slice is non-empty
  Evidence: rows=35; datasets=['ccfd', 'ieee_cis', 'lcld', 'sparkov']
- **results/figures/attack_comparison.png; results/figures/defence_heatmap.png**: default attack and defence source slices are non-empty
  Evidence: attacks=['capgd', 'hopskipjump', 'square']; defences=['adversarial_training', 'ensemble', 'input_validation', 'none']
- **results/figures/robustness_curves.png**: uses neural epsilon sweeps plus canonical single-point defences
  Evidence: series=12; multi_epsilon_series=4
- **results/figures/input_validation_analysis.csv**: LCLD neural robust PR-AUC is unchanged between baseline and input validation
  Evidence: value=0.1051; tolerance=0.0005
- **tree + CAPGD no-op invariant**: clean and robust PR-AUC are identical and should be interpreted as CAPGD inapplicability
  Evidence: rows=24
- **results/adv_examples/g1_projection/g1_projection_summary.csv**: reproduces from source CSV
  Evidence: source=g1_projection_results.csv; groupby=attack; metrics=robustness, feasibility, filtered_success_rate
- **results/adv_examples/ieee_ohe_projection/ieee_ohe_projection_summary.csv**: reproduces from source CSV
  Evidence: source=ieee_ohe_projection_results.csv; groupby=attack; legacy n_mutable_dims backfill from notebook output
- **results/adv_examples/cross_dataset_feasibility/cross_dataset_feasibility_gradient.csv**: reproduces from source CSV
  Evidence: source=cross_dataset_feasibility_results.csv; groupby=dataset; metrics=clean/adv feasibility and robust PR-AUC
- **results/mask_ablation/mask_ablation_summary.csv**: reproduces from mask result and feasibility CSVs
  Evidence: source=mask_ablation_results.csv,mask_ablation_feasibility.csv; groupby=variant
- **results/mask_ablation/e1_cost_summary.csv**: cost scale sensitivity is internally consistent
  Evidence: source=e1_cost_summary.csv; checks=0.5x/2.0x mean,median,p95 proportionality
- **results/metric_analysis/rank_sensitivity_lcld.png; results/metric_analysis/rank_sensitivity_lcld.pdf**: leaderboard formulas, ranks, and correlations are internally consistent
  Evidence: source=lcld_leaderboard_reranked.csv,rank_correlation_results.csv; degenerate_models=10
- **results/metric_analysis/adv_vs_advctr_lcld.png**: ADV vs ADV+CTR comparison CSV is internally consistent
  Evidence: source=adv_advctr_comparison.csv; checks=variants,gap_pp,bounded rates
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
