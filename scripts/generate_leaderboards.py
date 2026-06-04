#!/usr/bin/env python3
"""Regenerate the ICDM leaderboard companion CSVs from the canonical results.

The two leaderboards under ``results/icdm_2026/`` are presentation-ready rank
tables derived entirely from the canonical NB1-NB3 outputs:

- ``icdm_leaderboard_defence.csv``      <- capgd_grid_summary.csv
- ``icdm_leaderboard_model_family.csv`` <- square_family_summary.csv,
  square_family_results.csv, kendall_tau_protocol_ranking.csv

Output is deterministic except the ``# Generated:`` date stamp. ``--check``
compares the regenerated content against the committed files (ignoring that
stamp) and exits non-zero on drift, proving the leaderboards still trace to
the canonical CSVs.

Usage:
    .venv/bin/python scripts/generate_leaderboards.py            # rewrite both
    .venv/bin/python scripts/generate_leaderboards.py --check    # verify only
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import pandas as pd

ICDM_DIR = Path("results/icdm_2026")
EPSILON = 0.1
# Fixed presentation row order within each dataset (not rank order).
DEFENCE_ORDER = ["adversarial_training", "none", "input_validation"]
MODEL_ORDER = ["XGBoost", "Ensemble", "MLP"]
# CCFD has no binding constraints, so its deployment-aware protocol is plain A.
# Other datasets must have C2 rows; a missing row fails loudly via .loc.
DEPLOY_PROTOCOL = {"CCFD": "A_unconstrained"}
DEFAULT_DEPLOY = "C2_mask_projection"

log = logging.getLogger("generate_leaderboards")


def build_defence_leaderboard(cg_summary: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, defence): Protocol-A mean vs deployment-aware
    (C2; A for CCFD) mean/std/robust-accuracy, ranked by deploy mean within
    the dataset. MLP / CAPGD / eps=0.1 only."""
    sub = cg_summary[
        (cg_summary.model == "MLP") & (cg_summary.epsilon == EPSILON) & (cg_summary.defence.isin(DEFENCE_ORDER))
    ]
    blocks = []
    for dataset in sorted(sub.dataset.unique()):
        deploy = DEPLOY_PROTOCOL.get(dataset, DEFAULT_DEPLOY)
        rows = []
        for defence in DEFENCE_ORDER:
            cell = sub[(sub.dataset == dataset) & (sub.defence == defence)].set_index("protocol")
            a, c = cell.loc["A_unconstrained"], cell.loc[deploy]
            rows.append(
                {
                    "dataset": dataset,
                    "defence": defence,
                    "deploy_protocol": "A" if deploy == "A_unconstrained" else "C2",
                    "robust_pr_auc_A_mean": a.mean_robust_pr_auc,
                    "robust_pr_auc_deploy_mean": c.mean_robust_pr_auc,
                    "robust_pr_auc_deploy_std": c.std_robust_pr_auc,
                    "robust_accuracy_deploy_mean": c.mean_robust_accuracy,
                }
            )
        block = pd.DataFrame(rows)
        block["rank_deploy"] = block.robust_pr_auc_deploy_mean.rank(ascending=False, method="min").astype(int)
        blocks.append(block)
    out = pd.concat(blocks, ignore_index=True).round(4)
    return out[
        [
            "dataset",
            "defence",
            "deploy_protocol",
            "robust_pr_auc_A_mean",
            "robust_pr_auc_deploy_mean",
            "robust_pr_auc_deploy_std",
            "rank_deploy",
            "robust_accuracy_deploy_mean",
        ]
    ]


def build_model_family_leaderboard(
    sq_summary: pd.DataFrame, sq_results: pd.DataFrame, kendall_tau: pd.DataFrame
) -> pd.DataFrame:
    """One row per (dataset, model): clean PR-AUC (per-seed, ddof=1), robust
    PR-AUC/ROC-AUC under Square A and B with within-dataset ranks, Protocol-A
    aggregate feasibility, and the dataset's A-vs-B weighted-KT distance."""
    sub = sq_summary[(sq_summary.defence == "none") & (sq_summary.epsilon == EPSILON)]
    clean = sq_results[sq_results.protocol == "A_unconstrained"]
    kt_free = kendall_tau[kendall_tau.comparison == "free_model_A_vs_B"].set_index("dataset")
    blocks = []
    for dataset in sorted(sub.dataset.unique()):
        rows = []
        for model in MODEL_ORDER:
            cell = sub[(sub.dataset == dataset) & (sub.model == model)].set_index("protocol")
            a, b = cell.loc["A_unconstrained"], cell.loc["B_posthoc_filter"]
            seeds = clean[(clean.dataset == dataset) & (clean.model == model)].clean_pr_auc
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "clean_pr_auc_mean": seeds.mean(),
                    "clean_pr_auc_std": seeds.std(ddof=1),
                    "robust_pr_auc_A_mean": a.mean_robust_pr_auc,
                    "robust_pr_auc_A_std": a.std_robust_pr_auc,
                    "robust_pr_auc_B_mean": b.mean_robust_pr_auc,
                    "robust_pr_auc_B_std": b.std_robust_pr_auc,
                    "robust_roc_auc_A_mean": a.mean_robust_roc_auc,
                    "robust_roc_auc_B_mean": b.mean_robust_roc_auc,
                    "aggregate_feasibility_A_mean": a.mean_aggregate_feasibility,
                    "kendall_tau_distance_A_to_B": kt_free.loc[dataset, "kt_distance"],
                }
            )
        block = pd.DataFrame(rows)
        for rank_col, mean_col in [
            ("rank_A", "robust_pr_auc_A_mean"),
            ("rank_B", "robust_pr_auc_B_mean"),
        ]:
            block[rank_col] = block[mean_col].rank(ascending=False, method="min").astype(int)
        blocks.append(block)
    out = pd.concat(blocks, ignore_index=True).round(4)
    return out[
        [
            "dataset",
            "model",
            "clean_pr_auc_mean",
            "clean_pr_auc_std",
            "robust_pr_auc_A_mean",
            "robust_pr_auc_A_std",
            "rank_A",
            "robust_pr_auc_B_mean",
            "robust_pr_auc_B_std",
            "rank_B",
            "robust_roc_auc_A_mean",
            "robust_roc_auc_B_mean",
            "aggregate_feasibility_A_mean",
            "kendall_tau_distance_A_to_B",
        ]
    ]


def defence_header(date: str) -> str:
    return f"""\
# FraudBench ICDM 2026 - Defence robustness leaderboard (companion)
# Generated: {date}
# Source files: capgd_grid_summary.csv (derived from capgd_grid_results.csv)
# Filter: model=MLP, attack=CAPGD (white-box), epsilon=0.1, defence in {{none, adversarial_training, input_validation}}
# Aggregation: mean/std over seeds 42,123,456 (n=3)
# Attack note: CAPGD is used because the defence layer is MLP-only and CAPGD is its proper white-box attack (differs from the model-family table, which uses Square).
# deploy_protocol: C2 (deployment-aware: in-attack projection + mutability mask) for constrained datasets; A (unconstrained) for CCFD, which has no binding constraints.
# rank_deploy: 1 = most robust (highest robust PR-AUC) under the deployment-aware protocol within the dataset.
# Key finding: adversarial_training > none > input_validation on constrained datasets; input_validation falls BELOW no-defence (anti-pattern).
# Caveat: CCFD has high seed variance (std ~0.23-0.30); CCFD defence ranking is NOT meaningful (seed-unstable).
# Comment lines start with '#'; import with pandas.read_csv(comment='#') or delete these rows in Sheets.
"""


def model_family_header(date: str) -> str:
    return f"""\
# FraudBench ICDM 2026 - Model-family robustness leaderboard (enriched)
# Generated: {date}
# Source files: square_family_summary.csv, square_family_results.csv, kendall_tau_protocol_ranking.csv
# Filter: attack=Square (score-based black-box), defence=none, epsilon=0.1
# Aggregation: mean/std over seeds 42,123,456 (n=3)
# Std convention: sample std (ddof=1), consistent with the *_summary.csv files.
# Attack note: Square is used because it is the only cross-model-comparable axis (CAPGD is white-box / MLP-only).
# Protocols: A=unconstrained; B=post-hoc feasibility filter (TabularBench ADV+CTR convention).
# rank_A/rank_B: 1 = most robust (highest robust PR-AUC) within the dataset under that protocol.
# kendall_tau_distance_A_to_B: weighted Kendall-Tau distance between A-ranking and B-ranking of the 3 models (0=identical, 1=full reversal); dataset-level, repeated per row.
# aggregate_feasibility_A_mean: fraction of unconstrained-Square adversarial examples passing the full constraint conjunction; explains the A->B robustness jump.
# Caveat: CCFD robust PR-AUC has high seed variance (std ~0.22); CCFD ranking is NOT stable (seed-unstable).
# Comment lines start with '#'; import with pandas.read_csv(comment='#') or delete these rows in Sheets.
"""


def warn_if_header_prose_stale(defence: pd.DataFrame) -> None:
    """The 'Key finding' header line hard-codes the defence ordering; warn if
    the regenerated data no longer supports it (do not silently rewrite prose)."""
    for dataset, group in defence[defence.deploy_protocol == "C2"].groupby("dataset"):
        order = list(group.sort_values("rank_deploy").defence)
        if order != DEFENCE_ORDER:
            log.warning(
                "Key-finding header line is stale for %s: deploy ranking is now %s",
                dataset,
                " > ".join(order),
            )


def _without_date_stamp(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.startswith("# Generated:"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed leaderboards match regenerated content (date stamp ignored)",
    )
    parser.add_argument(
        "--date",
        default=dt.date.today().isoformat(),
        help="date for the '# Generated:' stamp (default: today)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cg_summary = pd.read_csv(ICDM_DIR / "capgd_grid_summary.csv")
    sq_summary = pd.read_csv(ICDM_DIR / "square_family_summary.csv")
    sq_results = pd.read_csv(ICDM_DIR / "square_family_results.csv")
    kendall_tau = pd.read_csv(ICDM_DIR / "kendall_tau_protocol_ranking.csv")

    defence = build_defence_leaderboard(cg_summary)
    warn_if_header_prose_stale(defence)
    model_family = build_model_family_leaderboard(sq_summary, sq_results, kendall_tau)

    outputs = {
        ICDM_DIR / "icdm_leaderboard_defence.csv": defence_header(args.date) + defence.to_csv(index=False),
        ICDM_DIR / "icdm_leaderboard_model_family.csv": model_family_header(args.date)
        + model_family.to_csv(index=False),
    }

    failed = []
    for path, content in outputs.items():
        if args.check:
            on_disk = path.read_text() if path.exists() else ""
            if _without_date_stamp(on_disk) == _without_date_stamp(content):
                log.info("OK %s matches regenerated content", path)
            else:
                log.error("DRIFT %s differs from regenerated content", path)
                failed.append(path)
        else:
            path.write_text(content)
            log.info("wrote %s", path)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
