# Figure Claim Audit Report

Command run:

```bash
rg "every dataset|all datasets|all cells|three seeds per configuration|two black-box companions|5000 queries|50-iteration|0\\.083|24\\.1|8\\.5|MLP\\+XGBoost|surrogate-gradient|epsilon = 0\\.1|z-score input validation|input validation reduces robust PR-AUC|input validation degrades|all four datasets|all configs" -n . --glob "*.tex" --glob "*.md" --glob "*.py"
```

## Scope

No `.tex` thesis files are present in this repository, so there are no LaTeX captions or result-section paragraphs to audit here. The scan covered Markdown docs and Python files.

## Findings

- `scripts/analyse_input_validation.py` used an over-broad title phrase, "input validation degrades". It has been revised to "input-validation effects" and the summary now reports 7/8 robust PR-AUC degradation configs plus 1/8 unchanged configs.
- `docs/FIX_DOCUMENT.md` and `docs/figures_verify_goal.md` contain historical problem statements and audit instructions. They intentionally mention stale values such as `0.083`, epsilon-mixing risks, and old checklist items; these are not current result claims.
- No active thesis source in the repository claims "three seeds per configuration" for HSJ. The data audit warns that HSJ has incomplete coverage for IEEE-CIS tree (2 seeds) and LCLD tree (1 seed).
- The current figure CSVs show LCLD neural input validation robust PR-AUC unchanged at `0.1051 -> 0.1051`, matching the required invariant.
