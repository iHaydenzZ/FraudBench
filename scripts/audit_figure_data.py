#!/usr/bin/env python3
"""Audit committed result figures and their source CSVs.

The audit is intentionally read-only for result artefacts. It verifies image
integrity, traces direct script/notebook provenance, recomputes committed
figure-adjacent CSVs from canonical CSVs, and writes machine- and
human-readable audit reports.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

from scripts.generate_figures import (
    CANONICAL_EPSILON,
    aggregate_seeds,
    filter_default_analysis_rows,
    filter_robustness_curve_rows,
    load_registry,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_REGISTRY = DEFAULT_RESULTS_DIR / "registry_clean.csv"
FIGURE_SUFFIXES = {".png", ".pdf", ".jpg", ".jpeg", ".svg"}
THESIS_AUDIT_REPORT = ROOT / "docs" / "thesis_figure_integration_audit.md"

STALE_CLAIM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("0.083", re.compile(r"\b0\.083\b", re.IGNORECASE)),
    ("5000 queries", re.compile(r"\b5\s*(?:\{,\}|,)?\s*000\s+queries\b", re.IGNORECASE)),
    ("50-iteration", re.compile(r"\b50\s*[-\u2010\u2011\u2013\u2014]\s*iteration\b", re.IGNORECASE)),
    ("two black-box companions", re.compile(r"\btwo\s+black[-\s]+box\s+companions\b", re.IGNORECASE)),
    ("three seeds per configuration", re.compile(r"\bthree\s+seeds\s+per\s+configuration\b", re.IGNORECASE)),
    (
        "input validation reduces robust PR-AUC on every dataset",
        re.compile(
            r"input\s+validation\s+reduces\s+robust\s+PR[-\s]+AUC\s+on\s+every\s+dataset",
            re.IGNORECASE,
        ),
    ),
    ("surrogate-gradient", re.compile(r"\bsurrogate[-\s]+gradient\b", re.IGNORECASE)),
)


@dataclass(frozen=True)
class AuditFinding:
    status: str
    item: str
    detail: str
    evidence: str


@dataclass(frozen=True)
class TexEntry:
    kind: str
    value: str
    tex_file: str
    line_no: int
    location: str
    start: int
    end: int


@dataclass(frozen=True)
class ThesisFigure:
    tex_file: str
    line_no: int
    location: str
    include_path: str
    caption: str
    label: str
    resolved_path: str | None
    context: str


@dataclass(frozen=True)
class StaleClaim:
    claim: str
    location: str
    matched_text: str
    context: str


@dataclass(frozen=True)
class TexInventory:
    tex_root: str
    figures: list[ThesisFigure]
    captions: list[TexEntry]
    labels: list[TexEntry]
    stale_claims: list[StaleClaim]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def display_path(path: Path) -> str:
    try:
        return rel(path)
    except ValueError:
        return str(path)


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy().replace("", np.nan)


def frames_match(expected: pd.DataFrame, actual: pd.DataFrame) -> tuple[bool, str]:
    exp = normalize_frame(expected).reset_index(drop=True)
    act = normalize_frame(actual).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            exp,
            act,
            check_dtype=False,
            check_exact=False,
            rtol=1e-8,
            atol=1e-10,
        )
        return True, ""
    except AssertionError as exc:
        return False, str(exc).splitlines()[0]


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    df = filter_default_analysis_rows(df)
    agg = aggregate_seeds(df)
    summary_cols = ["dataset", "model_type", "defence_type", "attack_type", "attack_epsilon"]
    metric_pairs = [
        ("clean_pr_auc_mean", "clean_pr_auc_std"),
        ("robust_pr_auc_mean", "robust_pr_auc_std"),
        ("clean_f1_mean", "clean_f1_std"),
        ("robust_f1_mean", "robust_f1_std"),
    ]

    rows: list[dict[str, object]] = []
    for _, row in agg.iterrows():
        out = {c: row[c] for c in summary_cols if c in row.index}
        for mean_col, std_col in metric_pairs:
            mean_value = row.get(mean_col, np.nan)
            std_value = row.get(std_col, np.nan)
            label = mean_col.replace("_mean", "")
            if pd.notna(mean_value):
                if pd.notna(std_value):
                    out[label] = f"{mean_value:.4f} +/- {std_value:.4f}"
                else:
                    out[label] = f"{mean_value:.4f}"
            else:
                out[label] = "n/a"
        rows.append(out)

    for row in rows:
        if row.get("model_type") == "tree" and row.get("attack_type") == "capgd":
            if row.get("robust_pr_auc") != "n/a":
                row["robust_pr_auc"] = f"{row['robust_pr_auc']} \u2020"
    return pd.DataFrame(rows)


def build_input_validation(df: pd.DataFrame) -> pd.DataFrame:
    from scripts.analyse_input_validation import aggregate_seeds as _aggregate
    from scripts.analyse_input_validation import compute_degradation

    return compute_degradation(_aggregate(df))


DATASET_CHARS = {
    "ccfd": {"fraud_rate": 0.00173, "n_features": 30, "n_samples": 284807},
    "ieee_cis": {"fraud_rate": 0.03499, "n_features": 394, "n_samples": 590540},
    "lcld": {"fraud_rate": 0.11300, "n_features": 57, "n_samples": 100653},
    "sparkov": {"fraud_rate": 0.00579, "n_features": 22, "n_samples": 1296675},
}


def build_adv_training(df: pd.DataFrame) -> pd.DataFrame:
    df = filter_default_analysis_rows(df)
    agg = aggregate_seeds(df)
    merge_keys = ["dataset", "model_type", "attack_type", "attack_epsilon"]
    baseline = agg[agg["defence_type"] == "none"].copy()
    adv_train = agg[agg["defence_type"] == "adversarial_training"].copy()
    merged = adv_train.merge(
        baseline[merge_keys + ["clean_pr_auc_mean", "robust_pr_auc_mean"]],
        on=merge_keys,
        suffixes=("_at", "_base"),
    )
    merged["clean_cost"] = merged["clean_pr_auc_mean_at"] - merged["clean_pr_auc_mean_base"]
    merged["robust_gain"] = merged["robust_pr_auc_mean_at"] - merged["robust_pr_auc_mean_base"]
    for dataset, chars in DATASET_CHARS.items():
        mask = merged["dataset"] == dataset
        for key, value in chars.items():
            merged.loc[mask, key] = value
    return merged


def build_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    from scripts.statistical_tests import pairwise_defence_tests

    return pairwise_defence_tests(df)


def build_g1_summary(results: pd.DataFrame) -> pd.DataFrame:
    agg_cols = {
        "robust_pr_auc": ["mean", "std"],
        "robust_accuracy": ["mean", "std"],
        "robust_recall": ["mean", "std"],
        "adv_feasibility": ["mean", "std"],
        "adv_g1_installment": ["mean", "std"],
        "adv_g2_open_total": ["mean", "std"],
        "adv_g3_bankruptcy": ["mean", "std"],
        "adv_g4_term_ohe": ["mean", "std"],
        "flipped_positives": ["mean"],
        "feasible_flipped": ["mean"],
    }
    summary = results.groupby("attack").agg(agg_cols)
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
    summary = summary.reset_index()
    summary["filtered_success_rate"] = (
        summary["feasible_flipped_mean"] / summary["flipped_positives_mean"].replace(0, np.nan)
    )
    order = ["unconstrained", "g1proj", "m1_g1proj"]
    summary["__order"] = summary["attack"].map({key: i for i, key in enumerate(order)})
    return summary.sort_values("__order").drop(columns="__order").reset_index(drop=True)


def extract_ieee_backfill_map() -> dict[str, int]:
    nb_path = ROOT / "notebooks" / "ieee_cis_ohe_projection_attack.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    text = json.dumps(nb)
    for match in re.finditer(r"Backfill n_mutable_dims for legacy rows: \{([^}]+)\}", text):
        out: dict[str, int] = {}
        for key, value in re.findall(r"'([^']+)':\s*([0-9]+)", match.group(0)):
            out[key] = int(value)
        if out:
            return out
    return {}


def build_ieee_summary(results: pd.DataFrame) -> pd.DataFrame:
    agg_cols = {
        "robust_pr_auc": ["mean", "std"],
        "robust_accuracy": ["mean", "std"],
        "adv_feasibility": ["mean", "std"],
        "adv_i_d_nonneg": ["mean", "std"],
        "flipped_positives": ["mean", "std"],
        "feasible_flipped": ["mean", "std"],
        "n_mutable_dims": ["mean"],
    }
    summary = results.groupby("attack").agg(agg_cols)
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
    summary = summary.reset_index()
    summary["filtered_success_rate"] = (
        summary["feasible_flipped_mean"] / summary["flipped_positives_mean"].replace(0, np.nan)
    )
    for attack, value in extract_ieee_backfill_map().items():
        mask = (summary["attack"] == attack) & summary["n_mutable_dims_mean"].isna()
        summary.loc[mask, "n_mutable_dims_mean"] = value
    order = ["unconstrained", "oheproj", "m_tight_oheproj", "m_oheproj", "m_wide_oheproj"]
    summary["__order"] = summary["attack"].map({key: i for i, key in enumerate(order)})
    return summary.sort_values("__order").drop(columns="__order").reset_index(drop=True)


def build_cross_gradient(results: pd.DataFrame) -> pd.DataFrame:
    agg = (
        results.groupby("dataset")
        .agg(
            clean_feas_mean=("clean_feasibility", "mean"),
            clean_feas_std=("clean_feasibility", "std"),
            adv_feas_mean=("adv_feasibility", "mean"),
            adv_feas_std=("adv_feasibility", "std"),
            clean_pr_auc_mean=("clean_pr_auc", "mean"),
            robust_pr_auc_mean=("robust_pr_auc", "mean"),
            robust_pr_auc_std=("robust_pr_auc", "std"),
        )
        .reset_index()
    )
    return agg.sort_values("adv_feas_mean").reset_index(drop=True)


def build_mask_summary(results: pd.DataFrame, feasibility: pd.DataFrame) -> pd.DataFrame:
    order = ["M0", "M1", "M2", "M3", "M4", "M5", "M6strict", "M6relaxed"]
    agg = results.groupby("variant").agg(
        {
            "n_mutable": "mean",
            "robust_pr_auc": ["mean", "std"],
            "robust_accuracy": ["mean", "std"],
            "robust_recall": ["mean", "std"],
            "robust_f1": ["mean", "std"],
        }
    )
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    agg = agg.reindex(order)
    feas_idx = feasibility.set_index("variant")
    agg["feasibility_seed42"] = feas_idx["aggregate"]
    agg["g1_pass_seed42"] = feas_idx["g1_installment"]
    agg["g4_pass_seed42"] = feas_idx["g4_ohe"]
    return agg.reset_index()


def image_integrity(figures: list[Path]) -> list[AuditFinding]:
    findings = []
    for path in figures:
        item = rel(path)
        if path.suffix.lower() == ".pdf":
            head = path.read_bytes()[:5]
            if head == b"%PDF-" and path.stat().st_size > 1024:
                findings.append(AuditFinding("PASS", item, "valid PDF", f"bytes={path.stat().st_size}"))
            else:
                findings.append(AuditFinding("FAIL", item, "invalid or tiny PDF", f"bytes={path.stat().st_size}"))
            continue
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                rgb = img.convert("RGB")
                extrema = rgb.getextrema()
                nonblank = any(low != high for low, high in extrema)
                width, height = rgb.size
            if width <= 10 or height <= 10 or not nonblank:
                findings.append(
                    AuditFinding("FAIL", item, "suspicious image size/content", f"size={width}x{height}")
                )
            else:
                findings.append(
                    AuditFinding("PASS", item, "valid nonblank image", f"size={width}x{height}; bytes={path.stat().st_size}")
                )
        except Exception as exc:  # noqa: BLE001 - audit should report every unreadable image.
            findings.append(AuditFinding("FAIL", item, "unreadable image", str(exc)))
    return findings


def provenance_index(results_dir: Path) -> dict[str, list[str]]:
    sources = [
        path
        for path in list((ROOT / "scripts").glob("*.py")) + list((ROOT / "notebooks").glob("*.ipynb"))
        if path != Path(__file__).resolve()
    ]
    figures = [
        p
        for p in results_dir.rglob("*")
        if p.suffix.lower() in FIGURE_SUFFIXES
    ]
    index: dict[str, list[str]] = {}
    for source in sources:
        try:
            if source.suffix == ".ipynb":
                nb = json.loads(source.read_text(encoding="utf-8"))
                text = "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))
            else:
                text = source.read_text(encoding="utf-8")
        except Exception:
            continue
        for fig in figures:
            if fig.name in text:
                index.setdefault(rel(fig), []).append(rel(source))
    return index


def strip_latex_comments_preserve_lines(text: str) -> str:
    """Drop unescaped LaTeX comments without changing line numbers."""
    lines = []
    for line in text.splitlines(keepends=True):
        newline = ""
        body = line
        if line.endswith("\r\n"):
            body = line[:-2]
            newline = "\r\n"
        elif line.endswith("\n"):
            body = line[:-1]
            newline = "\n"

        cut = len(body)
        for match in re.finditer("%", body):
            backslashes = 0
            idx = match.start() - 1
            while idx >= 0 and body[idx] == "\\":
                backslashes += 1
                idx -= 1
            if backslashes % 2 == 0:
                cut = match.start()
                break
        lines.append(body[:cut] + newline)
    return "".join(lines)


def collapse_latex_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def line_starts(text: str) -> list[int]:
    return [0] + [match.end() for match in re.finditer("\n", text)]


def line_no_for_index(starts: list[int], index: int) -> int:
    return bisect_right(starts, index)


def skip_latex_space(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def read_balanced_group(text: str, pos: int, opener: str = "{", closer: str = "}") -> tuple[str, int] | None:
    if pos >= len(text) or text[pos] != opener:
        return None
    depth = 0
    chars: list[str] = []
    idx = pos
    while idx < len(text):
        char = text[idx]
        escaped = idx > 0 and text[idx - 1] == "\\"
        if char == opener and not escaped:
            depth += 1
            if depth > 1:
                chars.append(char)
        elif char == closer and not escaped:
            depth -= 1
            if depth == 0:
                return "".join(chars), idx + 1
            chars.append(char)
        else:
            chars.append(char)
        idx += 1
    return None


def skip_latex_options(text: str, pos: int) -> int:
    while True:
        pos = skip_latex_space(text, pos)
        group = read_balanced_group(text, pos, "[", "]")
        if group is None:
            return pos
        _, pos = group


def parse_latex_entries(text: str, tex_file: str) -> list[TexEntry]:
    starts = line_starts(text)
    entries: list[TexEntry] = []
    for match in re.finditer(r"\\(includegraphics|caption|label)\b", text):
        kind = match.group(1)
        pos = match.end()
        if kind in {"includegraphics", "caption"}:
            pos = skip_latex_options(text, pos)
        else:
            pos = skip_latex_space(text, pos)
        group = read_balanced_group(text, pos)
        if group is None:
            continue
        value, end = group
        if kind == "label" and not value.startswith("fig:"):
            continue
        line_no = line_no_for_index(starts, match.start())
        entries.append(
            TexEntry(
                kind=kind,
                value=collapse_latex_text(value),
                tex_file=tex_file,
                line_no=line_no,
                location=f"{tex_file}:{line_no}",
                start=match.start(),
                end=end,
            )
        )
    return entries


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"\n\s*\n", text):
        end = match.start()
        if text[start:end].strip():
            spans.append((start, end))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def figure_block_span(text: str, include_start: int, include_end: int) -> tuple[int, int]:
    begins = list(re.finditer(r"\\begin\{figure\*?\}(?:\[[^\]]*\])?", text[:include_start]))
    if begins:
        begin = begins[-1]
        intervening_end = re.search(r"\\end\{figure\*?\}", text[begin.end() : include_start])
        if intervening_end is None:
            end = re.search(r"\\end\{figure\*?\}", text[include_end:])
            if end is not None:
                return begin.start(), include_end + end.end()
    for start, end in paragraph_spans(text):
        if start <= include_start < end:
            return start, end
    return include_start, include_end


def surrounding_paragraph_span(text: str, block_start: int, block_end: int) -> tuple[int, int]:
    spans = paragraph_spans(text)
    overlapping = [
        idx for idx, (start, end) in enumerate(spans) if start <= block_end and end >= block_start
    ]
    if not overlapping:
        return block_start, block_end
    first = max(0, overlapping[0] - 1)
    last = min(len(spans) - 1, overlapping[-1] + 1)
    return spans[first][0], spans[last][1]


def resolve_thesis_include(tex_root: Path, include_path: str) -> Path | None:
    clean = include_path.replace("\\", "/").strip()
    candidate = (tex_root / clean).resolve()
    if candidate.suffix:
        return candidate if candidate.exists() else None
    for suffix in [".pdf", ".png", ".jpg", ".jpeg", ".svg"]:
        with_suffix = candidate.with_suffix(suffix)
        if with_suffix.exists():
            return with_suffix
    return None


def stale_claims_in_context(
    text: str,
    context_start: int,
    starts: list[int],
    tex_file: str,
) -> list[StaleClaim]:
    claims: list[StaleClaim] = []
    for claim, pattern in STALE_CLAIM_PATTERNS:
        for match in pattern.finditer(text):
            absolute = context_start + match.start()
            line_no = line_no_for_index(starts, absolute)
            claims.append(
                StaleClaim(
                    claim=claim,
                    location=f"{tex_file}:{line_no}",
                    matched_text=collapse_latex_text(match.group(0)),
                    context=collapse_latex_text(text),
                )
            )
    return claims


def parse_tex_documents(tex_root: Path) -> TexInventory:
    tex_root = Path(tex_root).resolve()
    figures: list[ThesisFigure] = []
    captions: list[TexEntry] = []
    labels: list[TexEntry] = []
    stale_claims: list[StaleClaim] = []
    seen_claims: set[tuple[str, str, str]] = set()

    for tex in sorted(tex_root.rglob("*.tex")):
        try:
            relative = tex.relative_to(tex_root)
        except ValueError:
            continue
        if ".git" in relative.parts:
            continue
        tex_file = relative.as_posix()
        text = strip_latex_comments_preserve_lines(tex.read_text(encoding="utf-8", errors="ignore"))
        starts = line_starts(text)
        entries = parse_latex_entries(text, tex_file)
        captions.extend(entry for entry in entries if entry.kind == "caption")
        labels.extend(entry for entry in entries if entry.kind == "label")

        for include in [entry for entry in entries if entry.kind == "includegraphics"]:
            block_start, block_end = figure_block_span(text, include.start, include.end)
            block_entries = [entry for entry in entries if block_start <= entry.start <= block_end]
            caption = next((entry.value for entry in block_entries if entry.kind == "caption"), "")
            label = next((entry.value for entry in block_entries if entry.kind == "label"), "")
            context_start, context_end = surrounding_paragraph_span(text, block_start, block_end)
            context = text[context_start:context_end]
            resolved = resolve_thesis_include(tex_root, include.value)
            figures.append(
                ThesisFigure(
                    tex_file=tex_file,
                    line_no=include.line_no,
                    location=include.location,
                    include_path=include.value,
                    caption=caption,
                    label=label,
                    resolved_path=str(resolved) if resolved else None,
                    context=collapse_latex_text(context),
                )
            )
            for claim in stale_claims_in_context(context, context_start, starts, tex_file):
                key = (claim.claim, claim.location, claim.matched_text)
                if key not in seen_claims:
                    stale_claims.append(claim)
                    seen_claims.add(key)

    return TexInventory(
        tex_root=str(tex_root),
        figures=figures,
        captions=captions,
        labels=labels,
        stale_claims=stale_claims,
    )


def figure_digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def provenance_refs_for_figure(path: Path, provenance: dict[str, list[str]]) -> list[str]:
    candidate_keys = {str(path), path.as_posix(), path.name, display_path(path)}
    refs: set[str] = set()
    for key in candidate_keys:
        refs.update(provenance.get(key, []))
    return sorted(refs)


def matching_result_figures(
    thesis_figure: ThesisFigure,
    result_figures: list[Path],
    result_hashes: dict[Path, str | None],
) -> list[Path]:
    include_name = Path(thesis_figure.include_path).name
    include_suffix = Path(include_name).suffix.lower()
    resolved = Path(thesis_figure.resolved_path) if thesis_figure.resolved_path else None

    if resolved and resolved.exists():
        exact = [fig for fig in result_figures if fig.name == resolved.name]
        if exact:
            return exact
        same_stem_and_suffix = [
            fig
            for fig in result_figures
            if fig.stem == resolved.stem and fig.suffix.lower() == resolved.suffix.lower()
        ]
        if same_stem_and_suffix:
            return same_stem_and_suffix
        digest = figure_digest(resolved)
        if digest:
            hash_matches = [fig for fig in result_figures if result_hashes.get(fig) == digest]
            if hash_matches:
                return hash_matches

    if include_suffix:
        return [fig for fig in result_figures if fig.name == include_name]
    return [fig for fig in result_figures if fig.stem == include_name]


def thesis_integration_findings(
    inventory: TexInventory,
    result_figures: list[Path],
    provenance: dict[str, list[str]],
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    result_hashes = {fig: figure_digest(fig) for fig in result_figures}
    used_figures: set[Path] = set()

    for claim in inventory.stale_claims:
        findings.append(
            AuditFinding(
                "WARN",
                claim.location,
                "stale numeric claim found in thesis figure context",
                f"claim={claim.claim}; matched={claim.matched_text}",
            )
        )

    for thesis_figure in inventory.figures:
        matches = matching_result_figures(thesis_figure, result_figures, result_hashes)
        used_figures.update(matches)
        item = f"{thesis_figure.location} {thesis_figure.include_path}"
        refs: set[str] = set()
        for match in matches:
            refs.update(provenance_refs_for_figure(match, provenance))
        if refs:
            matched = ", ".join(
                f"{match.name} <- {', '.join(provenance_refs_for_figure(match, provenance))}"
                for match in matches
                if provenance_refs_for_figure(match, provenance)
            )
            findings.append(AuditFinding("PASS", item, "thesis figure has provenance entry", matched))
        elif matches:
            findings.append(
                AuditFinding(
                    "FAIL",
                    item,
                    "thesis figure has no provenance entry",
                    "matched result figure without provenance: " + ", ".join(match.name for match in matches),
                )
            )
        else:
            findings.append(
                AuditFinding(
                    "FAIL",
                    item,
                    "thesis figure has no provenance entry",
                    "include did not match any result figure",
                )
            )

    for result_figure in result_figures:
        if result_figure in used_figures:
            continue
        refs = provenance_refs_for_figure(result_figure, provenance)
        findings.append(
            AuditFinding(
                "WARN",
                display_path(result_figure),
                "result figure exists but is not used in thesis",
                "provenance=" + (", ".join(refs) if refs else "none"),
            )
        )
    return findings


def write_thesis_integration_report(
    inventory: TexInventory,
    findings: list[AuditFinding],
    output_path: Path,
    tex_root: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {status: sum(1 for finding in findings if finding.status == status) for status in ["PASS", "WARN", "FAIL"]}
    lines = [
        "# Thesis Figure Integration Audit",
        "",
        f"**TeX root:** `{Path(tex_root).resolve()}`",
        f"**Included figures:** {len(inventory.figures)}",
        f"**Captions parsed:** {len(inventory.captions)}",
        f"**Figure labels parsed:** {len(inventory.labels)}",
        f"**Passes:** {counts['PASS']}",
        f"**Warnings:** {counts['WARN']}",
        f"**Failures:** {counts['FAIL']}",
        "",
        "## Included Figures",
        "",
    ]
    if inventory.figures:
        for figure in inventory.figures:
            lines.append(f"- `{figure.location}` `{figure.include_path}`")
            if figure.label:
                lines.append(f"  Label: `{figure.label}`")
            if figure.resolved_path:
                lines.append(f"  Resolved thesis asset: `{figure.resolved_path}`")
            if figure.caption:
                lines.append(f"  Caption: {figure.caption}")
    else:
        lines.append("No `\\includegraphics` entries were found.")
    lines.append("")

    if inventory.stale_claims:
        lines.extend(["## Stale Claim Matches", ""])
        for claim in inventory.stale_claims:
            lines.append(f"- `{claim.location}` `{claim.claim}` matched `{claim.matched_text}`")
        lines.append("")

    lines.extend(["## Parsed Figure Labels", ""])
    if inventory.labels:
        for label in inventory.labels:
            lines.append(f"- `{label.location}` `{label.value}`")
    else:
        lines.append("No `\\label{fig:*}` entries were found.")
    lines.append("")

    lines.extend(["## Parsed Captions", ""])
    if inventory.captions:
        for caption in inventory.captions:
            lines.append(f"- `{caption.location}` {caption.value}")
    else:
        lines.append("No `\\caption` entries were found.")
    lines.append("")

    for status, title in [("FAIL", "Failures"), ("WARN", "Warnings"), ("PASS", "Passes")]:
        group = [finding for finding in findings if finding.status == status]
        if not group:
            continue
        lines.extend([f"## {title}", ""])
        for finding in group:
            lines.append(f"- **{finding.item}**: {finding.detail}")
            if finding.evidence:
                lines.append(f"  Evidence: {finding.evidence}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def latex_figure_inventory() -> list[str]:
    tex_files = sorted(ROOT.rglob("*.tex"))
    if not tex_files:
        return []
    records = []
    pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    for tex in tex_files:
        for line_no, line in enumerate(tex.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            match = pattern.search(line)
            if match:
                records.append(f"{rel(tex)}:{line_no}: {match.group(1)}")
    return records


def registry_csv_checks(registry: pd.DataFrame, results_dir: Path) -> list[AuditFinding]:
    checks = [
        (
            "results/figures/summary_table.csv",
            build_summary_table,
            "source=results/registry_clean.csv; filters=exclude z5/z10/eps_sweep; groupby=dataset,model_type,defence_type,attack_type,attack_epsilon",
        ),
        (
            "results/figures/input_validation_analysis.csv",
            build_input_validation,
            "source=results/registry_clean.csv; filters=epsilon=0.1, exclude z5/z10/eps_sweep; groupby=dataset,model_type,attack_type,attack_epsilon",
        ),
        (
            "results/figures/adv_training_tradeoffs.csv",
            build_adv_training,
            "source=results/registry_clean.csv; filters=exclude z5/z10/eps_sweep; groupby=dataset,model_type,attack_type,attack_epsilon",
        ),
        (
            "results/figures/statistical_tests.csv",
            build_statistical_tests,
            "source=results/registry_clean.csv; filters=epsilon=0.1, exclude z5/z10/eps_sweep; pairs=seed,attack_type,attack_epsilon",
        ),
    ]
    findings = []
    for csv_rel, builder, evidence in checks:
        actual_path = ROOT / csv_rel
        actual = pd.read_csv(actual_path)
        expected = builder(registry)
        matched, detail = frames_match(expected, actual)
        if matched:
            findings.append(AuditFinding("PASS", csv_rel, "reproduces from canonical registry", evidence))
        else:
            findings.append(AuditFinding("FAIL", csv_rel, f"does not reproduce: {detail}", evidence))
    return findings


def direct_csv_checks(results_dir: Path) -> list[AuditFinding]:
    checks = [
        (
            "results/adv_examples/g1_projection/g1_projection_summary.csv",
            build_g1_summary,
            results_dir / "adv_examples" / "g1_projection" / "g1_projection_results.csv",
            "source=g1_projection_results.csv; groupby=attack; metrics=robustness, feasibility, filtered_success_rate",
        ),
        (
            "results/adv_examples/ieee_ohe_projection/ieee_ohe_projection_summary.csv",
            build_ieee_summary,
            results_dir / "adv_examples" / "ieee_ohe_projection" / "ieee_ohe_projection_results.csv",
            "source=ieee_ohe_projection_results.csv; groupby=attack; legacy n_mutable_dims backfill from notebook output",
        ),
        (
            "results/adv_examples/cross_dataset_feasibility/cross_dataset_feasibility_gradient.csv",
            build_cross_gradient,
            results_dir / "adv_examples" / "cross_dataset_feasibility" / "cross_dataset_feasibility_results.csv",
            "source=cross_dataset_feasibility_results.csv; groupby=dataset; metrics=clean/adv feasibility and robust PR-AUC",
        ),
    ]
    findings = []
    for csv_rel, builder, source_path, evidence in checks:
        actual = pd.read_csv(ROOT / csv_rel)
        expected = builder(pd.read_csv(source_path))
        matched, detail = frames_match(expected, actual)
        if matched:
            findings.append(AuditFinding("PASS", csv_rel, "reproduces from source CSV", evidence))
            if "ieee_ohe_projection_summary.csv" in csv_rel and pd.read_csv(source_path)["n_mutable_dims"].isna().any():
                findings.append(
                    AuditFinding(
                        "WARN",
                        csv_rel,
                        "legacy n_mutable_dims blanks are backfilled from notebook output",
                        "source rows predate n_mutable_dims; summary/dose-response use notebook-recorded values",
                    )
                )
        else:
            findings.append(AuditFinding("FAIL", csv_rel, f"mismatch from source CSV: {detail}", evidence))
    return findings


def mask_and_metric_checks(results_dir: Path) -> list[AuditFinding]:
    findings = []
    try:
        actual = pd.read_csv(results_dir / "mask_ablation" / "mask_ablation_summary.csv")
        expected = build_mask_summary(
            pd.read_csv(results_dir / "mask_ablation" / "mask_ablation_results.csv"),
            pd.read_csv(results_dir / "mask_ablation" / "mask_ablation_feasibility.csv"),
        )
        matched, detail = frames_match(expected, actual)
        if matched:
            findings.append(
                AuditFinding(
                    "PASS",
                    "results/mask_ablation/mask_ablation_summary.csv",
                    "reproduces from mask result and feasibility CSVs",
                    "source=mask_ablation_results.csv,mask_ablation_feasibility.csv; groupby=variant",
                )
            )
        else:
            findings.append(
                AuditFinding("FAIL", "results/mask_ablation/mask_ablation_summary.csv", detail, "source mask CSVs")
            )
    except Exception as exc:  # noqa: BLE001
        findings.append(AuditFinding("FAIL", "results/mask_ablation/mask_ablation_summary.csv", str(exc), "source mask CSVs"))

    try:
        e1 = pd.read_csv(results_dir / "mask_ablation" / "e1_cost_summary.csv")
        base = e1[e1["cost_scale"] == 1.0].set_index("variant")
        for scale in [0.5, 2.0]:
            scaled = e1[e1["cost_scale"] == scale].set_index("variant")
            for col in ["mean", "median", "p95"]:
                if not np.allclose(scaled[col], base[col] * scale, rtol=1e-8, atol=1e-10):
                    raise AssertionError(f"{scale}x {col} is not proportional to base")
        findings.append(
            AuditFinding(
                "PASS",
                "results/mask_ablation/e1_cost_summary.csv",
                "cost scale sensitivity is internally consistent",
                "source=e1_cost_summary.csv; checks=0.5x/2.0x mean,median,p95 proportionality",
            )
        )
        findings.append(
            AuditFinding(
                "WARN",
                "results/mask_ablation/e1_cost_distribution.png; results/mask_ablation/e1_affordable_curve.png",
                "original LCLD training split for p1/p99 cost ranges is not committed; figure verification is summary-level only",
                "source=e1_cost_summary.csv; checks=scale proportionality",
            )
        )
    except Exception as exc:  # noqa: BLE001
        findings.append(AuditFinding("FAIL", "results/mask_ablation/e1_cost_summary.csv", str(exc), "source=e1_cost_summary.csv"))

    try:
        leaderboard = pd.read_csv(results_dir / "metric_analysis" / "lcld_leaderboard_reranked.csv")
        correlations = pd.read_csv(results_dir / "metric_analysis" / "rank_correlation_results.csv")
        expected_f1 = 2 * (leaderboard["precision"] * leaderboard["recall"]) / (
            leaderboard["precision"] + leaderboard["recall"] + 1e-10
        )
        if not np.allclose(expected_f1, leaderboard["f1"], rtol=1e-8, atol=1e-10):
            raise AssertionError("f1 formula mismatch")
        computed = leaderboard.copy()
        computed["score_original"] = (computed["accuracy"] * 100 + computed["adv_ctr"] * 100) / 2
        computed["score_f1"] = (computed["f1"] * 100 + computed["adv_ctr"] * 100) / 2
        computed["mcc_normalized"] = (computed["mcc"] + 1) / 2 * 100
        computed["score_mcc"] = (computed["mcc_normalized"] + computed["adv_ctr"] * 100) / 2
        computed["score_auc"] = (computed["auc"] * 100 + computed["adv_ctr"] * 100) / 2
        computed["score_harmonic"] = 2 * (
            computed["f1"] * 100 * computed["adv_ctr"] * 100
        ) / (computed["f1"] * 100 + computed["adv_ctr"] * 100 + 1e-10)
        for col in ["score_original", "score_f1", "score_mcc", "score_auc", "score_harmonic"]:
            if not np.allclose(computed[col], leaderboard[col], rtol=1e-8, atol=1e-10):
                raise AssertionError(f"{col} mismatch")
            rank_col = f"rank_{col}"
            expected_rank = computed[col].rank(ascending=False, method="min").astype(int)
            if not np.array_equal(expected_rank.values, leaderboard[rank_col].values):
                raise AssertionError(f"{rank_col} mismatch")
        rows = []
        for name, col_a, col_b in [
            ("Original vs F1", "rank_score_original", "rank_score_f1"),
            ("Original vs MCC", "rank_score_original", "rank_score_mcc"),
            ("Original vs AUC", "rank_score_original", "rank_score_auc"),
            ("Original vs Harmonic", "rank_score_original", "rank_score_harmonic"),
        ]:
            tau, p_tau = stats.kendalltau(leaderboard[col_a], leaderboard[col_b])
            rho, p_rho = stats.spearmanr(leaderboard[col_a], leaderboard[col_b])
            rows.append(
                {
                    "pair": name,
                    "kendall_tau": tau,
                    "kendall_p": p_tau,
                    "spearman_rho": rho,
                    "spearman_p": p_rho,
                }
            )
        matched, detail = frames_match(pd.DataFrame(rows), correlations)
        if not matched:
            raise AssertionError(f"rank correlations mismatch: {detail}")
        degenerate = (leaderboard["accuracy"] < 0.25) | (leaderboard["mcc"].abs() < 0.01)
        findings.append(
            AuditFinding(
                "PASS",
                "results/metric_analysis/rank_sensitivity_lcld.png; results/metric_analysis/rank_sensitivity_lcld.pdf",
                "leaderboard formulas, ranks, and correlations are internally consistent",
                f"source=lcld_leaderboard_reranked.csv,rank_correlation_results.csv; degenerate_models={int(degenerate.sum())}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        findings.append(AuditFinding("FAIL", "results/metric_analysis/rank_sensitivity_lcld", str(exc), "metric analysis CSVs"))

    try:
        comparison = pd.read_csv(results_dir / "metric_analysis" / "adv_advctr_comparison.csv")
        if set(comparison["variant"]) != {"unmasked", "masked"}:
            raise AssertionError("variants mismatch")
        if not np.allclose(comparison["gap_pp"], (comparison["adv_ctr"] - comparison["adv"]) * 100):
            raise AssertionError("gap_pp formula mismatch")
        for col in ["clean_recall", "adv", "adv_ctr", "feasible_rate"]:
            if not comparison[col].between(0, 1).all():
                raise AssertionError(f"{col} outside [0,1]")
        findings.append(
            AuditFinding(
                "PASS",
                "results/metric_analysis/adv_vs_advctr_lcld.png",
                "ADV vs ADV+CTR comparison CSV is internally consistent",
                "source=adv_advctr_comparison.csv; checks=variants,gap_pp,bounded rates",
            )
        )
    except Exception as exc:  # noqa: BLE001
        findings.append(AuditFinding("FAIL", "results/metric_analysis/adv_vs_advctr_lcld.png", str(exc), "adv_advctr_comparison.csv"))

    return findings


def core_invariant_checks(registry: pd.DataFrame) -> list[AuditFinding]:
    findings = []
    default = filter_default_analysis_rows(registry)
    excluded_names = default.get("experiment_name", pd.Series(dtype=object)).fillna("").astype(str)
    if excluded_names.str.contains("_z5|_z10|eps_sweep", regex=True).any():
        findings.append(
            AuditFinding("FAIL", "default figure filters", "z-threshold or epsilon-sweep rows remain", "registry_clean.csv")
        )
    else:
        findings.append(
            AuditFinding(
                "PASS",
                "default figure filters",
                "exclude z5/z10 and epsilon-sweep rows",
                f"rows={len(default)}; canonical_epsilon={CANONICAL_EPSILON}",
            )
        )

    agg = aggregate_seeds(default)
    bars = agg[
        agg["robust_pr_auc_mean"].notna()
        & (agg["robust_pr_auc_mean"] > 0)
        & np.isclose(agg["attack_epsilon"], CANONICAL_EPSILON)
    ]
    if bars.empty:
        findings.append(AuditFinding("FAIL", "results/figures/robustness_bars.png", "canonical source slice is empty", "registry_clean.csv"))
    else:
        findings.append(
            AuditFinding(
                "PASS",
                "results/figures/robustness_bars.png",
                "canonical epsilon source slice is non-empty",
                f"rows={len(bars)}; datasets={sorted(bars['dataset'].unique())}",
            )
        )

    attack_comparison = agg[agg["robust_pr_auc_mean"].notna() & (agg["robust_pr_auc_mean"] > 0)]
    defended = attack_comparison[attack_comparison["defence_type"] != "none"]
    if defended.empty:
        findings.append(AuditFinding("FAIL", "results/figures/defence_heatmap.png", "defended source slice is empty", "registry_clean.csv"))
    else:
        findings.append(
            AuditFinding(
                "PASS",
                "results/figures/attack_comparison.png; results/figures/defence_heatmap.png",
                "default attack and defence source slices are non-empty",
                f"attacks={sorted(attack_comparison['attack_type'].dropna().unique())}; defences={sorted(attack_comparison['defence_type'].dropna().unique())}",
            )
        )

    curve_rows = filter_robustness_curve_rows(registry)
    curve_agg = aggregate_seeds(curve_rows)
    counts = curve_agg.groupby(["dataset", "model_type", "defence_type", "attack_type"])["attack_epsilon"].nunique()
    if not (counts > 1).any():
        findings.append(AuditFinding("FAIL", "results/figures/robustness_curves.png", "no multi-epsilon neural series", "registry_clean.csv"))
    else:
        findings.append(
            AuditFinding(
                "PASS",
                "results/figures/robustness_curves.png",
                "uses neural epsilon sweeps plus canonical single-point defences",
                f"series={len(counts)}; multi_epsilon_series={int((counts > 1).sum())}",
            )
        )

    input_validation = build_input_validation(registry)
    lcld = input_validation[
        (input_validation["dataset"] == "lcld")
        & (input_validation["model_type"] == "neural")
        & (input_validation["attack_type"] == "capgd")
    ]
    if len(lcld) != 1 or not np.isclose(
        lcld["robust_prauc_baseline"].iloc[0], lcld["robust_prauc_input_val"].iloc[0], atol=5e-4
    ):
        findings.append(
            AuditFinding(
                "FAIL",
                "results/figures/input_validation_analysis.csv",
                "LCLD neural robust PR-AUC is not unchanged between baseline and input validation",
                "expected invariant tolerance=0.0005",
            )
        )
    else:
        findings.append(
            AuditFinding(
                "PASS",
                "results/figures/input_validation_analysis.csv",
                "LCLD neural robust PR-AUC is unchanged between baseline and input validation",
                f"value={lcld['robust_prauc_baseline'].iloc[0]:.4f}; tolerance=0.0005",
            )
        )

    tree_capgd = default[(default["model_type"] == "tree") & (default["attack_type"] == "capgd")]
    if tree_capgd.empty or not np.allclose(tree_capgd["clean_pr_auc"], tree_capgd["robust_pr_auc"], atol=1e-10):
        findings.append(AuditFinding("FAIL", "tree + CAPGD no-op invariant", "clean and robust PR-AUC differ", "registry_clean.csv"))
    else:
        findings.append(
            AuditFinding(
                "PASS",
                "tree + CAPGD no-op invariant",
                "clean and robust PR-AUC are identical and should be interpreted as CAPGD inapplicability",
                f"rows={len(tree_capgd)}",
            )
        )

    hsj = default[default["attack_type"] == "hopskipjump"]
    if not hsj.empty:
        coverage = hsj.groupby(["dataset", "model_type", "defence_type"])["seed"].nunique()
        incomplete = coverage[coverage < 3]
        if incomplete.empty:
            findings.append(AuditFinding("PASS", "HSJ seed coverage", "all HSJ groups have 3 seeds", f"groups={len(coverage)}"))
        else:
            findings.append(
                AuditFinding(
                    "WARN",
                    "HSJ seed coverage",
                    "HSJ must not be described as complete 3-seed coverage for every group",
                    "; ".join(f"{idx} seeds={value}" for idx, value in incomplete.items()),
                )
            )
    return findings


def write_provenance_report(figures: list[Path], provenance: dict[str, list[str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Figure Provenance Audit",
        "",
        "This report maps committed result figures to direct filename references in scripts and notebooks.",
        "",
    ]
    latex_records = latex_figure_inventory()
    if latex_records:
        lines.extend(["## LaTeX Includes", ""])
        lines.extend(f"- `{record}`" for record in latex_records)
        lines.append("")
    else:
        lines.extend(["## LaTeX Includes", "", "No `.tex` figure includes were found in this repository.", ""])
    lines.extend(["## Result Figures", ""])
    for fig in figures:
        refs = provenance.get(rel(fig), [])
        source_text = ", ".join(f"`{ref}`" for ref in sorted(refs)) if refs else "No direct reference found"
        lines.append(f"- `{rel(fig)}`: {source_text}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_audit_outputs(
    findings: list[AuditFinding],
    report_path: Path,
    csv_path: Path,
    figure_count: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [asdict(finding) for finding in findings]
    pd.DataFrame(rows, columns=["status", "item", "detail", "evidence"]).to_csv(csv_path, index=False)

    counts = {status: sum(1 for finding in findings if finding.status == status) for status in ["PASS", "WARN", "FAIL"]}
    lines = [
        "# Figure Data Audit Report",
        "",
        f"**Figures discovered:** {figure_count}",
        f"**Passes:** {counts['PASS']}",
        f"**Warnings:** {counts['WARN']}",
        f"**Failures:** {counts['FAIL']}",
        "",
    ]
    for status, title in [("FAIL", "Failures"), ("WARN", "Warnings"), ("PASS", "Passes")]:
        group = [finding for finding in findings if finding.status == status]
        if not group:
            continue
        lines.extend([f"## {title}", ""])
        for finding in group:
            lines.append(f"- **{finding.item}**: {finding.detail}")
            if finding.evidence:
                lines.append(f"  Evidence: {finding.evidence}")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(
    registry_path: Path = DEFAULT_REGISTRY,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    report_path: Path = ROOT / "docs" / "figure_data_audit_report.md",
    csv_path: Path = DEFAULT_RESULTS_DIR / "audit" / "figure_data_audit.csv",
    provenance_report_path: Path = ROOT / "docs" / "figure_provenance_audit.md",
    tex_root: Path | None = None,
    thesis_report_path: Path = THESIS_AUDIT_REPORT,
) -> tuple[int, list[AuditFinding]]:
    registry = load_registry(str(registry_path))
    figures = sorted(
        p
        for p in results_dir.rglob("*")
        if p.suffix.lower() in FIGURE_SUFFIXES
    )

    findings: list[AuditFinding] = []
    findings.extend(image_integrity(figures))

    provenance = provenance_index(results_dir)
    for fig in figures:
        refs = provenance.get(rel(fig), [])
        if refs:
            findings.append(
                AuditFinding("PASS", rel(fig), "direct filename provenance found", ", ".join(sorted(refs)))
            )
        else:
            findings.append(
                AuditFinding("WARN", rel(fig), "no direct filename reference in scripts/notebooks", "provenance search")
            )

    if tex_root is None and not latex_figure_inventory():
        findings.append(
            AuditFinding(
                "WARN",
                "LaTeX figure inventory",
                "no .tex includegraphics entries were found; audit scope is committed results figures",
                "searched repository for *.tex",
            )
        )

    findings.extend(registry_csv_checks(registry, results_dir))
    findings.extend(core_invariant_checks(registry))
    findings.extend(direct_csv_checks(results_dir))
    findings.extend(mask_and_metric_checks(results_dir))

    if tex_root is not None:
        inventory = parse_tex_documents(tex_root)
        thesis_findings = thesis_integration_findings(inventory, figures, provenance)
        findings.extend(thesis_findings)
        write_thesis_integration_report(inventory, thesis_findings, thesis_report_path, tex_root)

    write_audit_outputs(findings, report_path, csv_path, len(figures))
    write_provenance_report(figures, provenance, provenance_report_path)
    return (1 if any(finding.status == "FAIL" for finding in findings) else 0), findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit committed result figures and figure-level CSVs.")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Canonical registry CSV")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Results directory to audit")
    parser.add_argument("--report", default=str(ROOT / "docs" / "figure_data_audit_report.md"))
    parser.add_argument("--csv", default=str(DEFAULT_RESULTS_DIR / "audit" / "figure_data_audit.csv"))
    parser.add_argument("--provenance-report", default=str(ROOT / "docs" / "figure_provenance_audit.md"))
    parser.add_argument("--tex-root", default=None, help="Read-only thesis LaTeX root to audit for figure integration")
    parser.add_argument("--thesis-report", default=str(THESIS_AUDIT_REPORT))
    args = parser.parse_args()

    exit_code, findings = run_audit(
        registry_path=Path(args.registry),
        results_dir=Path(args.results_dir),
        report_path=Path(args.report),
        csv_path=Path(args.csv),
        provenance_report_path=Path(args.provenance_report),
        tex_root=Path(args.tex_root) if args.tex_root else None,
        thesis_report_path=Path(args.thesis_report),
    )
    counts = {status: sum(1 for finding in findings if finding.status == status) for status in ["PASS", "WARN", "FAIL"]}
    print(f"Figures discovered: {sum(1 for finding in findings if 'valid ' in finding.detail)}")
    print(f"Passes: {counts['PASS']}")
    print(f"Warnings: {counts['WARN']}")
    print(f"Failures: {counts['FAIL']}")
    print(f"Wrote {args.report}")
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.provenance_report}")
    if args.tex_root:
        print(f"Wrote {args.thesis_report}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
