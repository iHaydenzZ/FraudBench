"""Tests for the figure data audit report writer."""

from pathlib import Path

import pandas as pd


def test_write_audit_outputs_creates_markdown_and_csv(tmp_path):
    from scripts.audit_figure_data import AuditFinding, write_audit_outputs

    findings = [
        AuditFinding(
            status="PASS",
            item="results/figures/summary_table.csv",
            detail="reproduces from results/registry_clean.csv",
            evidence="source=results/registry_clean.csv",
        ),
        AuditFinding(
            status="WARN",
            item="results/mask_ablation/e1_affordable_curve.png",
            detail="raw per-sample costs are not committed",
            evidence="summary-only check",
        ),
    ]

    report_path = tmp_path / "docs" / "figure_data_audit_report.md"
    csv_path = tmp_path / "results" / "audit" / "figure_data_audit.csv"

    write_audit_outputs(
        findings=findings,
        report_path=report_path,
        csv_path=csv_path,
        figure_count=16,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "# Figure Data Audit Report" in report
    assert "**Figures discovered:** 16" in report
    assert "**Failures:** 0" in report
    assert "results/figures/summary_table.csv" in report
    assert "raw per-sample costs are not committed" in report

    rows = pd.read_csv(csv_path)
    assert rows.to_dict("records") == [
        {
            "status": "PASS",
            "item": "results/figures/summary_table.csv",
            "detail": "reproduces from results/registry_clean.csv",
            "evidence": "source=results/registry_clean.csv",
        },
        {
            "status": "WARN",
            "item": "results/mask_ablation/e1_affordable_curve.png",
            "detail": "raw per-sample costs are not committed",
            "evidence": "summary-only check",
        },
    ]


def test_write_audit_outputs_sorts_failure_section_first(tmp_path):
    from scripts.audit_figure_data import AuditFinding, write_audit_outputs

    report_path = tmp_path / "report.md"
    csv_path = tmp_path / "audit.csv"
    write_audit_outputs(
        findings=[
            AuditFinding("PASS", "ok-item", "passed", "source=a.csv"),
            AuditFinding("FAIL", "bad-item", "mismatch", "source=b.csv"),
        ],
        report_path=report_path,
        csv_path=csv_path,
        figure_count=1,
    )

    report = report_path.read_text(encoding="utf-8")
    assert report.index("## Failures") < report.index("## Passes")
    assert "- **bad-item**: mismatch" in report
    assert Path(csv_path).exists()


def test_provenance_index_excludes_audit_script():
    from scripts.audit_figure_data import DEFAULT_RESULTS_DIR, provenance_index

    index = provenance_index(DEFAULT_RESULTS_DIR)

    assert index
    for refs in index.values():
        assert "scripts/audit_figure_data.py" not in refs


def test_parse_tex_documents_extracts_figures_and_stale_claim_context(tmp_path):
    from scripts.audit_figure_data import parse_tex_documents

    tex_root = tmp_path / "thesis"
    tex_root.mkdir()
    (tex_root / "chapter.tex").write_text(
        r"""
This paragraph describes a stale 50-iteration setting.

\begin{figure}
\includegraphics[width=\textwidth]{figures/robustness_bars}
\caption{Robustness bars for the default run.}
\label{fig:robustness-bars}
\end{figure}

The next paragraph mentions surrogate-gradient attacks.
""",
        encoding="utf-8",
    )

    inventory = parse_tex_documents(tex_root)

    assert len(inventory.figures) == 1
    figure = inventory.figures[0]
    assert figure.include_path == "figures/robustness_bars"
    assert figure.caption == "Robustness bars for the default run."
    assert figure.label == "fig:robustness-bars"
    assert figure.location == "chapter.tex:5"
    assert sorted(claim.claim for claim in inventory.stale_claims) == [
        "50-iteration",
        "surrogate-gradient",
    ]


def test_parse_tex_documents_detects_all_required_stale_claim_patterns(tmp_path):
    from scripts.audit_figure_data import parse_tex_documents

    tex_root = tmp_path / "thesis"
    tex_root.mkdir()
    (tex_root / "chapter.tex").write_text(
        r"""
The stale values are 0.083, 5000 queries, 50-iteration, two black-box
companions, three seeds per configuration, input validation reduces robust
PR-AUC on every dataset, and surrogate-gradient.

\begin{figure}
\includegraphics{figures/known}
\caption{Known figure.}
\label{fig:known}
\end{figure}
""",
        encoding="utf-8",
    )

    inventory = parse_tex_documents(tex_root)

    assert {claim.claim for claim in inventory.stale_claims} == {
        "0.083",
        "5000 queries",
        "50-iteration",
        "two black-box companions",
        "three seeds per configuration",
        "input validation reduces robust PR-AUC on every dataset",
        "surrogate-gradient",
    }


def test_thesis_integration_findings_fail_missing_provenance_and_warn_unused(tmp_path):
    from scripts.audit_figure_data import (
        AuditFinding,
        thesis_integration_findings,
        parse_tex_documents,
    )

    tex_root = tmp_path / "thesis"
    tex_root.mkdir()
    (tex_root / "chapter.tex").write_text(
        r"""
\begin{figure}
\includegraphics{figures/used_without_provenance}
\caption{Used figure.}
\label{fig:used}
\end{figure}

\begin{figure}
\includegraphics{figures/known}
\caption{Known figure.}
\label{fig:known}
\end{figure}
""",
        encoding="utf-8",
    )
    figures = [
        tmp_path / "results" / "used_without_provenance.png",
        tmp_path / "results" / "known.png",
        tmp_path / "results" / "unused.png",
    ]
    for figure in figures:
        figure.parent.mkdir(parents=True, exist_ok=True)
        figure.write_bytes(b"not-used-by-this-test")

    inventory = parse_tex_documents(tex_root)
    findings = thesis_integration_findings(
        inventory=inventory,
        result_figures=figures,
        provenance={
            str(figures[1]): ["scripts/build_known.py"],
            str(figures[2]): ["scripts/build_unused.py"],
        },
    )

    assert AuditFinding(
        "FAIL",
        "chapter.tex:3 figures/used_without_provenance",
        "thesis figure has no provenance entry",
        "matched result figure without provenance: used_without_provenance.png",
    ) in findings
    assert AuditFinding(
        "PASS",
        "chapter.tex:9 figures/known",
        "thesis figure has provenance entry",
        "known.png <- scripts/build_known.py",
    ) in findings
    assert AuditFinding(
        "WARN",
        str(figures[2]),
        "result figure exists but is not used in thesis",
        "provenance=scripts/build_unused.py",
    ) in findings


def test_thesis_integration_matches_renamed_thesis_copy_by_content_hash(tmp_path):
    from scripts.audit_figure_data import parse_tex_documents, thesis_integration_findings

    tex_root = tmp_path / "thesis"
    (tex_root / "figures").mkdir(parents=True)
    (tex_root / "chapter.tex").write_text(
        r"""
\begin{figure}
\includegraphics{figures/cross_dataset_gradient}
\caption{Renamed copy.}
\label{fig:cross-gradient}
\end{figure}
""",
        encoding="utf-8",
    )
    image_bytes = b"same image bytes"
    (tex_root / "figures" / "cross_dataset_gradient.png").write_bytes(image_bytes)

    result_figure = tmp_path / "results" / "adv_examples" / "cross_dataset_feasibility" / "gradient.png"
    result_figure.parent.mkdir(parents=True)
    result_figure.write_bytes(image_bytes)

    findings = thesis_integration_findings(
        inventory=parse_tex_documents(tex_root),
        result_figures=[result_figure],
        provenance={str(result_figure): ["notebooks/cross_dataset_feasibility.ipynb"]},
    )

    assert not [finding for finding in findings if finding.status == "FAIL"]
    assert any(
        finding.status == "PASS"
        and finding.item == "chapter.tex:3 figures/cross_dataset_gradient"
        and "gradient.png <- notebooks/cross_dataset_feasibility.ipynb" in finding.evidence
        for finding in findings
    )


def test_write_thesis_integration_report_summarizes_entries(tmp_path):
    from scripts.audit_figure_data import AuditFinding, parse_tex_documents, write_thesis_integration_report

    tex_root = tmp_path / "thesis"
    tex_root.mkdir()
    (tex_root / "chapter.tex").write_text(
        r"""
\begin{figure}
\includegraphics{figures/known}
\caption{Known figure with 0.083 stale value.}
\label{fig:known}
\end{figure}
""",
        encoding="utf-8",
    )
    inventory = parse_tex_documents(tex_root)
    report_path = tmp_path / "docs" / "thesis_figure_integration_audit.md"

    write_thesis_integration_report(
        inventory=inventory,
        findings=[
            AuditFinding("WARN", "chapter.tex:4", "stale numeric claim found", "claim=0.083"),
            AuditFinding("PASS", "chapter.tex:3 figures/known", "thesis figure has provenance entry", "known.png"),
        ],
        output_path=report_path,
        tex_root=tex_root,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "# Thesis Figure Integration Audit" in report
    assert "**TeX root:**" in report
    assert "figures/known" in report
    assert "fig:known" in report
    assert "Known figure with 0.083 stale value." in report
    assert "stale numeric claim found" in report
