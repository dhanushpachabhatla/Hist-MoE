from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.utils.paths import MODEL_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comparison tables and failure analysis from saved experiment summaries.")
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR / "reports")
    parser.add_argument("--metadata-summary", type=Path, default=MODEL_DIR / "phase1_metadata_v1" / "summary.json")
    parser.add_argument("--image-summary", type=Path, default=MODEL_DIR / "phase1_image_v1" / "summary.json")
    parser.add_argument("--baseline-summary", type=Path, default=MODEL_DIR / "phase1_multimodal_v1" / "summary.json")
    parser.add_argument("--moe-v1-summary", type=Path, default=MODEL_DIR / "phase2_moe" / "minimal_moe_v1" / "summary.json")
    parser.add_argument("--moe-v2-summary", type=Path, default=MODEL_DIR / "phase2_moe" / "minimal_moe_v2_all_backbone" / "summary.json")
    parser.add_argument("--moe-v3-summary", type=Path, default=MODEL_DIR / "phase2_moe" / "minimal_moe_v3_image_gate" / "summary.json")
    parser.add_argument("--moe-v4-summary", type=Path, default=MODEL_DIR / "phase2_moe" / "minimal_moe_v4_3experts_image_gate" / "summary.json")
    parser.add_argument(
        "--loso-baseline-summary",
        type=Path,
        default=MODEL_DIR / "slide_validation" / "loso_baseline_multimodal" / "summary.json",
    )
    parser.add_argument(
        "--loso-moe-summary",
        type=Path,
        default=MODEL_DIR / "slide_validation" / "loso_moe_image_gate" / "summary.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def metric_from_single_run(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["best_metrics"]
    return {
        "spot_pearson": metrics["spot_pearson"],
        "gene_pearson_mean": metrics["gene_pearson_mean"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float) -> str:
    return f"{value:.4f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row[key]) for _, key in columns) + " |")
    return "\n".join([header, separator, *body])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    single_runs = [
        ("Metadata Only", args.metadata_summary),
        ("Image Only", args.image_summary),
        ("Baseline Multimodal", args.baseline_summary),
        ("MoE v1", args.moe_v1_summary),
        ("MoE v2 Full Backbone", args.moe_v2_summary),
        ("MoE v3 Image Gate", args.moe_v3_summary),
        ("MoE v4 3 Experts", args.moe_v4_summary),
    ]

    single_rows: list[dict[str, Any]] = []
    for label, path in single_runs:
        summary = load_json(path)
        metrics = metric_from_single_run(summary)
        single_rows.append(
            {
                "run": label,
                "spot_pearson": format_float(metrics["spot_pearson"]),
                "gene_pearson_mean": format_float(metrics["gene_pearson_mean"]),
                "summary_path": str(path),
            }
        )

    loso_baseline = load_json(args.loso_baseline_summary)
    loso_moe = load_json(args.loso_moe_summary)

    loso_rows = [
        {
            "model": "Baseline Multimodal",
            "mean_spot_pearson": format_float(loso_baseline["aggregate"]["mean_spot_pearson"]),
            "std_spot_pearson": format_float(loso_baseline["aggregate"]["std_spot_pearson"]),
            "mean_gene_pearson": format_float(loso_baseline["aggregate"]["mean_gene_pearson"]),
            "std_gene_pearson": format_float(loso_baseline["aggregate"]["std_gene_pearson"]),
            "summary_path": str(args.loso_baseline_summary),
        },
        {
            "model": "MoE 2 Experts Image Gate",
            "mean_spot_pearson": format_float(loso_moe["aggregate"]["mean_spot_pearson"]),
            "std_spot_pearson": format_float(loso_moe["aggregate"]["std_spot_pearson"]),
            "mean_gene_pearson": format_float(loso_moe["aggregate"]["mean_gene_pearson"]),
            "std_gene_pearson": format_float(loso_moe["aggregate"]["std_gene_pearson"]),
            "summary_path": str(args.loso_moe_summary),
        },
    ]

    baseline_per_fold = loso_baseline["aggregate"]["per_fold"]
    moe_per_fold = loso_moe["aggregate"]["per_fold"]
    shared_slides = sorted(set(baseline_per_fold) & set(moe_per_fold))

    per_slide_rows: list[dict[str, Any]] = []
    for slide in shared_slides:
        baseline_metrics = baseline_per_fold[slide]
        moe_metrics = moe_per_fold[slide]
        baseline_spot = baseline_metrics["spot_pearson"]
        moe_spot = moe_metrics["spot_pearson"]
        baseline_gene = baseline_metrics["gene_pearson_mean"]
        moe_gene = moe_metrics["gene_pearson_mean"]

        per_slide_rows.append(
            {
                "slide": slide,
                "baseline_spot_pearson": format_float(baseline_spot),
                "moe_spot_pearson": format_float(moe_spot),
                "delta_spot_pearson": format_float(moe_spot - baseline_spot),
                "baseline_gene_pearson": format_float(baseline_gene),
                "moe_gene_pearson": format_float(moe_gene),
                "delta_gene_pearson": format_float(moe_gene - baseline_gene),
                "baseline_samples": baseline_metrics["num_samples"],
                "moe_samples": moe_metrics["num_samples"],
            }
        )

    worst_baseline_slide = min(shared_slides, key=lambda slide: baseline_per_fold[slide]["spot_pearson"])
    worst_moe_slide = min(shared_slides, key=lambda slide: moe_per_fold[slide]["spot_pearson"])
    best_moe_slide = max(shared_slides, key=lambda slide: moe_per_fold[slide]["spot_pearson"])
    improved_spot_slides = sum(
        moe_per_fold[slide]["spot_pearson"] > baseline_per_fold[slide]["spot_pearson"] for slide in shared_slides
    )
    improved_gene_slides = sum(
        moe_per_fold[slide]["gene_pearson_mean"] > baseline_per_fold[slide]["gene_pearson_mean"]
        for slide in shared_slides
    )

    takeaways = [
        f"Best single-split model is MoE v3 Image Gate with spot Pearson {format_float(metric_from_single_run(load_json(args.moe_v3_summary))['spot_pearson'])} and gene Pearson {format_float(metric_from_single_run(load_json(args.moe_v3_summary))['gene_pearson_mean'])}.",
        f"Under leave-one-slide-out validation, the MoE improves mean spot Pearson from {format_float(loso_baseline['aggregate']['mean_spot_pearson'])} to {format_float(loso_moe['aggregate']['mean_spot_pearson'])}.",
        f"The MoE improves spot-wise Pearson on {improved_spot_slides}/{len(shared_slides)} held-out slides and gene-wise Pearson on {improved_gene_slides}/{len(shared_slides)} held-out slides.",
        f"The hardest held-out slide for both models is {worst_moe_slide}; baseline spot Pearson there is {format_float(baseline_per_fold[worst_moe_slide]['spot_pearson'])} and MoE spot Pearson is {format_float(moe_per_fold[worst_moe_slide]['spot_pearson'])}.",
        f"The strongest MoE held-out slide is {best_moe_slide} with spot Pearson {format_float(moe_per_fold[best_moe_slide]['spot_pearson'])}.",
    ]

    report_md = "\n\n".join(
        [
            "# histMOE Results Report",
            "## Single-Split Comparison",
            markdown_table(
                single_rows,
                [
                    ("Run", "run"),
                    ("Spot Pearson", "spot_pearson"),
                    ("Gene Pearson", "gene_pearson_mean"),
                ],
            ),
            "## Leave-One-Slide-Out Aggregate Comparison",
            markdown_table(
                loso_rows,
                [
                    ("Model", "model"),
                    ("Mean Spot Pearson", "mean_spot_pearson"),
                    ("Std Spot Pearson", "std_spot_pearson"),
                    ("Mean Gene Pearson", "mean_gene_pearson"),
                    ("Std Gene Pearson", "std_gene_pearson"),
                ],
            ),
            "## Per-Slide Failure Analysis",
            markdown_table(
                per_slide_rows,
                [
                    ("Slide", "slide"),
                    ("Baseline Spot", "baseline_spot_pearson"),
                    ("MoE Spot", "moe_spot_pearson"),
                    ("Delta Spot", "delta_spot_pearson"),
                    ("Baseline Gene", "baseline_gene_pearson"),
                    ("MoE Gene", "moe_gene_pearson"),
                    ("Delta Gene", "delta_gene_pearson"),
                ],
            ),
            "## Takeaways\n" + "\n".join(f"- {line}" for line in takeaways),
        ]
    )

    report_json = {
        "single_split": single_rows,
        "loso_aggregate": loso_rows,
        "per_slide_failure_analysis": per_slide_rows,
        "takeaways": takeaways,
        "worst_baseline_slide": worst_baseline_slide,
        "worst_moe_slide": worst_moe_slide,
    }

    write_csv(
        args.output_dir / "single_split_comparison.csv",
        single_rows,
        ["run", "spot_pearson", "gene_pearson_mean", "summary_path"],
    )
    write_csv(
        args.output_dir / "loso_comparison.csv",
        loso_rows,
        ["model", "mean_spot_pearson", "std_spot_pearson", "mean_gene_pearson", "std_gene_pearson", "summary_path"],
    )
    write_csv(
        args.output_dir / "per_slide_failure_analysis.csv",
        per_slide_rows,
        [
            "slide",
            "baseline_spot_pearson",
            "moe_spot_pearson",
            "delta_spot_pearson",
            "baseline_gene_pearson",
            "moe_gene_pearson",
            "delta_gene_pearson",
            "baseline_samples",
            "moe_samples",
        ],
    )

    (args.output_dir / "results_report.md").write_text(report_md, encoding="utf-8")
    (args.output_dir / "results_report.json").write_text(json.dumps(report_json, indent=2), encoding="utf-8")

    print(f"Saved report to: {args.output_dir / 'results_report.md'}")


if __name__ == "__main__":
    main()
