"""CLI runner for Phase 19 repaired-benchmark validation (no expert training)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neuroforge.training.phase19_benchmark_validation import (
    generate_phase19_markdown_report,
    run_phase19_benchmark_validation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 19 Benchmark Repair Validation")
    parser.add_argument("--output-dir", type=str, default="results/phase19_benchmark_validation")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics/phase19_benchmark_validation")
    parser.add_argument("--figures-dir", type=str, default="figures/phase19_benchmark_validation")
    parser.add_argument("--report-path", type=str, default="results/reports/phase19_benchmark_validation.md")
    parser.add_argument("--samples-per-type", type=int, default=120)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    args = parser.parse_args()

    print("=" * 72)
    print("NeuroForge Phase 19: Mixed-Benchmark Repair Validation")
    print("=" * 72)

    summary = run_phase19_benchmark_validation(
        output_dir=args.output_dir,
        metrics_dir=args.metrics_dir,
        figures_dir=args.figures_dir,
        seeds=tuple(args.seeds),
        samples_per_type=args.samples_per_type,
    )

    report_path = Path(args.report_path)
    generate_phase19_markdown_report(summary, report_path)
    print(f"Report written to: {report_path}")
    print(f"  - Verdict: {summary.get('verdict_case', '?')} - {summary.get('verdict_label', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
