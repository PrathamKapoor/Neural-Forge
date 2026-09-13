"""CLI runner for Phase 14 Readout/Head Bottleneck Diagnosis."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neuroforge.training.phase14_readout_diagnosis import (
    generate_phase14_markdown_report,
    run_phase14_readout_diagnosis,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 14 Readout Bottleneck Diagnosis")
    parser.add_argument("--output-dir", type=str, default="results/phase14_readout_diagnosis")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics/phase14_readout_diagnosis")
    parser.add_argument("--figures-dir", type=str, default="figures/phase14_readout_diagnosis")
    parser.add_argument("--report-path", type=str, default="results/reports/phase14_readout_diagnosis.md")
    parser.add_argument("--jointco-epochs", type=int, default=40)
    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--samples-per-type", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=60)
    args = parser.parse_args()

    print("=" * 72)
    print("NeuroForge Phase 14: Readout/Head Bottleneck Diagnosis")
    print("=" * 72)

    summary = run_phase14_readout_diagnosis(
        output_dir=args.output_dir,
        metrics_dir=args.metrics_dir,
        figures_dir=args.figures_dir,
        jointco_epochs=args.jointco_epochs,
        head_epochs=args.head_epochs,
        samples_per_type=args.samples_per_type,
        batch_size=args.batch_size,
    )

    report_path = Path(args.report_path)
    generate_phase14_markdown_report(summary, report_path)
    print(f"Report written to: {report_path}")

    print("\nPhase 14 Completed:")
    print(f"  - Metrics & Manifest: {args.metrics_dir}/")
    print(f"  - Figures: {len(summary.get('generated_figures', []))} in {args.figures_dir}")
    print(f"  - Report: {args.report_path}")
    print(f"  - Verdict: {summary.get('verdict_case', '?')} - {summary.get('verdict_label', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
