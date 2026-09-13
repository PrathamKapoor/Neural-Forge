"""CLI runner for Phase 20 clean portfolio re-evaluation (repaired benchmark)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neuroforge.training.phase20_repaired_portfolio import (
    generate_phase20_markdown_report,
    run_phase20_repaired_portfolio,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 20 Repaired Portfolio Re-evaluation")
    parser.add_argument("--output-dir", type=str, default="results/phase20_repaired_portfolio")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics/phase20_repaired_portfolio")
    parser.add_argument("--figures-dir", type=str, default="figures/phase20_repaired_portfolio")
    parser.add_argument("--report-path", type=str, default="results/reports/phase20_repaired_portfolio.md")
    parser.add_argument("--expert-epochs", type=int, default=40)
    parser.add_argument("--router-epochs", type=int, default=30)
    parser.add_argument("--samples-per-type", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--stages", type=str, nargs="+", default=["experts", "routers", "finalize"])
    args = parser.parse_args()

    print("=" * 72)
    print("NeuroForge Phase 20: Clean Portfolio Re-evaluation (repaired benchmark)")
    print("=" * 72)

    summary = run_phase20_repaired_portfolio(
        output_dir=args.output_dir,
        metrics_dir=args.metrics_dir,
        figures_dir=args.figures_dir,
        seeds=tuple(args.seeds),
        expert_epochs=args.expert_epochs,
        router_epochs=args.router_epochs,
        samples_per_type=args.samples_per_type,
        batch_size=args.batch_size,
        stages=tuple(args.stages),
    )

    if "verdict_case" in summary:
        report_path = Path(args.report_path)
        generate_phase20_markdown_report(summary, report_path)
        print(f"Report written to: {report_path}")
        print(f"  - Verdict: {summary.get('verdict_case', '?')} - {summary.get('verdict_label', '')}")
    else:
        print(f"Stages done: {summary.get('stages_done', [])}; rerun with finalize.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
