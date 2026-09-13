"""Phase 22B forensic forensics — actual computation only, no fabrication."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("results/metrics/phase22b_reproducibility_forensics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(p: Path) -> str:
    if not p.exists():
        return "MISSING"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compute_artifact_hashes() -> list[dict[str, Any]]:
    files = {
        "phase20_summary": Path("results/metrics/phase20_repaired_portfolio/summary.json"),
        "phase21_summary": Path("results/metrics/phase21_rc_diagnosis/summary.json"),
        "phase20_report": Path("results/reports/phase20_repaired_portfolio.md"),
        "phase21_report": Path("docs/research/phase21_rc_diagnosis.md"),
        "phase20_log": Path("docs/architecture/phase20_log.md"),
        "benchmark_source": Path("src/neuroforge/datasets/phase9_mixed_repaired.py"),
        "phase20_evaluator": Path("src/neuroforge/evaluation/phase20_repaired_portfolio.py"),
        "phase21_evaluator": Path("src/neuroforge/evaluation/phase21_rc_diagnosis.py"),
        "phase20_training": Path("src/neuroforge/training/phase20_repaired_portfolio.py"),
        "phase21_training": Path("src/neuroforge/training/phase21_rc_diagnosis.py"),
        "phase21_script": Path("scripts/phase21_rc_diagnosis.py"),
        "audit_script": Path("scripts/phase22_audit_provenance.py"),
        "prev_artifact_hashes": Path("results/metrics/phase22b_reproducibility_forensics/artifact_hashes.csv"),
        "prev_first_divergence": Path("results/metrics/phase22b_reproducibility_forensics/first_divergence.json"),
        "prev_metric_replay": Path("results/metrics/phase22b_reproducibility_forensics/metric_replay.csv"),
        "prev_sample_trace": Path("results/metrics/phase22b_reproducibility_forensics/sample_trace.csv"),
    }
    rows = []
    for name, p in files.items():
        exists = p.exists()
        size_bytes = p.stat().st_size if exists else 0
        h = sha256_file(p)
        rows.append({
            "artifact": name,
            "path": str(p),
            "exists": "YES" if exists else "NO",
            "size_bytes": size_bytes,
            "sha256": h,
            "note": "" if exists else "MISSING",
        })
    return rows


if __name__ == "__main__":
    rows = compute_artifact_hashes()
    out_path = RESULTS_DIR / "artifact_hashes.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["artifact", "path", "exists", "size_bytes", "sha256", "note"])
        for r in rows:
            writer.writerow([r["artifact"], r["path"], r["exists"], r["size_bytes"], r["sha256"], r["note"]])
    print(f"Wrote {len(rows)} rows to {out_path}")
