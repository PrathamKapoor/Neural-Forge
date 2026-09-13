"""Train NeuroForge and write an auditable local run directory."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from neuroforge.config import NeuroForgeConfig
from neuroforge.training import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = Path(args.output or f"experiments/runs/{datetime.now():%Y%m%d-%H%M%S}")
    run_experiment(NeuroForgeConfig.from_yaml(args.config), output)
    print(f"Artifacts written to {output}")


if __name__ == "__main__": main()
