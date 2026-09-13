"""Reproduce Phase 2 controlled routing diagnosis."""
from __future__ import annotations

import argparse
from neuroforge.training import run_routing_diagnosis


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="results/metrics/controlled_routing_diagnosis")
    parser.add_argument("--epochs", type=int, default=40); args = parser.parse_args()
    run_routing_diagnosis(args.output, epochs=args.epochs)
    print(f"Artifacts written to {args.output}")


if __name__ == "__main__": main()
